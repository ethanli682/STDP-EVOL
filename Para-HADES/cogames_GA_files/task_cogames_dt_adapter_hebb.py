"""Para-HADES task: ES optimizes Hebbian Adapter coefficients for a frozen DT.

Stage B eval harness — analogous to task_cogames_hebb.py but the gene
controls per-connection ABCD+η Hebbian coefficients for the two linear layers
of the DT Adapter module, not the standalone Hebbian MLP.

Base DT (encoder + body + heads + adapter.norm) is frozen.
Gene (5 flat arrays each of size N_SYN = 2 * adapter_dim^2):
  A, B, C, D, lr — one coefficient per synapse (fc1 + fc2 of the Adapter)

Gene layout mirrors task_cogames_hebb.py (stacked to (N_SYN, 5) for the
evaluate script):
  rows 0..adapter_dim^2-1          : fc1 synapses
  rows adapter_dim^2..2*adapter_dim^2-1 : fc2 synapses

For the default adapter_dim=256: N_SYN = 131072, total gene = 655360 floats.

Subprocess architecture (same pattern as task_cogames_hebb.py):
  - Main process: loads gene from newMember.pkl/json, saves hebb_coeffs to disk
  - Worker subprocess: loads hebb_coeffs + DT checkpoint, runs rollouts, prints score
  - Main process: parses score from stdout, applies L2 penalty, writes result

fitness_stat: composite_v7_zstat (tutorial.scout, 4 agents, bottom2_mean reduction)
so DT+Adapter and standalone-Hebbian scores are directly comparable.
"""

import os, sys, json, subprocess, signal, argparse, pickle, zstandard
import shlex
import glob
import multiprocessing as mp
import time
import tempfile
import math
import yaml
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from slurmHPCHelper import *
from GA_utils_misc_func import pickle_loads_compat


# ---------------------------------------------------------------------------
global args
global taskPath
global params_config

FAILURE_RAW_SCORE     = None
FAILURE_FITNESS_SCORE = None

# ---------------------------------------------------------------------------

def _debug_stage(args, stage):
    try:
        if getattr(args, 'debug_no_subprocess', False):
            print(
                f"DEBUG_STAGE {stage} pid={os.getpid()} agent={getattr(args,'agent_idx','NA')} "
                f"test={getattr(args,'agent_test_num','NA')} epoch={getattr(args,'epoch','NA')} "
                f"counter={getattr(args,'agent_counter','NA')}",
                flush=True,
            )
    except Exception:
        pass


def _describe_return_code(return_code):
    try:
        rc = int(return_code)
    except Exception:
        return f'exit-{return_code}'
    if rc < 0:
        sig_num = -rc
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = 'UNKNOWN'
        return f'signal-{sig_num}-{sig_name}'
    return f'exit-{rc}'


def _run_subprocess_interruptible(cmd, timeout=None):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, (stdout or ''), (stderr or '')
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
        raise
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                pass
        raise


def _pickle_loads_worker(payload, send_conn):
    try:
        obj = pickle_loads_compat(payload)
        send_conn.send(('ok', obj))
    except Exception as exc:
        send_conn.send(('err', str(exc)))
    finally:
        try:
            send_conn.close()
        except Exception:
            pass


def safe_pickle_loads_isolated(payload, timeout_sec=45):
    ctx = mp.get_context('spawn')
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_pickle_loads_worker, args=(payload, send_conn))
    proc.start()
    send_conn.close()
    status = 'err'
    data = 'unknown-error'
    try:
        if recv_conn.poll(timeout_sec):
            status, data = recv_conn.recv()
        else:
            data = f'timeout-after-{timeout_sec}s'
    except EOFError:
        data = 'worker-crashed-before-response'
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
    elif proc.exitcode not in [0, None] and status != 'ok':
        data = f'worker-exit-{proc.exitcode}; {data}'
    if status != 'ok':
        raise RuntimeError(data)
    return data


def _load_candidate_pkl_file_worker(pkl_file_path, out_pickle_path, send_conn):
    try:
        with zstandard.open(pkl_file_path, 'rb') as f:
            payload = f.read()
        obj = pickle_loads_compat(payload)
        with open(out_pickle_path, 'wb') as fout:
            pickle.dump(obj, fout, protocol=pickle.HIGHEST_PROTOCOL)
        send_conn.send(('ok', 'loaded'))
    except Exception as exc:
        send_conn.send(('err', str(exc)))
    finally:
        try:
            send_conn.close()
        except Exception:
            pass


def safe_load_candidate_pkl_isolated(pkl_file_path, timeout_sec=90):
    fd, out_pickle_path = tempfile.mkstemp(prefix='candidate_load_', suffix='.pkl')
    os.close(fd)
    ctx = mp.get_context('spawn')
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_load_candidate_pkl_file_worker,
        args=(pkl_file_path, out_pickle_path, send_conn)
    )
    proc.start()
    send_conn.close()
    status = 'err'
    data = 'unknown-error'
    try:
        if recv_conn.poll(timeout_sec):
            status, data = recv_conn.recv()
        else:
            data = f'timeout-after-{timeout_sec}s'
    except EOFError:
        data = 'worker-crashed-before-response'
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
    elif proc.exitcode not in [0, None] and status != 'ok':
        data = f'worker-exit-{proc.exitcode}; {data}'
    try:
        if status != 'ok':
            raise RuntimeError(data)
        with open(out_pickle_path, 'rb') as fin:
            return pickle.load(fin)
    finally:
        try:
            os.remove(out_pickle_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------

def run_task(param, args):
    global taskPath
    _debug_stage(args, 'run_task.start')
    worker_input  = os.path.join(
        taskPath, f'worker_in_{args.agent_idx}_{args.agent_test_num}_{os.getpid()}.pkl'
    )
    worker_output = os.path.join(
        taskPath, f'worker_out_{args.agent_idx}_{args.agent_test_num}_{os.getpid()}.pkl'
    )
    try:
        with zstandard.open(worker_input, 'wb') as f:
            f.write(pickle.dumps(param))
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--worker_mode', 'True',
            '--workerInput',  worker_input,
            '--workerOutput', worker_output,
            '--path', args.path,
            '--agent_idx',                str(args.agent_idx),
            '--agent_test_num',           str(args.agent_test_num),
            '--agent_counter',            str(args.agent_counter) + '.' + str(args.epoch),
            '--total_test_with_same_param',    str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu',    'True' if args.gpu else 'False',
            '--slurm',  'True' if args.slurm else 'False',
            '--parallel', 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', str(args.paramFile),
        ]
        # Per-candidate timeout: DT body is heavier than standalone Hebbian MLP.
        # Default eval is 10 episodes × ~2 min/episode = ~20 min; 3600s gives margin.
        return_code, stdout, stderr = _run_subprocess_interruptible(cmd, timeout=3600)
        _debug_stage(args, f'run_task.worker_returned.{_describe_return_code(return_code)}')
        if stdout:
            print('run_task worker stdout:\n' + stdout, flush=True)
        if stderr:
            print('run_task worker stderr:\n' + stderr, flush=True)
        if return_code != 0:
            _cleanup_live_lock(args)
            crash_desc = _describe_return_code(return_code)
            print(f'run_task worker failed: {crash_desc}', flush=True)
            return _fallback_outcome(args, reason=f'worker-{crash_desc}')
        if not os.path.exists(worker_output):
            _cleanup_live_lock(args)
            return _fallback_outcome(args, reason='worker-no-output')
        with zstandard.open(worker_output, 'rb') as f:
            worker_result = pickle_loads_compat(f.read())
        if not isinstance(worker_result, dict):
            _cleanup_live_lock(args)
            return _fallback_outcome(args, reason='worker-invalid-output-payload')
        return worker_result
    except subprocess.TimeoutExpired:
        _cleanup_live_lock(args)
        return _fallback_outcome(args, reason='worker-timeout')
    except KeyboardInterrupt:
        _cleanup_live_lock(args)
        print('Interrupted by user (Ctrl+C).', flush=True)
        raise
    except Exception as exc:
        _cleanup_live_lock(args)
        return _fallback_outcome(args, reason=f'worker-launch-error-{str(exc)}')
    finally:
        for f in (worker_input, worker_output):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


def _cleanup_live_lock(args):
    global taskPath
    try:
        lock_file = taskPath + '/' + str(args.agent_test_num) + '.live.lock'
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except Exception:
        pass


def _fallback_outcome(args, reason='unknown'):
    return {
        'raw_scores': [FAILURE_RAW_SCORE],
        'fitnessScore': FAILURE_FITNESS_SCORE,
        'failureReason': reason,
    }


def _quarantine_corrupt_file(path):
    try:
        if os.path.exists(path):
            corrupt_path = path + f".corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            os.replace(path, corrupt_path)
            print(f"WARNING: quarantined unreadable candidate file: {path} -> {corrupt_path}", flush=True)
            return corrupt_path
    except Exception as e:
        print(f"WARNING: failed to quarantine corrupt file {path}: {e}", flush=True)
    return None


# ---------------------------------------------------------------------------

def _run_task_impl(param, args):
    global taskPath
    import numpy as np
    import torch
    _debug_stage(args, 'task_impl.start')

    try:
        with open(taskPath + '/' + str(args.agent_test_num) + '.live.lock', 'wt') as f:
            f.write('.')
    except Exception:
        print(
            f'Problem creating lock file for agent {args.agent_idx} test {args.agent_test_num}',
            flush=True
        )
        return

    import time
    print(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()

    # =========================================================================
    # Stage B — DT + Hebbian Adapter
    # =========================================================================
    #
    # Adapter architecture (both layers are plastic):
    #   fc1: Linear(ADAPTER_DIM, ADAPTER_DIM)  no bias plasticity
    #   fc2: Linear(ADAPTER_DIM, ADAPTER_DIM)  no bias plasticity
    #   norm: LayerNorm(ADAPTER_DIM)            frozen
    #
    # Gene: 5 flat arrays (A, B, C, D, lr) each of size N_SYN = 2*ADAPTER_DIM^2
    # Stacked into (N_SYN, 5) matrix for the evaluate script.
    # Synapses: [fc1: ADAPTER_DIM^2] ++ [fc2: ADAPTER_DIM^2]
    # =========================================================================

    outP = {'raw_scores': []}

    task_dir              = os.path.dirname(os.path.abspath(__file__))
    EVAL_SCRIPT           = os.path.join(task_dir, 'evaluate_dt_adapter_hebb_cogames.py')
    MISSION               = 'tutorial.scout'
    FITNESS_STAT          = 'composite_v7_zstat'
    eval_episodes         = 10
    seed_base             = 12345
    eval_render           = False
    num_agents_override   = None
    ADAPTER_DIM           = 256
    target_rtg            = 1.0
    weight_clamp          = 3.0
    dt_checkpoint_path    = ''
    fitness_v7_weights_str = ''

    try:
        if isinstance(args.paramFile, str) and os.path.exists(args.paramFile):
            with open(args.paramFile, 'rt') as _f:
                _yaml_cfg = yaml.safe_load(_f) or {}
            _heb_cfg = ((_yaml_cfg.get('main', {}) or {}).get('HebEval', {}) or {})

            if _heb_cfg.get('eval_episodes') is not None:
                eval_episodes = int(_heb_cfg['eval_episodes'])
            if _heb_cfg.get('seed_base') is not None:
                _epoch = int(getattr(args, 'epoch', 0))
                seed_base = int(_heb_cfg['seed_base']) + _epoch * 999983
            if _heb_cfg.get('render') is not None:
                eval_render = bool(_heb_cfg['render'])
            if _heb_cfg.get('mission') is not None:
                MISSION = str(_heb_cfg['mission'])
            if _heb_cfg.get('fitness_stat') is not None:
                FITNESS_STAT = str(_heb_cfg['fitness_stat'])
            if _heb_cfg.get('num_agents') is not None:
                num_agents_override = int(_heb_cfg['num_agents'])
            if _heb_cfg.get('adapter_dim') is not None:
                ADAPTER_DIM = int(_heb_cfg['adapter_dim'])
            if _heb_cfg.get('target_rtg') is not None:
                target_rtg = float(_heb_cfg['target_rtg'])
            if _heb_cfg.get('weight_clamp') is not None:
                weight_clamp = float(_heb_cfg['weight_clamp'])
            if _heb_cfg.get('dt_checkpoint_path') is not None:
                dt_checkpoint_path = str(_heb_cfg['dt_checkpoint_path'])
            _fv7 = _heb_cfg.get('fitness_v7', None)
            if isinstance(_fv7, dict):
                _parts = []
                for _k, _v in _fv7.items():
                    if _v is None:
                        continue
                    if _k == 'reduction':
                        _parts.append(f'{_k}={str(_v)}')
                    else:
                        _parts.append(f'{_k}={float(_v)}')
                fitness_v7_weights_str = ','.join(_parts)
    except Exception as _cfg_exc:
        print(f'WARNING: failed to parse HebEval config from YAML ({_cfg_exc}); using defaults.', flush=True)

    if not dt_checkpoint_path or not os.path.exists(dt_checkpoint_path):
        reason = f'dt-checkpoint-not-found:{dt_checkpoint_path}'
        print(f'ERROR: {reason}', flush=True)
        os.system('rm -f ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
        return {'raw_scores': [FAILURE_RAW_SCORE], 'fitnessScore': FAILURE_FITNESS_SCORE,
                'failureReason': reason}

    eval_episodes = max(1, int(eval_episodes))

    # --- Derive gene dimensions from ADAPTER_DIM
    N_SYN     = 2 * ADAPTER_DIM * ADAPTER_DIM   # total synapses (fc1 + fc2)
    N_COEFFS  = 5                                # A, B, C, lr, D

    def _get_array(key):
        val = param['param'][key]
        arr = np.asarray(val, dtype=np.float32).ravel()
        if arr.size != N_SYN:
            raise ValueError(
                f"Gene key '{key}' has {arr.size} values; "
                f"expected {N_SYN} (adapter_dim={ADAPTER_DIM}, 2*dim^2)."
            )
        return arr

    A_arr  = _get_array('A')
    B_arr  = _get_array('B')
    C_arr  = _get_array('C')
    D_arr  = _get_array('D')
    lr_arr = _get_array('lr')
    _debug_stage(args, 'task_impl.gene_arrays_loaded')

    # Stack into (N_SYN, 5): cols [A, B, C, lr, D] (matches ABCD_lr_D_in convention)
    hebb_coeffs = np.stack([A_arr, B_arr, C_arr, lr_arr, D_arr], axis=1)
    assert hebb_coeffs.shape == (N_SYN, N_COEFFS), \
        f"Expected shape ({N_SYN}, {N_COEFFS}), got {hebb_coeffs.shape}"

    print(
        f'DT+Adapter task: adapter_dim={ADAPTER_DIM} N_SYN={N_SYN}  '
        f'A∈[{A_arr.min():.3f},{A_arr.max():.3f}]  '
        f'B∈[{B_arr.min():.3f},{B_arr.max():.3f}]  '
        f'C∈[{C_arr.min():.3f},{C_arr.max():.3f}]  '
        f'D∈[{D_arr.min():.3f},{D_arr.max():.3f}]  '
        f'lr∈[{lr_arr.min():.5f},{lr_arr.max():.5f}]',
        flush=True,
    )

    # Save hebb_coeffs for subprocess
    unique_suffix    = f"{os.getpid()}_{time.time_ns()}"
    hebb_coeffs_path = os.path.join(
        taskPath,
        f'heb_coeffs_dt_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.dat'
    )
    sidecar_path_v7 = os.path.join(
        taskPath,
        f'components_v7_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.json'
    )
    torch.save(hebb_coeffs, hebb_coeffs_path)
    _debug_stage(args, 'task_impl.hebb_saved')

    # Build evaluate subprocess command
    cmd = [
        sys.executable, EVAL_SCRIPT,
        '--hebb_coeffs_path',   hebb_coeffs_path,
        '--dt_checkpoint_path', dt_checkpoint_path,
        '--mission',            MISSION,
        '--fitness_stat',       FITNESS_STAT,
        '--eval_episodes',      str(eval_episodes),
        '--seed_base',          str(seed_base),
        '--render',             'True' if eval_render else 'False',
        '--target_rtg',         str(target_rtg),
        '--weight_clamp',       str(weight_clamp),
        '--adapter_dim',        str(ADAPTER_DIM),
    ]
    if num_agents_override is not None:
        cmd += ['--num_agents', str(num_agents_override)]
    if fitness_v7_weights_str:
        cmd += ['--fitness_v7_weights', fitness_v7_weights_str]
    if FITNESS_STAT == 'composite_v7_zstat':
        cmd += ['--components_sidecar', sidecar_path_v7]

    print(f'evaluate_dt_adapter_hebb_script={EVAL_SCRIPT}', flush=True)
    print('Running: ' + ' '.join(cmd), flush=True)

    raw_score      = None
    ep_reward_mean = None
    failure_reason = None

    try:
        _debug_stage(args, 'task_impl.evaluate_dt.start')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3500)
        _debug_stage(args, f'task_impl.evaluate_dt.returncode.{result.returncode}')
        stdout = result.stdout
        stderr = result.stderr
        print('evaluate_dt_adapter_hebb stdout:\n' + stdout, flush=True)
        if stderr:
            print('evaluate_dt_adapter_hebb stderr:\n' + stderr, flush=True)

        if result.returncode != 0:
            print(f'WARNING: evaluate_dt_adapter_hebb.py exited with {result.returncode}. Failure penalty.', flush=True)
            raw_score = FAILURE_RAW_SCORE
            failure_reason = f'evaluate-dt-return-code-{result.returncode}'
        else:
            import re
            patterns = [
                r'[Ee]pisode\s+cumulative\s+rewards?\s+(-?[\d.]+)',
                r'[Ff]itness[:\s]+(-?[\d.]+)',
                r'[Ss]core[:\s]+(-?[\d.]+)',
                r'[Rr]eward[:\s]+(-?[\d.]+)',
                r'^(-?[\d.]+)\s*$',
            ]
            for line in reversed(stdout.strip().splitlines()):
                for pat in patterns:
                    m = re.search(pat, line.strip())
                    if m:
                        candidate = float(m.group(1))
                        if math.isfinite(candidate):
                            raw_score = candidate
                        break
                if raw_score is not None:
                    break

            if raw_score is None:
                print('WARNING: Could not parse finite score. Failure penalty.', flush=True)
                raw_score = FAILURE_RAW_SCORE
                failure_reason = 'evaluate-dt-score-parse-failed'

            _ep_vals = []
            for _ln in stdout.splitlines():
                if 'v7:' in _ln:
                    _m = re.search(r'\bep_reward=\s*(-?[\d.eE+-]+)', _ln)
                    if _m:
                        try:
                            _v = float(_m.group(1))
                            if math.isfinite(_v):
                                _ep_vals.append(_v)
                        except ValueError:
                            pass
            if _ep_vals:
                ep_reward_mean = sum(_ep_vals) / len(_ep_vals)

    except subprocess.TimeoutExpired:
        print('WARNING: evaluate_dt_adapter_hebb.py timed out. Failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = 'evaluate-dt-timeout'
    except Exception as _e:
        print(f'WARNING: evaluate_dt_adapter_hebb.py exception: {_e}. Failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = f'evaluate-dt-exception-{_e}'
    finally:
        try:
            os.remove(hebb_coeffs_path)
        except Exception:
            pass

    components_v7 = None
    if FITNESS_STAT == 'composite_v7_zstat' and os.path.exists(sidecar_path_v7):
        try:
            with open(sidecar_path_v7, 'rt') as _scf:
                components_v7 = json.load(_scf)
        except Exception as _sc_exc:
            print(f'WARNING: failed to load v7 sidecar ({_sc_exc}).', flush=True)
        finally:
            try:
                os.remove(sidecar_path_v7)
            except Exception:
                pass

    outP['raw_scores'].append(raw_score)
    outP['ep_reward_mean'] = ep_reward_mean
    if components_v7 is not None:
        outP['components_v7'] = components_v7

    print(f'Raw DT+Adapter score: {raw_score}  ep_reward_mean: {ep_reward_mean}', flush=True)

    if failure_reason is not None:
        outP['fitnessScore'] = FAILURE_FITNESS_SCORE
        outP['failureReason'] = failure_reason
    else:
        # L2 penalty: same scale as task_cogames_hebb.py so selection pressure
        # on coefficients is comparable across the two variants.
        GENE_ABS_MAX = 10.0
        l2_penalty = -0.01 * np.mean(hebb_coeffs ** 2) / (GENE_ABS_MAX ** 2)
        penalized_score = raw_score + l2_penalty
        print(f'L2 penalty: {l2_penalty:.6f}  penalized score: {penalized_score:.4f}', flush=True)

        if args.evolutionTarget == -1:
            outP['fitnessScore'] = float(-1.0 * penalized_score)
        else:
            outP['fitnessScore'] = float(penalized_score)

    print(f'---- End Task ----- {datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    print(f'---- Task Total Time ----- {(time.time() - start_time)} sec', flush=True)

    os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
    return outP


# ---------------------------------------------------------------------------

def main(args):
    global taskPath, params_config

    taskPath = args.path + '/running/' + str(args.agent_idx)
    os.makedirs(taskPath, exist_ok=True)
    _debug_stage(args, 'main.start')

    json_member_path = args.path + '/running/' + str(args.agent_idx) + '.newMember.json'
    pkl_member_path  = args.path + '/running/' + str(args.agent_idx) + '.newMember.pkl'

    try:
        with open(json_member_path, 'rt') as f:
            params_config = json.load(f)
        format = 'json'
    except Exception as json_exc:
        pkl_exc = None
        for attempt in range(3):
            try:
                params_config = safe_load_candidate_pkl_isolated(pkl_member_path, timeout_sec=120)
                format = 'pkl'
                pkl_exc = None
                break
            except Exception as e:
                pkl_exc = e
                time.sleep(1.0)
        if pkl_exc is not None:
            _quarantine_corrupt_file(pkl_member_path)
            if args.slurm:
                print(f'INFO: unreadable newMember for agent={args.agent_idx}; exiting non-zero.', flush=True)
                sys.exit(3)
            raise RuntimeError(
                f'Failed to load newMember for agent={args.agent_idx}. '
                f'json_error={json_exc}; pkl_error={pkl_exc}'
            )

    if not isinstance(params_config, list):
        raise RuntimeError(f'Loaded newMember has invalid type: {type(params_config)}')

    _debug_stage(args, f'main.members_loaded.{format}.count_{len(params_config)}')

    for gene_num in range(len(params_config)):
        fitness      = []
        raw_scores   = []
        ep_reward_means  = []
        failure_reasons  = []
        components_v7_per_test = []

        for test in range(args.total_test_with_same_param):
            outP = run_task(params_config[gene_num], args)
            if not isinstance(outP, dict):
                outP = _fallback_outcome(args, reason='main-nondict-task-output')

            try:
                score = float(outP.get('fitnessScore', FAILURE_FITNESS_SCORE))
                if not math.isfinite(score):
                    raise ValueError('non-finite score')
            except Exception:
                score = FAILURE_FITNESS_SCORE
                failure_reasons.append('main-invalid-fitnessscore')
            fitness.append(score)

            test_raw = outP.get('raw_scores', [])
            raw_scores.extend(test_raw if isinstance(test_raw, list) else [test_raw])

            _er = outP.get('ep_reward_mean', None)
            if _er is not None:
                try:
                    _er_f = float(_er)
                    if math.isfinite(_er_f):
                        ep_reward_means.append(_er_f)
                except (TypeError, ValueError):
                    pass

            reason = outP.get('failureReason', None)
            if reason not in [None, '', 'None']:
                failure_reasons.append(str(reason))

            _cv7 = outP.get('components_v7', None)
            if _cv7 is not None:
                components_v7_per_test.append(_cv7)

        params_config[gene_num]['fitnessScore'] = fitness
        params_config[gene_num]['raw_scores']   = raw_scores
        if ep_reward_means:
            import numpy as np
            params_config[gene_num]['ep_reward_mean'] = float(np.mean(ep_reward_means))
        if failure_reasons:
            params_config[gene_num]['failureReason'] = sorted(set(failure_reasons))
        if components_v7_per_test:
            params_config[gene_num]['components_v7_per_test'] = components_v7_per_test

    _debug_stage(args, 'main.write_results.start')
    if format == 'json':
        with open(taskPath + '/' + str(args.agent_test_num) + '.resoult.json', 'wt') as f:
            json.dump(params_config, f)
    else:
        with zstandard.open(taskPath + '/' + str(args.agent_test_num) + '.resoult.pkl', 'wb') as f:
            f.write(pickle.dumps(params_config))

    if args.parallel or args.slurm:
        _debug_stage(args, 'main.callback.schedule')
        run(args, self=False, params=params_config[-1].copy())

    os.system('rm -f ' + shlex.quote(args.path + '/running/' + str(args.agent_idx) + '.newMember.json'))
    os.system('rm -f ' + shlex.quote(args.path + '/running/' + str(args.agent_idx) + '.newMember.pkl'))
    print('Done Task: ' + str(args.agent_test_num) + ' for agent: ' + str(args.agent_idx), flush=True)


def run(args, self=False, params=None):
    args.agent_counter = str(args.agent_counter) + '.' + str(args.epoch)

    if self:
        run_process = [
            __file__,
            '--callBackScript', '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"', ''),
            '--path', args.path,
            '--agent_idx', str(args.agent_idx),
            '--agent_test_num', str(args.agent_test_num),
            '--agent_counter', str(args.agent_counter),
            '--total_test_with_same_param', str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'True' if args.parallel else 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
        ]
    else:
        run_process = [
            '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"', ''),
            '--part', '2',
            '--taskFilename', '"' + __file__ + '"',
            '--path', args.path,
            '--agent_idx', str(args.agent_idx),
            '--agent_test_num', str(args.agent_test_num),
            '--agent_counter', str(args.agent_counter),
            '--total_test_with_same_param', str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
        ]

    if args.slurm:
        os.makedirs(args.path + '/scripts/', exist_ok=True)
        temp_file = (
            args.path + '/scripts/Agent_idx_' + str(args.agent_idx) +
            '_Task_' + str(args.agent_test_num) +
            '_Epoch_' + str(args.epoch) + '.sh'
        )
        job_name = args.path.split('/')[-1] or args.path.split('/')[-2]
        script = SlurmScript(
            jobName=('T_' if self else 'A_') + job_name,
            jobTime=params['main']['Task_resource']['time'] if self else params['main']['Main_resource']['time'],
            jobMemory=params['main']['Task_resource']['memory'] if self else params['main']['Main_resource']['memory'],
            output_path=args.path + (params['main']['Task_resource']['path'] if self else params['main']['Main_resource']['path']),
            jobCPUs=params['main']['Task_resource']['cpu'] if self else params['main']['Main_resource']['cpu'],
            jobGPUs=params['main']['Task_resource']['gpu'] if self else params['main']['Main_resource']['gpu'],
            excludeNodes=params['main']['Task_resource']['exclude_nodes'] if self else params['main']['Main_resource']['exclude_nodes'],
            condaEnv=params['main']['Task_resource']['conda_env'] if self else params['main']['Main_resource']['conda_env'],
            partition=params['main']['Task_resource']['partition'] if self else params['main']['Main_resource']['partition'],
            singularity_image=params['main']['Task_resource']['singularity_image'] if self else params['main']['Main_resource']['singularity_image'],
            command='python ' + str(run_process).replace('[', '').replace(']', '').replace('\', \'', ' ').replace('\'', ''),
            nextTask=temp_file,
        )
        with open(temp_file, 'w') as file:
            file.write(script)
            file.flush()
    elif args.parallel:
        run_process = str(['python3'] + run_process).replace('[', '').replace(']', '').replace('\'', '').replace(',', ' ')
        subprocess.Popen(run_process, shell=True)


def signal_handler_SIGTERM(sig, frame):
    global args, taskPath, params_config
    os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
    run(args=argparse._copy_items(args), self=True, params=params_config.copy())
    print('got a SIGTERM!!', flush=True)
    sys.exit(0)


if __name__ == '__main__':
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',                       type=str, default='FindSeq')
    parser.add_argument('--agent_test_num',             type=int, default=-1)
    parser.add_argument('--agent_idx',                  type=int, default=-1)
    parser.add_argument('--agent_counter',              type=str, default='-1.0')
    parser.add_argument('--total_test_with_same_param', type=int, default=1)
    parser.add_argument('--total_testsGene_with_same_agent', type=int, default=1)
    parser.add_argument('--gpu',         type=str, default='False')
    parser.add_argument('--slurm',       type=str, default='False')
    parser.add_argument('--parallel',    type=str, default='True')
    parser.add_argument('--callBackScript', type=str, default=None)
    parser.add_argument('--evolutionTarget', type=int, default=1)
    parser.add_argument('--paramFile',   type=str, default=None)
    parser.add_argument('--worker_mode', type=str, default='False')
    parser.add_argument('--workerInput', type=str, default=None)
    parser.add_argument('--workerOutput', type=str, default=None)
    parser.add_argument('--main_worker_mode', type=str, default='False')
    parser.add_argument('--debug_no_subprocess', type=str, default='False')

    args = parser.parse_args()
    args.gpu         = args.gpu in ('True', 'true')
    args.slurm       = args.slurm in ('True', 'true')
    args.parallel    = args.parallel in ('True', 'true')
    args.worker_mode = args.worker_mode in ('True', 'true')
    args.main_worker_mode   = args.main_worker_mode in ('True', 'true')
    args.debug_no_subprocess = args.debug_no_subprocess in ('True', 'true')
    _debug_stage(args, 'entry.args_parsed')

    if isinstance(args.paramFile, str) and args.paramFile.strip().lower() in ('', 'none', 'null'):
        args.paramFile = None
    if args.paramFile is None:
        task_root, _ = os.path.splitext(os.path.abspath(__file__))
        default_yaml = task_root + '.yaml'
        default_yml  = task_root + '.yml'
        args.paramFile = default_yaml
        if not os.path.exists(default_yaml) and os.path.exists(default_yml):
            args.paramFile = default_yml

    if args.parallel:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS']      = '1'
        os.environ['NUMEXPR_NUM_THREADS']  = '1'
        os.environ['OMP_NUM_THREADS']      = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    if args.slurm:
        signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
        signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
        signal.signal(signal.SIGINT,  signal_handler_SIGTERM)

    if args.callBackScript is not None:
        args.callBackScript = args.callBackScript.replace('"', '')

    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.worker_mode:
        _debug_stage(args, 'entry.worker_mode.start')
        taskPath = args.path + '/running/' + str(args.agent_idx)
        os.makedirs(taskPath, exist_ok=True)
        try:
            with zstandard.open(args.workerInput, 'rb') as f:
                worker_param = pickle_loads_compat(f.read())
            worker_out = _run_task_impl(worker_param, argparse._copy_items(args))
            with zstandard.open(args.workerOutput, 'wb') as f:
                f.write(pickle.dumps(worker_out))
            sys.exit(0)
        except Exception as exc:
            print(f'Worker mode failure: {exc}', flush=True)
            sys.exit(2)

    if args.slurm and not args.main_worker_mode and not args.debug_no_subprocess:
        _debug_stage(args, 'entry.main_worker_wrapper.start')
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--main_worker_mode', 'True',
            '--path', args.path,
            '--agent_test_num', str(args.agent_test_num),
            '--agent_idx', str(args.agent_idx),
            '--agent_counter', str(args.agent_counter) + '.' + str(args.epoch),
            '--total_test_with_same_param', str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'False',
            '--callBackScript', str(args.callBackScript),
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', str(args.paramFile),
            '--worker_mode', 'False',
            '--debug_no_subprocess', 'False',
        ]
        return_code, stdout, stderr = _run_subprocess_interruptible(cmd, timeout=None)
        if stdout:
            print('main worker stdout:\n' + stdout, flush=True)
        if stderr:
            print('main worker stderr:\n' + stderr, flush=True)
        if return_code != 0:
            crash_desc = _describe_return_code(return_code)
            print(f'WARNING: main worker crashed ({crash_desc}).', flush=True)
        sys.exit(0)

    if args.slurm:
        args.parallel = False

    # Resolve agent_idx
    running_dir = os.path.join(args.path, 'running')
    expected_json = os.path.join(running_dir, str(args.agent_idx) + '.newMember.json')
    expected_pkl  = os.path.join(running_dir, str(args.agent_idx) + '.newMember.pkl')
    if not (os.path.exists(expected_json) or os.path.exists(expected_pkl)):
        msg = f'No work for agent {args.agent_idx}: no *.newMember.(json|pkl) found in {running_dir}'
        if args.slurm:
            print(f'INFO: {msg}; exiting non-zero.', flush=True)
            sys.exit(3)
        print(f'ERROR: {msg}', flush=True)
        sys.exit(2)

    main(argparse._copy_items(args))
