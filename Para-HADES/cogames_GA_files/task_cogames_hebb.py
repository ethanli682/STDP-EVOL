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


#--------------------
global args
global taskPath
global params_config


FAILURE_RAW_SCORE = None
FAILURE_FITNESS_SCORE = None


def _debug_stage(args, stage):
    try:
        if getattr(args, 'debug_no_subprocess', False):
            print(
                f"DEBUG_STAGE {stage} pid={os.getpid()} agent={getattr(args, 'agent_idx', 'NA')} test={getattr(args, 'agent_test_num', 'NA')} epoch={getattr(args, 'epoch', 'NA')} counter={getattr(args, 'agent_counter', 'NA')}",
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
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
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
    proc = ctx.Process(target=_load_candidate_pkl_file_worker, args=(pkl_file_path, out_pickle_path, send_conn))
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

def run_task(param, args):
    global taskPath
    _debug_stage(args, 'run_task.start')
    worker_input = os.path.join(taskPath, f'worker_in_{args.agent_idx}_{args.agent_test_num}_{os.getpid()}.pkl')
    worker_output = os.path.join(taskPath, f'worker_out_{args.agent_idx}_{args.agent_test_num}_{os.getpid()}.pkl')

    try:
        with zstandard.open(worker_input, 'wb') as f:
            f.write(pickle.dumps(param))
        _debug_stage(args, 'run_task.worker_input_written')

        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            '--worker_mode', 'True',
            '--workerInput', worker_input,
            '--workerOutput', worker_output,
            '--path', args.path,
            '--agent_idx', str(args.agent_idx),
            '--agent_test_num', str(args.agent_test_num),
            '--agent_counter', str(args.agent_counter) + '.' + str(args.epoch),
            '--total_test_with_same_param', str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', str(args.paramFile),
        ]

        return_code, stdout, stderr = _run_subprocess_interruptible(cmd, timeout=930)
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
        try:
            if os.path.exists(worker_input):
                os.remove(worker_input)
        except Exception:
            pass
        try:
            if os.path.exists(worker_output):
                os.remove(worker_output)
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


def _load_params_config_for_agent(args):

    running_dir = os.path.join(args.path, "running")
    print(f"[DEBUG] Listing files in {running_dir} before loading candidate:", flush=True)
    try:
        for fname in os.listdir(running_dir):
            fpath = os.path.join(running_dir, fname)
            try:
                stat = os.stat(fpath)
                print(f"[DEBUG]   {fname} size={stat.st_size} mtime={stat.st_mtime}", flush=True)
            except Exception as e:
                print(f"[DEBUG]   {fname} (stat error: {e})", flush=True)
    except Exception as e:
        print(f"[DEBUG] Could not list directory {running_dir}: {e}", flush=True)

    json_file = args.path + "/running/" + str(args.agent_idx) + '.newMember.json'
    pkl_file = args.path + "/running/" + str(args.agent_idx) + '.newMember.pkl'

    if os.path.exists(json_file):
        with open(json_file, 'rt') as f:
            return json.load(f), 'json'

    with zstandard.open(pkl_file, 'rb') as f:
        return pickle_loads_compat(f.read()), 'pkl'


def _resolve_agent_idx_for_existing_member(args):
    running_dir = args.path + "/running"
    expected_json = os.path.join(running_dir, str(args.agent_idx) + '.newMember.json')
    expected_pkl = os.path.join(running_dir, str(args.agent_idx) + '.newMember.pkl')

    if os.path.exists(expected_json) or os.path.exists(expected_pkl):
        return args.agent_idx

    candidates = glob.glob(os.path.join(running_dir, '*.newMember.json')) + glob.glob(os.path.join(running_dir, '*.newMember.pkl'))
    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    newest = os.path.basename(candidates[0])
    try:
        return int(newest.split('.newMember.')[0])
    except Exception:
        return None


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


def _write_fallback_results_and_continue(args, reason):
    global taskPath
    _debug_stage(args, f'fallback.start.{reason}')
    effective_agent_idx = args.agent_idx
    running_dir = args.path + "/running"
    expected_json = os.path.join(running_dir, str(args.agent_idx) + '.newMember.json')
    expected_pkl = os.path.join(running_dir, str(args.agent_idx) + '.newMember.pkl')

    if (not os.path.exists(expected_json)) and (not os.path.exists(expected_pkl)):
        # Our own candidate is gone. NEVER adopt another agent's newest candidate:
        # the old code did exactly that and then wrote a failure result into, and
        # deleted, a different (healthy) agent's files -- corrupting the GA and
        # cascading. With no candidate of our own there is nothing legitimate to
        # fall back for, so leave every other agent's files untouched and bail out.
        print(f'ERROR: no newMember for requested agent {args.agent_idx}; refusing to adopt '
              f"another agent's candidate. No fallback written.", flush=True)
        return False

    taskPath = args.path + "/running/" + str(effective_agent_idx)
    os.makedirs(taskPath, exist_ok=True)
    src_json = args.path + "/running/" + str(effective_agent_idx) + '.newMember.json'
    src_pkl = args.path + "/running/" + str(effective_agent_idx) + '.newMember.pkl'
    dst_json = taskPath + '/' + str(args.agent_test_num)  + '.resoult.json'
    dst_pkl = taskPath + '/' + str(args.agent_test_num)  + '.resoult.pkl'

    has_any_source = os.path.exists(src_json) or os.path.exists(src_pkl)
    if not has_any_source:
        print('ERROR: no newMember file found for fallback copy.', flush=True)
        return False

    fallback_payload = _fallback_outcome(args, reason=reason)
    fallback_entry = {
        'fitnessScore': [fallback_payload['fitnessScore']],
        'raw_scores': fallback_payload['raw_scores'],
        'failureReason': reason,
        'loss': None,
        'build_in_param': {},
    }
    fallback_list = [fallback_entry.copy() for _ in range(max(1, int(args.total_testsGene_with_same_agent)))]

    try:
        with open(dst_json, 'wt') as f:
            json.dump(fallback_list, f)
        print(f'Fallback wrote minimal json report: {dst_json}', flush=True)
    except Exception as e:
        print(f'ERROR: fallback minimal json write failed: {e}', flush=True)
        return False

    if os.path.exists(src_pkl):
        try:
            with zstandard.open(dst_pkl, 'wb') as f:
                f.write(pickle.dumps(fallback_list, protocol=pickle.HIGHEST_PROTOCOL))
            print(f'Fallback wrote synthetic safe pickle report: {dst_pkl}', flush=True)
        except Exception as e:
            print(f'ERROR: fallback synthetic pkl write failed: {e}', flush=True)
            return False

    callback_agent_idx = effective_agent_idx
    report_files_for_agent = glob.glob(os.path.join(taskPath, '*.resoult.json')) + glob.glob(os.path.join(taskPath, '*.resoult.pkl'))
    if len(report_files_for_agent) == 0:
        global_report_files = glob.glob(os.path.join(running_dir, '*', '*.resoult.json')) + glob.glob(os.path.join(running_dir, '*', '*.resoult.pkl'))
        if len(global_report_files) > 0:
            global_report_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            newest_report = global_report_files[0]
            try:
                callback_agent_idx = int(os.path.basename(os.path.dirname(newest_report)))
                print(f'Fallback callback agent adjusted from {effective_agent_idx} to {callback_agent_idx} based on report file: {newest_report}', flush=True)
            except Exception:
                pass
    else:
        print(f'Fallback created report files for agent {effective_agent_idx}: {report_files_for_agent}', flush=True)

    callback_script = (args.callBackScript or '').replace('"', '')
    if callback_script != '':
        if args.slurm:
            print(
                'Fallback: SLURM mode detected; skipping inline GA callback execution. '
                'Reports were written and control returns to scheduler.',
                flush=True,
            )
        else:
            callback_cmd = (
                'python3 ' + shlex.quote(callback_script) +
                ' --part 2' +
                ' --taskFilename ' + shlex.quote(os.path.abspath(__file__)) +
                ' --path ' + shlex.quote(args.path) +
                ' --agent_idx ' + shlex.quote(str(callback_agent_idx)) +
                ' --agent_test_num ' + shlex.quote(str(args.agent_test_num)) +
                ' --agent_counter ' + shlex.quote(str(args.agent_counter) + '.' + str(args.epoch)) +
                ' --total_test_with_same_param ' + shlex.quote(str(args.total_test_with_same_param)) +
                ' --total_testsGene_with_same_agent ' + shlex.quote(str(args.total_testsGene_with_same_agent)) +
                ' --gpu ' + shlex.quote('True' if args.gpu else 'False') +
                ' --slurm ' + shlex.quote('False') +
                ' --parallel ' + shlex.quote('False') +
                ' --evolutionTarget ' + shlex.quote(str(args.evolutionTarget)) +
                ' --paramFile ' + shlex.quote(str(args.paramFile))
            )
            callback_rc = os.system(callback_cmd)
            if callback_rc != 0:
                print('WARNING: callback command failed in fallback flow.', flush=True)
    else:
        print('WARNING: fallback flow skipped callback because callBackScript is empty.', flush=True)

    # NOTE: deliberately do NOT delete the .newMember candidate here. The old code
    # removed it (and, via the remap above, other agents' candidates), which is what
    # left requeued jobs with "no candidate" and triggered the cascade. Leaving the
    # candidate in place lets the framework requeue/redispatch this agent if needed.

    print('Fallback result flow completed for crashed main worker.', flush=True)
    _debug_stage(args, 'fallback.done')
    return True


def _run_task_impl(param, args):
    global taskPath
    import numpy as np
    _debug_stage(args, 'task_impl.start')

    # show I am alive!
    try:
        with open(taskPath + '/' + str(args.agent_test_num) + '.live.lock', 'wt') as f:
            f.write('.')
    except:
        print('Problem creating agent ' + str(args.agent_idx) + ', task number' + str(args.agent_test_num) + ' lock file: ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock', flush=True)
        return

    import time
    print(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()

    #------------------------Task Start--------------------------------------

    # =========================================================================
    # HebbianMetaLearning  –  CoGames tutorial.scout, ABCD_lr
    # =========================================================================
    #
    # Network architecture (all FC layers are plastic; no CNN):
    #   Layer 1:  900 (300 tokens × 3)  →   64   →  57,600 synapses
    #   Layer 2:   64                   →   32   →   2,048 synapses
    #   Layer 3:   32                   →    5   →     160 synapses
    #   TOTAL:                               →  59,808 synapses
    #
    # The YAML evolves 5 flat arrays each of length 59,808:
    #   param['param']['A']  – shape (59808,)
    #   param['param']['B']  – shape (59808,)
    #   param['param']['C']  – shape (59808,)
    #   param['param']['D']  – shape (59808,)
    #   param['param']['lr'] – shape (59808,)
    # =========================================================================

    outP = dict()
    outP['raw_scores'] = []

    # --- Path configuration --------------------------------------------------
    task_dir = os.path.dirname(os.path.abspath(__file__))
    HEBB_EVAL_SCRIPT = os.path.join(task_dir, 'evaluate_hebb_cogames.py')
    HEBB_RULE        = 'ABCD_lr'
    MISSION          = 'tutorial.scout'
    INIT_WEIGHTS     = 'ka_uni'
    FITNESS_STAT     = 'episode_reward'
    eval_episodes    = 3
    seed_base        = 12345
    eval_render      = False
    num_agents_override = None
    # Observation pipeline defaults — overridable via the HebEval YAML block.
    # Hebbian always runs single-frame legacy_flat; the encoder optionally
    # projects the raw 900-dim observation to encoder_dim before the MLP.
    obs_encoder      = 'none'    # 'none' | 'random_proj'
    encoder_dim      = 64
    encoder_seed     = 1337
    fitness_v4_weights_str = ''
    fitness_v5_weights_str = ''
    fitness_v7_weights_str = ''
    fitness_v8_weights_str = ''
    max_steps_override = None

    # Optional overrides from YAML (main.HebEval)
    try:
        if isinstance(args.paramFile, str) and os.path.exists(args.paramFile):
            with open(args.paramFile, 'rt') as _f:
                _yaml_cfg = yaml.safe_load(_f) or {}
            _heb_eval_cfg = ((_yaml_cfg.get('main', {}) or {}).get('HebEval', {}) or {})
            if 'eval_episodes' in _heb_eval_cfg and _heb_eval_cfg['eval_episodes'] is not None:
                eval_episodes = int(_heb_eval_cfg['eval_episodes'])
            if 'seed_base' in _heb_eval_cfg and _heb_eval_cfg['seed_base'] is not None:
                # Common-random-seeds: every candidate in a generation shares map
                # seeds so selection isolates coefficient quality from map luck.
                _epoch = int(getattr(args, 'epoch', 0))
                seed_base = int(_heb_eval_cfg['seed_base']) + _epoch * 999983
            if 'render' in _heb_eval_cfg and _heb_eval_cfg['render'] is not None:
                eval_render = bool(_heb_eval_cfg['render'])
            if 'mission' in _heb_eval_cfg and _heb_eval_cfg['mission'] is not None:
                MISSION = str(_heb_eval_cfg['mission'])
            if 'fitness_stat' in _heb_eval_cfg and _heb_eval_cfg['fitness_stat'] is not None:
                FITNESS_STAT = str(_heb_eval_cfg['fitness_stat'])
            if 'init_weights' in _heb_eval_cfg and _heb_eval_cfg['init_weights'] is not None:
                INIT_WEIGHTS = str(_heb_eval_cfg['init_weights'])
            if 'num_agents' in _heb_eval_cfg and _heb_eval_cfg['num_agents'] is not None:
                num_agents_override = int(_heb_eval_cfg['num_agents'])
            if 'obs_encoder' in _heb_eval_cfg and _heb_eval_cfg['obs_encoder'] is not None:
                obs_encoder = str(_heb_eval_cfg['obs_encoder'])
            if 'encoder_dim' in _heb_eval_cfg and _heb_eval_cfg['encoder_dim'] is not None:
                encoder_dim = int(_heb_eval_cfg['encoder_dim'])
            if 'encoder_seed' in _heb_eval_cfg and _heb_eval_cfg['encoder_seed'] is not None:
                encoder_seed = int(_heb_eval_cfg['encoder_seed'])
            _fv4 = _heb_eval_cfg.get('fitness_v4', None)
            if isinstance(_fv4, dict):
                fitness_v4_weights_str = ','.join(
                    f'{_k}={float(_v)}' for _k, _v in _fv4.items() if _v is not None
                )
            _fv5 = _heb_eval_cfg.get('fitness_v5', None)
            if isinstance(_fv5, dict):
                _parts = []
                for _k, _v in _fv5.items():
                    if _v is None:
                        continue
                    if _k == 'reduction':
                        _parts.append(f'{_k}={str(_v)}')
                    else:
                        _parts.append(f'{_k}={float(_v)}')
                fitness_v5_weights_str = ','.join(_parts)
            _fv7 = _heb_eval_cfg.get('fitness_v7', None)
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
            _fv8 = _heb_eval_cfg.get('fitness_v8', None)
            if isinstance(_fv8, dict):
                _parts = []
                for _k, _v in _fv8.items():
                    if _v is None:
                        continue
                    if _k == 'reduction':
                        _parts.append(f'{_k}={str(_v)}')
                    else:
                        _parts.append(f'{_k}={float(_v)}')
                fitness_v8_weights_str = ','.join(_parts)
            if 'max_steps' in _heb_eval_cfg and _heb_eval_cfg['max_steps'] is not None:
                max_steps_override = int(_heb_eval_cfg['max_steps'])
    except Exception as _cfg_exc:
        print(f'WARNING: failed to parse HebEval config from YAML ({_cfg_exc}); using defaults.', flush=True)

    eval_episodes = max(1, int(eval_episodes))

    # --- Architecture constants (must match evaluate_hebb_cogames.py) --------
    # INPUT_DIM depends on whether the frozen random projection is enabled:
    # raw obs is 900 dims for tutorial.scout/arena; encoder collapses to
    # encoder_dim. LAYER_SIZES drives every per-layer synapse count below.
    if obs_encoder == 'random_proj':
        _INPUT_DIM = int(encoder_dim)
    else:
        _INPUT_DIM = 900
    LAYER_SIZES    = [(_INPUT_DIM, 64), (64, 32), (32, 5)]
    N_COEFFS       = 5                                    # A, B, C, D, lr
    N_SYNAPSES_PER_LAYER = [i * o for i, o in LAYER_SIZES]
    TOTAL_SYNAPSES = sum(N_SYNAPSES_PER_LAYER)
    print(
        f'Hebb task: obs_encoder={obs_encoder} encoder_dim={encoder_dim} '
        f'-> INPUT_DIM={_INPUT_DIM} LAYER_SIZES={LAYER_SIZES} '
        f'TOTAL_SYNAPSES={TOTAL_SYNAPSES}',
        flush=True,
    )

    # seed_base is shared across all candidates in this generation so their
    # fitness differences reflect coefficient differences, not map luck. The
    # initial random network weights still differ per candidate because they
    # come from the gene (A/B/C/D/lr coefficients), not from seed_base.
    _agent_idx = int(getattr(args, 'agent_idx', 0))
    print(f'Effective seed_base={seed_base} (agent_idx={_agent_idx})', flush=True)

    # --- Extract and validate coefficient arrays from the gene ---------------
    def _get_array(key):
        val = param['param'][key]
        arr = np.asarray(val, dtype=np.float32).ravel()
        if arr.size != TOTAL_SYNAPSES:
            raise ValueError(
                f"Gene key '{key}' has {arr.size} values; "
                f"expected {TOTAL_SYNAPSES} (one per FC synapse)."
            )
        return arr

    A_arr  = _get_array('A')
    B_arr  = _get_array('B')
    C_arr  = _get_array('C')
    D_arr  = _get_array('D')
    lr_arr = _get_array('lr')
    _debug_stage(args, 'task_impl.gene_arrays_loaded')

    # --- Assemble (123712, 5) matrix: each row = one synapse, cols = [A,B,C,lr,D]
    # Index mapping matches hebbian_update_ABCD_lr_D_in: col[3]=lr, col[4]=D
    hebb_coeffs = np.stack([A_arr, B_arr, C_arr, lr_arr, D_arr], axis=1)
    assert hebb_coeffs.shape == (TOTAL_SYNAPSES, N_COEFFS), \
        f"Expected shape ({TOTAL_SYNAPSES}, {N_COEFFS}), got {hebb_coeffs.shape}"

    print(
        f"Hebb coeffs assembled: shape={hebb_coeffs.shape}  "
        f"A∈[{A_arr.min():.3f}, {A_arr.max():.3f}]  "
        f"B∈[{B_arr.min():.3f}, {B_arr.max():.3f}]  "
        f"C∈[{C_arr.min():.3f}, {C_arr.max():.3f}]  "
        f"D∈[{D_arr.min():.3f}, {D_arr.max():.3f}]  "
        f"lr∈[{lr_arr.min():.5f}, {lr_arr.max():.5f}]",
        flush=True
    )

    # --- Save hebb coeffs for subprocess -------------------------------------
    import torch
    unique_suffix = f"{os.getpid()}_{time.time_ns()}"
    hebb_coeffs_path = os.path.join(
        taskPath,
        f'heb_coeffs_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.dat'
    )
    torch.save(hebb_coeffs, hebb_coeffs_path)
    _debug_stage(args, 'task_impl.hebb_saved')

    sidecar_path = os.path.join(
        taskPath,
        f'components_v5_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.json'
    )
    sidecar_path_v7 = os.path.join(
        taskPath,
        f'components_v7_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.json'
    )
    sidecar_path_v8 = os.path.join(
        taskPath,
        f'components_v8_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.json'
    )

    # --- Build and launch evaluate_hebb_cogames.py ---------------------------
    cmd = [
        sys.executable, HEBB_EVAL_SCRIPT,
        '--mission', MISSION,
        '--hebb_rule', HEBB_RULE,
        '--hebb_coeffs_path', hebb_coeffs_path,
        '--init_weights', INIT_WEIGHTS,
        '--render', 'True' if eval_render else 'False',
        '--eval_episodes', str(eval_episodes),
        '--seed_base', str(seed_base),
        '--fitness_stat', FITNESS_STAT,
        '--obs_encoder', obs_encoder,
        '--encoder_dim', str(encoder_dim),
        '--encoder_seed', str(encoder_seed),
    ]
    if num_agents_override is not None:
        cmd += ['--num_agents', str(num_agents_override)]
    if max_steps_override is not None:
        cmd += ['--max_steps', str(max_steps_override)]
    if fitness_v4_weights_str:
        cmd += ['--fitness_v4_weights', fitness_v4_weights_str]
    if fitness_v5_weights_str:
        cmd += ['--fitness_v5_weights', fitness_v5_weights_str]
    if fitness_v7_weights_str:
        cmd += ['--fitness_v7_weights', fitness_v7_weights_str]
    if fitness_v8_weights_str:
        cmd += ['--fitness_v8_weights', fitness_v8_weights_str]
    if FITNESS_STAT == 'composite_v5_role_event':
        cmd += ['--components_sidecar', sidecar_path]
    elif FITNESS_STAT == 'composite_v7_zstat':
        cmd += ['--components_sidecar', sidecar_path_v7]
    elif FITNESS_STAT == 'composite_v8_zstat':
        cmd += ['--components_sidecar', sidecar_path_v8]
    print(f'evaluate_hebb_script={HEBB_EVAL_SCRIPT}', flush=True)
    print('Running: ' + ' '.join(cmd), flush=True)

    raw_score = None
    ep_reward_mean = None   # mean env-reward per episode, parsed from v2 log lines
    failure_reason = None
    try:
        _debug_stage(args, 'task_impl.evaluate_hebb.start')
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        _debug_stage(args, f'task_impl.evaluate_hebb.returncode.{result.returncode}')
        stdout = result.stdout
        stderr = result.stderr
        print('evaluate_hebb_cogames stdout:\n' + stdout, flush=True)
        if stderr:
            print('evaluate_hebb_cogames stderr:\n' + stderr, flush=True)

        if result.returncode != 0:
            print(
                f'WARNING: evaluate_hebb_cogames.py exited with code {result.returncode}. Applying failure penalty.',
                flush=True,
            )
            raw_score = FAILURE_RAW_SCORE
            failure_reason = f'evaluate-hebb-return-code-{result.returncode}'
        else:
            import re
            patterns = [
                r'[Ee]pisode\s+cumulative\s+rewards?\s+(-?[\d.]+(?:[eE][+-]?\d+)?)',
                r'[Ff]itness[:\s]+(-?[\d.]+(?:[eE][+-]?\d+)?)',
                r'[Ss]core[:\s]+(-?[\d.]+(?:[eE][+-]?\d+)?)',
                r'[Rr]eward[:\s]+(-?[\d.]+(?:[eE][+-]?\d+)?)',
                r'^(-?[\d.]+(?:[eE][+-]?\d+)?)\s*$',
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
                print('WARNING: Could not parse finite score. Applying failure penalty.', flush=True)
                raw_score = FAILURE_RAW_SCORE
                failure_reason = 'evaluate-hebb-score-parse-failed'

            # Parse per-episode env-reward from composite_exploration_v2 /
            # composite_v3_ep_dominant / composite_v4_role_gated /
            # composite_v5_role_event log lines.
            _ep_vals = []
            for _ln in stdout.splitlines():
                if ('v2:' in _ln or 'v3:' in _ln or 'v4:' in _ln
                        or 'v5:' in _ln or 'v7:' in _ln or 'v8:' in _ln):
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
        print('WARNING: evaluate_hebb_cogames.py timed out. Applying failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = 'evaluate-hebb-timeout'
    except Exception as _e:
        print(f'WARNING: evaluate_hebb_cogames.py exception: {_e}. Applying failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = f'evaluate-hebb-exception-{_e}'
    finally:
        try:
            os.remove(hebb_coeffs_path)
        except Exception:
            pass

    components_v5 = None
    if FITNESS_STAT == 'composite_v5_role_event' and os.path.exists(sidecar_path):
        try:
            with open(sidecar_path, 'rt') as _scf:
                components_v5 = json.load(_scf)
        except Exception as _sc_exc:
            print(f'WARNING: failed to load v5 components sidecar ({_sc_exc}).', flush=True)
        finally:
            try:
                os.remove(sidecar_path)
            except Exception:
                pass

    components_v7 = None
    if FITNESS_STAT == 'composite_v7_zstat' and os.path.exists(sidecar_path_v7):
        try:
            with open(sidecar_path_v7, 'rt') as _scf:
                components_v7 = json.load(_scf)
        except Exception as _sc_exc:
            print(f'WARNING: failed to load v7 components sidecar ({_sc_exc}).', flush=True)
        finally:
            try:
                os.remove(sidecar_path_v7)
            except Exception:
                pass

    components_v8 = None
    if FITNESS_STAT == 'composite_v8_zstat' and os.path.exists(sidecar_path_v8):
        try:
            with open(sidecar_path_v8, 'rt') as _scf:
                components_v8 = json.load(_scf)
        except Exception as _sc_exc:
            print(f'WARNING: failed to load v8 components sidecar ({_sc_exc}).', flush=True)
        finally:
            try:
                os.remove(sidecar_path_v8)
            except Exception:
                pass

    outP['raw_scores'].append(raw_score)
    outP['ep_reward_mean'] = ep_reward_mean
    if components_v5 is not None:
        outP['components_v5'] = components_v5
    if components_v7 is not None:
        outP['components_v7'] = components_v7
    if components_v8 is not None:
        outP['components_v8'] = components_v8
    print(f'Raw CoGames score: {raw_score}  ep_reward_mean: {ep_reward_mean}', flush=True)

    if failure_reason is not None:
        outP['fitnessScore'] = FAILURE_FITNESS_SCORE
        outP['failureReason'] = failure_reason
    else:
        # L2 penalty normalized by squared gene range so its magnitude matches
        # the raw score signal (~0.01-0.05) instead of swamping it. Previous
        # formulation `-0.01 * mean(coeffs**2)` produced penalty ≈ -0.27 on a
        # [-10,10] gene range — 10x larger than the fitness signal — so the
        # optimizer spent all effort shrinking coefficients instead of
        # improving behavior. Scale factor chosen so a fully-saturated gene
        # (all values at ±max) incurs penalty ≈ -0.01.
        GENE_ABS_MAX = 10.0
        l2_penalty = -0.01 * np.mean(hebb_coeffs**2) / (GENE_ABS_MAX ** 2)
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


def main(args):

    global taskPath
    global params_config

    taskPath = args.path + "/running/" + str(args.agent_idx)
    os.makedirs(taskPath, exist_ok=True)
    _debug_stage(args, 'main.start')

    running_dir = os.path.join(args.path, "running")

    def _debug_list_running_dir(prefix):
        try:
            print(f"{prefix} running_dir={running_dir}", flush=True)
            if not os.path.exists(running_dir):
                print(f"{prefix} running_dir_missing", flush=True)
                return
            entries = sorted(os.listdir(running_dir))
            print(f"{prefix} running_dir_entries_count={len(entries)}", flush=True)
            for fname in entries:
                fpath = os.path.join(running_dir, fname)
                try:
                    st = os.stat(fpath)
                    print(f"{prefix} file={fname} size={st.st_size} mtime={st.st_mtime}", flush=True)
                except Exception as stat_exc:
                    print(f"{prefix} file={fname} stat_error={stat_exc}", flush=True)
        except Exception as dir_exc:
            print(f"{prefix} list_error={dir_exc}", flush=True)

    # load the gene from save tasks
    json_member_path = args.path + "/running/" + str(args.agent_idx) + '.newMember.json'
    pkl_member_path = args.path + "/running/" + str(args.agent_idx) + '.newMember.pkl'
    try:
        _debug_list_running_dir('[DEBUG main.members_load.before_json]')
        print(f"[DEBUG main.members_load.paths] json_path={json_member_path} pkl_path={pkl_member_path}", flush=True)
        _debug_stage(args, 'main.members_load.json.try')
        with open(json_member_path, 'rt') as f:
            params_config = json.load(f)
        format = 'json'
        _debug_stage(args, 'main.members_load.json.ok')
    except Exception as json_exc:
        _debug_stage(args, 'main.members_load.json.fail')
        pkl_exc = None
        for attempt in range(3):
            try:
                try:
                    print(
                        f"[DEBUG main.members_load.pktarget.attempt_{attempt+1}] exists={os.path.exists(pkl_member_path)} size={os.path.getsize(pkl_member_path) if os.path.exists(pkl_member_path) else 'na'}",
                        flush=True,
                    )
                except Exception as pkl_stat_exc:
                    print(f"[DEBUG main.members_load.pktarget.attempt_{attempt+1}] stat_error={pkl_stat_exc}", flush=True)
                _debug_stage(args, f'main.members_load.pkl.deserialize_isolated_file.attempt_{attempt+1}')
                params_config = safe_load_candidate_pkl_isolated(pkl_member_path, timeout_sec=120)
                format = 'pkl'
                _debug_stage(args, f'main.members_load.pkl.ok.attempt_{attempt+1}')
                pkl_exc = None
                break
            except Exception as e:
                pkl_exc = e
                time.sleep(1.0)

        if pkl_exc is not None:
            _quarantine_corrupt_file(pkl_member_path)
            if args.slurm:
                print(
                    f"INFO: unreadable newMember for agent={args.agent_idx}; json_error={json_exc}; pkl_error={pkl_exc}. Exiting with non-zero to stop SLURM requeue chain.",
                    flush=True,
                )
                sys.exit(3)
            raise RuntimeError(
                f"Failed to load newMember for agent={args.agent_idx}. "
                f"json_error={json_exc}; pkl_error={pkl_exc}; "
                f"json_path={json_member_path}; pkl_path={pkl_member_path}"
            )

    if not isinstance(params_config, list):
        raise RuntimeError(f"Loaded newMember has invalid type: {type(params_config)} (expected list)")

    owner_mismatch_count = 0
    owner_missing_count = 0
    for _idx, _entry in enumerate(params_config):
        if not isinstance(_entry, dict):
            continue
        expected_owner = _entry.get('candidate_agent_idx', None)
        if expected_owner is None:
            owner_missing_count += 1
            continue
        try:
            if int(expected_owner) != int(args.agent_idx):
                owner_mismatch_count += 1
        except Exception:
            owner_mismatch_count += 1

    if owner_mismatch_count > 0:
        mismatch_msg = (
            f"Candidate ownership mismatch for agent={args.agent_idx}: "
            f"{owner_mismatch_count}/{len(params_config)} entries tagged for a different agent."
        )
        if args.slurm:
            print(f"ERROR: {mismatch_msg}", flush=True)
            sys.exit(4)
        raise RuntimeError(mismatch_msg)

    if owner_missing_count > 0:
        print(
            f"WARNING: {owner_missing_count}/{len(params_config)} candidate entries missing candidate_agent_idx metadata.",
            flush=True,
        )

    _debug_stage(args, f'main.members_loaded.{format}.count_{len(params_config)}')

    for gene_num in range(len(params_config)):
        _debug_stage(args, f'main.gene_loop.start.{gene_num}')
        fitness = []
        raw_scores = []
        ep_reward_means = []
        failure_reasons = []
        components_v5_per_test = []
        components_v7_per_test = []
        components_v8_per_test = []
        for test in range(args.total_test_with_same_param):
            #-----------------------
            outP = run_task(params_config[gene_num], args)
            #-----------------------
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

            test_raw_scores = outP.get('raw_scores', [])
            if isinstance(test_raw_scores, list):
                raw_scores.extend(test_raw_scores)
            else:
                raw_scores.append(test_raw_scores)

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

            _cv5 = outP.get('components_v5', None)
            if _cv5 is not None:
                components_v5_per_test.append(_cv5)

            _cv7 = outP.get('components_v7', None)
            if _cv7 is not None:
                components_v7_per_test.append(_cv7)

            _cv8 = outP.get('components_v8', None)
            if _cv8 is not None:
                components_v8_per_test.append(_cv8)

        params_config[gene_num]['fitnessScore'] = fitness
        params_config[gene_num]['raw_scores'] = raw_scores
        if ep_reward_means:
            params_config[gene_num]['ep_reward_mean'] = float(np.mean(ep_reward_means))
        if len(failure_reasons) > 0:
            params_config[gene_num]['failureReason'] = sorted(set(failure_reasons))
        if components_v5_per_test:
            params_config[gene_num]['components_v5_per_test'] = components_v5_per_test
        if components_v7_per_test:
            params_config[gene_num]['components_v7_per_test'] = components_v7_per_test
        if components_v8_per_test:
            params_config[gene_num]['components_v8_per_test'] = components_v8_per_test

    # write resoults
    _debug_stage(args, 'main.write_results.start')
    if format == 'json':
        with open(taskPath + '/' + str(args.agent_test_num)  + '.resoult.json', 'wt') as f:
            json.dump(params_config, f)
    else:
        with zstandard.open(taskPath + '/' + str(args.agent_test_num)  + '.resoult.pkl', 'wb') as f:
            f.write(pickle.dumps(params_config))

    if args.parallel or args.slurm:
        _debug_stage(args, 'main.callback.schedule')
        run(args, self=False, params=params_config[-1].copy())

    # delete the gene file
    os.system('rm -f ' + shlex.quote(args.path + "/running/" + str(args.agent_idx) + '.newMember.json'))
    os.system('rm -f ' + shlex.quote(args.path + "/running/" + str(args.agent_idx) + '.newMember.pkl'))

    print('Done Task: ' + str(args.agent_test_num) + ' for agent: ' + str(args.agent_idx), flush=True)
    return

def run(args, self=False, params=None):
    # OVERloading agent_counter
    args.agent_counter = str(args.agent_counter) + '.' + str(args.epoch)

    if self:
        run_process = [
            __file__,
            "--callBackScript", '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"',''),
            "--path", args.path,
            '--agent_idx', str(args.agent_idx),
            "--agent_test_num", str(args.agent_test_num),
            "--agent_counter", str(args.agent_counter),
            "--total_test_with_same_param", str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'True' if args.parallel else 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
            ]
    else:
        run_process = [
            '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"',''),
            '--part', '2',
            '--taskFilename', '"' + __file__ + '"',
            "--path", args.path,
            '--agent_idx', str(args.agent_idx),
            "--agent_test_num", str(args.agent_test_num),
            "--agent_counter", str(args.agent_counter),
            "--total_test_with_same_param", str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
            ]

    if args.slurm:
        os.makedirs(args.path + '/scripts/', exist_ok=True)
        temp_file = args.path + '/scripts/Agent_idx_' + str(args.agent_idx) + '_Task_' + str(args.agent_test_num)
        temp_file += '_Epoch_' + str(args.epoch) + '.sh'

        job_name  = args.path.split('/')[-1]
        if job_name == '':
            job_name = args.path.split('/')[-2]
        script  = SlurmScript(
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
            command="python " + str(run_process).replace('[','').replace(']','').replace('\', \'',' ').replace('\'',''),
            nextTask=temp_file,
        )

        with open(temp_file, 'w') as file:
            file.write(script)
            file.flush()

    elif args.parallel:
        run_process = str(['python3'] + run_process).replace('[','').replace(']','').replace('\'','').replace(',', ' ')
        subprocess.Popen(run_process, shell=True)

# must be after run function
def signal_handler_SIGTERM(sig, frame):
    global args
    # Preemption handling: the framework requeues and restarts the job cleanly, so we
    # deliberately do NOT save state or resume here. We only remove this agent's own
    # transient in-progress files (live lock + worker temp files) so the restarted job
    # starts on a clean work dir. The .newMember candidate is a sibling file in
    # running/ (NOT inside this work dir) and is intentionally left untouched so the
    # requeued job can redo the work.
    try:
        work_dir = os.path.join(args.path, 'running', str(args.agent_idx))
        transient = (
            glob.glob(os.path.join(work_dir, '*.live.lock'))
            + glob.glob(os.path.join(work_dir, 'worker_in_*.pkl'))
            + glob.glob(os.path.join(work_dir, 'worker_out_*.pkl'))
        )
        for _f in transient:
            try:
                os.remove(_f)
            except Exception:
                pass
    except Exception:
        pass
    print('got a SIGTERM (preemption); cleaned transient files, candidate left intact for requeue.', flush=True)
    sys.exit(0)

if __name__ == "__main__":
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='FindSeq')
    parser.add_argument("--agent_test_num", type=int, default=-1)
    parser.add_argument("--agent_idx", type=int, default=-1)
    parser.add_argument("--agent_counter", type=str, default='-1.0')
    parser.add_argument("--total_test_with_same_param", type=int, default=1)
    parser.add_argument("--total_testsGene_with_same_agent", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='False')
    parser.add_argument("--slurm", type=str, default='False')
    parser.add_argument("--parallel", type=str, default='True')
    parser.add_argument("--callBackScript", type=str, default=None)
    parser.add_argument("--evolutionTarget", type=int, default=1)
    parser.add_argument("--paramFile", type=str, default=None)
    parser.add_argument("--worker_mode", type=str, default='False')
    parser.add_argument("--workerInput", type=str, default=None)
    parser.add_argument("--workerOutput", type=str, default=None)
    parser.add_argument("--main_worker_mode", type=str, default='False')
    parser.add_argument("--debug_no_subprocess", type=str, default='False')

    args = parser.parse_args()

    args.gpu = True if args.gpu == 'True' or args.gpu == 'true' else False
    args.slurm = True if args.slurm == 'True' or args.slurm == 'true' else False
    args.parallel = True if args.parallel == 'True' or args.parallel == 'true' else False
    args.worker_mode = True if args.worker_mode == 'True' or args.worker_mode == 'true' else False
    args.main_worker_mode = True if args.main_worker_mode == 'True' or args.main_worker_mode == 'true' else False
    args.debug_no_subprocess = True if args.debug_no_subprocess == 'True' or args.debug_no_subprocess == 'true' else False
    _debug_stage(args, 'entry.args_parsed')

    if isinstance(args.paramFile, str) and args.paramFile.strip().lower() in ['', 'none', 'null']:
        args.paramFile = None
    if args.paramFile is None:
        task_root, _ = os.path.splitext(os.path.abspath(__file__))
        default_yaml = task_root + '.yaml'
        default_yml = task_root + '.yml'
        args.paramFile = default_yaml
        if (not os.path.exists(default_yaml)) and os.path.exists(default_yml):
            args.paramFile = default_yml

    if args.parallel:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    if args.slurm:
        signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
        signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
        signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
        signal.signal(signal.SIGINT, signal_handler_SIGTERM)

    if not args.callBackScript is None:
        args.callBackScript = args.callBackScript.replace('"','')

    # OVERloading agent_counter
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.worker_mode:
        _debug_stage(args, 'entry.worker_mode.start')
        taskPath = args.path + "/running/" + str(args.agent_idx)
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

    if args.slurm and (not args.main_worker_mode) and (not args.debug_no_subprocess):
        _debug_stage(args, 'entry.main_worker_wrapper.start')
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
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
        _debug_stage(args, f'entry.main_worker_wrapper.returned.{_describe_return_code(return_code)}')
        if stdout:
            print('main worker stdout:\n' + stdout, flush=True)
        if stderr:
            print('main worker stderr:\n' + stderr, flush=True)

        if return_code in (3, 4):
            # Deliberate sentinels from the inner worker: 3 = no work / no candidate /
            # remap-refused, 4 = owner mismatch. These are NOT crashes. The old behavior
            # treated them as crashes and wrote "fallback" results, which fabricated
            # failure scores and deleted candidate files belonging to OTHER agents,
            # corrupting the GA. Treat them as a clean no-op and let the framework
            # requeue/redispatch as needed.
            print(f'INFO: main worker reported no-work/mismatch ({_describe_return_code(return_code)}); '
                  f'not a crash, no fallback written, exiting 0.', flush=True)
            sys.exit(0)

        if return_code != 0:
            crash_desc = _describe_return_code(return_code)
            print(f'WARNING: main worker crashed ({crash_desc}). Writing fallback results.', flush=True)
            fallback_ok = _write_fallback_results_and_continue(args, reason=f'main-worker-{crash_desc}')
            if not fallback_ok:
                print('ERROR: fallback recovery failed; exiting non-zero to avoid false-success job chaining.', flush=True)
                sys.exit(2)

        sys.exit(0)

    if args.slurm and (not args.main_worker_mode) and args.debug_no_subprocess:
        print('DEBUG: running without main-worker subprocess (--debug_no_subprocess=True).', flush=True)
        _debug_stage(args, 'entry.debug_no_subprocess.inline_mode')

    if args.slurm:
        args.parallel = False

    resolved_agent_idx = _resolve_agent_idx_for_existing_member(args)
    if resolved_agent_idx is None:
        msg = f"No work for agent {args.agent_idx}: no *.newMember.(json|pkl) found in {args.path}/running"
        running_dir = os.path.join(args.path, 'running')
        json_candidates = glob.glob(os.path.join(running_dir, '*.newMember.json'))
        pkl_candidates = glob.glob(os.path.join(running_dir, '*.newMember.pkl'))
        print(
            f"INFO: candidate inventory json={len(json_candidates)} pkl={len(pkl_candidates)} path={running_dir}",
            flush=True,
        )
        if args.slurm:
            print(f"INFO: {msg}; exiting with non-zero to stop SLURM requeue chain.", flush=True)
            sys.exit(3)
        print(f"ERROR: {msg}", flush=True)
        sys.exit(2)
    if resolved_agent_idx != args.agent_idx:
        if args.slurm:
            print(
                f"INFO: requested agent={args.agent_idx} has no candidate; nearest available agent={resolved_agent_idx}. "
                f"Skipping remap in SLURM mode to avoid cross-agent collisions.",
                flush=True,
            )
            sys.exit(3)
        print(f"Agent index remapped: requested={args.agent_idx}, using={resolved_agent_idx}", flush=True)
        args.agent_idx = resolved_agent_idx
    _debug_stage(args, 'entry.before_main')

    main(argparse._copy_items(args))
