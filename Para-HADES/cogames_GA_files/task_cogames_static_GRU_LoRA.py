import os, sys, json, subprocess, signal, argparse, pickle, zstandard
import shlex
import glob
import multiprocessing as mp
import time
import tempfile
import math
import yaml
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
        candidates = glob.glob(os.path.join(running_dir, '*.newMember.json')) + glob.glob(os.path.join(running_dir, '*.newMember.pkl'))
        if len(candidates) > 0:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            best = os.path.basename(candidates[0])
            try:
                effective_agent_idx = int(best.split('.newMember.')[0])
                print(f'Fallback detected agent_idx mismatch: requested={args.agent_idx}, using={effective_agent_idx}', flush=True)
            except Exception:
                pass

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

    try:
        os.system('rm -f ' + shlex.quote(args.path + "/running/" + str(effective_agent_idx) + '.newMember.json'))
        os.system('rm -f ' + shlex.quote(args.path + "/running/" + str(effective_agent_idx) + '.newMember.pkl'))
    except Exception:
        pass

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
    # Static GRU LoRA (non-plastic recurrent, low-rank input projection)
    #   –  CoGames tutorial.scout
    # =========================================================================
    # Only the tall input projection W_in is factorised (86% of GRU params).
    # Small GRU/output matrices are evolved densely.
    #   W_in  = beta_in * W0_in + U_in @ V_in     (W0 frozen, rank=8)
    #   W_ih, W_hh, W_out: dense flat arrays.
    #
    # Param budget (proj=64, hidden=32, rank=8):
    #   beta_in: 1   U_in: 900*8=7200   V_in: 8*64=512
    #   W_ih: 6144   W_hh: 3072   W_out: 160
    #   TOTAL = 17,089 params
    # =========================================================================

    outP = dict()
    outP['raw_scores'] = []

    # --- Path configuration --------------------------------------------------
    task_dir = os.path.dirname(os.path.abspath(__file__))
    STATIC_EVAL_SCRIPT = os.path.join(task_dir, 'evaluate_static_GRU_cogames.py')
    MISSION            = 'tutorial.scout'
    FITNESS_STAT       = 'episode_reward'
    eval_episodes      = 3
    seed_base          = 12345
    eval_render        = False
    num_agents_override = None
    fitness_v5_weights_str = ''
    proj_dim           = 64
    hidden_dim         = 32
    lora_rank          = 8
    w0_seed            = 12345
    obs_mode           = 'stacked_grid'
    frame_stack        = 4
    grid_h             = 11
    grid_w             = 11
    num_channels       = 3
    global_state_dim   = 70

    # Optional overrides from YAML (main.StaticEval + main.StaticLoRA)
    try:
        if isinstance(args.paramFile, str) and os.path.exists(args.paramFile):
            with open(args.paramFile, 'rt') as _f:
                _yaml_cfg = yaml.safe_load(_f) or {}
            _cfg = ((_yaml_cfg.get('main', {}) or {}).get('StaticEval', {}) or {})
            _lora_cfg = ((_yaml_cfg.get('main', {}) or {}).get('StaticLoRA', {}) or {})
            if 'eval_episodes' in _cfg and _cfg['eval_episodes'] is not None:
                eval_episodes = int(_cfg['eval_episodes'])
            if 'seed_base' in _cfg and _cfg['seed_base'] is not None:
                # Common-random-seeds: every candidate in a generation shares map
                # seeds so selection isolates weight quality from map luck.
                _epoch = int(getattr(args, 'epoch', 0))
                seed_base = int(_cfg['seed_base']) + _epoch * 999983
            if 'render' in _cfg and _cfg['render'] is not None:
                eval_render = bool(_cfg['render'])
            if 'mission' in _cfg and _cfg['mission'] is not None:
                MISSION = str(_cfg['mission'])
            if 'fitness_stat' in _cfg and _cfg['fitness_stat'] is not None:
                FITNESS_STAT = str(_cfg['fitness_stat'])
            if 'num_agents' in _cfg and _cfg['num_agents'] is not None:
                num_agents_override = int(_cfg['num_agents'])
            if 'proj' in _cfg and _cfg['proj'] is not None:
                proj_dim = int(_cfg['proj'])
            if 'hidden' in _cfg and _cfg['hidden'] is not None:
                hidden_dim = int(_cfg['hidden'])
            if 'obs_mode' in _cfg and _cfg['obs_mode'] is not None:
                obs_mode = str(_cfg['obs_mode'])
            if 'frame_stack' in _cfg and _cfg['frame_stack'] is not None:
                frame_stack = int(_cfg['frame_stack'])
            if 'grid_h' in _cfg and _cfg['grid_h'] is not None:
                grid_h = int(_cfg['grid_h'])
            if 'grid_w' in _cfg and _cfg['grid_w'] is not None:
                grid_w = int(_cfg['grid_w'])
            if 'num_channels' in _cfg and _cfg['num_channels'] is not None:
                num_channels = int(_cfg['num_channels'])
            if 'global_state_dim' in _cfg and _cfg['global_state_dim'] is not None:
                global_state_dim = int(_cfg['global_state_dim'])
            if 'rank' in _lora_cfg and _lora_cfg['rank'] is not None:
                lora_rank = int(_lora_cfg['rank'])
            if 'w0_seed' in _lora_cfg and _lora_cfg['w0_seed'] is not None:
                w0_seed = int(_lora_cfg['w0_seed'])
            _fv5 = _cfg.get('fitness_v5', None)
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
    except Exception as _cfg_exc:
        print(f'WARNING: failed to parse StaticEval/StaticLoRA config from YAML ({_cfg_exc}); using defaults.', flush=True)

    if lora_rank <= 0:
        raise ValueError(f'Invalid LoRA rank={lora_rank}.')

    eval_episodes = max(1, int(eval_episodes))
    if obs_mode == 'stacked_grid':
        INPUT_DIM = (grid_h * grid_w * num_channels + global_state_dim) * frame_stack
    else:
        INPUT_DIM = 900
    ACTION_DIM = 5
    EXPECTED = {
        'beta_in': 1,
        'W_in_u':  INPUT_DIM * lora_rank,
        'W_in_v':  lora_rank * proj_dim,
        'W_ih':    3 * hidden_dim * proj_dim,
        'W_hh':    3 * hidden_dim * hidden_dim,
        'W_out':   ACTION_DIM * hidden_dim,
    }

    _agent_idx = int(getattr(args, 'agent_idx', 0))
    # seed_base is shared across all candidates in this generation; see comment above.
    print(
        f'Effective seed_base={seed_base} (agent_idx={_agent_idx})  '
        f'proj={proj_dim}  hidden={hidden_dim}  rank={lora_rank}  w0_seed={w0_seed}  '
        f'obs_mode={obs_mode}  frame_stack={frame_stack}  input_dim={INPUT_DIM}',
        flush=True,
    )

    # --- Extract and validate weight arrays from the gene -------------------
    def _get_array(key, expected_size):
        val = param['param'][key]
        arr = np.asarray(val, dtype=np.float32).ravel()
        if arr.size != expected_size:
            raise ValueError(
                f"Gene key '{key}' has {arr.size} values; expected {expected_size}."
            )
        return arr

    beta_in_arr = _get_array('beta_in', EXPECTED['beta_in'])
    W_in_u_arr  = _get_array('W_in_u',  EXPECTED['W_in_u'])
    W_in_v_arr  = _get_array('W_in_v',  EXPECTED['W_in_v'])
    W_ih  = _get_array('W_ih',  EXPECTED['W_ih'])
    W_hh  = _get_array('W_hh',  EXPECTED['W_hh'])
    W_out = _get_array('W_out', EXPECTED['W_out'])

    # Compose W_in = beta_in * W0_in + U_in @ V_in, then transpose to (proj, input)
    beta_in = float(beta_in_arr[0])
    U_in    = W_in_u_arr.reshape(INPUT_DIM, lora_rank)
    V_in    = W_in_v_arr.reshape(lora_rank, proj_dim)
    rng_w0  = np.random.default_rng(int(w0_seed))
    sigma_w0 = 1.0 / np.sqrt(float(INPUT_DIM))
    W0_in = rng_w0.normal(0.0, sigma_w0, size=(INPUT_DIM, proj_dim)).astype(np.float32)
    W_in_dense = (beta_in * W0_in + np.matmul(U_in, V_in)).astype(np.float32)
    # nn.Linear.weight is (out, in) = (proj, input_dim)
    W_in = W_in_dense.T.reshape(-1)
    _debug_stage(args, 'task_impl.gene_arrays_loaded')
    print(
        f"GRU weights loaded: "
        f"W_in∈[{W_in.min():.3f}, {W_in.max():.3f}]  "
        f"W_ih∈[{W_ih.min():.3f}, {W_ih.max():.3f}]  "
        f"W_hh∈[{W_hh.min():.3f}, {W_hh.max():.3f}]  "
        f"W_out∈[{W_out.min():.3f}, {W_out.max():.3f}]",
        flush=True,
    )

    # --- Save weights blob for subprocess ------------------------------------
    import torch
    unique_suffix = f"{os.getpid()}_{time.time_ns()}"
    weights_path = os.path.join(
        taskPath,
        f'gru_weights_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.dat'
    )
    torch.save({'W_in': W_in, 'W_ih': W_ih, 'W_hh': W_hh, 'W_out': W_out}, weights_path)
    _debug_stage(args, 'task_impl.weights_saved')

    sidecar_path = os.path.join(
        taskPath,
        f'components_v5_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.json'
    )

    # --- Build and launch evaluate_static_GRU_cogames.py ---------------------
    cmd = [
        sys.executable, STATIC_EVAL_SCRIPT,
        '--mission', MISSION,
        '--weights_path', weights_path,
        '--proj', str(proj_dim),
        '--hidden', str(hidden_dim),
        '--render', 'True' if eval_render else 'False',
        '--eval_episodes', str(eval_episodes),
        '--seed_base', str(seed_base),
        '--fitness_stat', FITNESS_STAT,
        '--obs_mode', obs_mode,
        '--frame_stack', str(frame_stack),
    ]
    if num_agents_override is not None:
        cmd += ['--num_agents', str(num_agents_override)]
    if fitness_v5_weights_str:
        cmd += ['--fitness_v5_weights', fitness_v5_weights_str]
    if FITNESS_STAT == 'composite_v5_role_event':
        cmd += ['--components_sidecar', sidecar_path]
    print(f'evaluate_static_gru_script={STATIC_EVAL_SCRIPT}', flush=True)
    print('Running: ' + ' '.join(cmd), flush=True)

    raw_score = None
    failure_reason = None
    try:
        _debug_stage(args, 'task_impl.evaluate_static.start')
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        _debug_stage(args, f'task_impl.evaluate_static.returncode.{result.returncode}')
        stdout = result.stdout
        stderr = result.stderr
        print('evaluate_static_cogames stdout:\n' + stdout, flush=True)
        if stderr:
            print('evaluate_static_cogames stderr:\n' + stderr, flush=True)

        if result.returncode != 0:
            print(
                f'WARNING: evaluate_static_cogames.py exited with code {result.returncode}. Applying failure penalty.',
                flush=True,
            )
            raw_score = FAILURE_RAW_SCORE
            failure_reason = f'evaluate-static-return-code-{result.returncode}'
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
                print('WARNING: Could not parse finite score. Applying failure penalty.', flush=True)
                raw_score = FAILURE_RAW_SCORE
                failure_reason = 'evaluate-static-score-parse-failed'

    except subprocess.TimeoutExpired:
        print('WARNING: evaluate_static_cogames.py timed out. Applying failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = 'evaluate-static-timeout'
    except Exception as _e:
        print(f'WARNING: evaluate_static_cogames.py exception: {_e}. Applying failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = f'evaluate-static-exception-{_e}'
    finally:
        try:
            os.remove(weights_path)
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

    outP['raw_scores'].append(raw_score)
    if components_v5 is not None:
        outP['components_v5'] = components_v5
    print(f'Raw CoGames score: {raw_score}', flush=True)

    if failure_reason is not None:
        outP['fitnessScore'] = FAILURE_FITNESS_SCORE
        outP['failureReason'] = failure_reason
    else:
        if args.evolutionTarget == -1:
            outP['fitnessScore'] = float(-1.0 * raw_score)
        else:
            outP['fitnessScore'] = float(raw_score)

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
        failure_reasons = []
        components_v5_per_test = []
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

            reason = outP.get('failureReason', None)
            if reason not in [None, '', 'None']:
                failure_reasons.append(str(reason))

            _cv5 = outP.get('components_v5', None)
            if _cv5 is not None:
                components_v5_per_test.append(_cv5)

        params_config[gene_num]['fitnessScore'] = fitness
        params_config[gene_num]['raw_scores'] = raw_scores
        if len(failure_reasons) > 0:
            params_config[gene_num]['failureReason'] = sorted(set(failure_reasons))
        if components_v5_per_test:
            params_config[gene_num]['components_v5_per_test'] = components_v5_per_test

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
    global taskPath
    global params_config
    os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
    run(args=argparse._copy_items(args), self=True, params=params_config.copy())
    print('got a SIGTERM!!', flush=True)
    print('Done signal_handler', flush=True)
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
