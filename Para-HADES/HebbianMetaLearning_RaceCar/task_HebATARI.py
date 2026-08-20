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
    # HebbianMetaLearning  –  CarRacing-v0, ABCD_lr
    # =========================================================================
    #
    # Network architecture (FC layers only are plastic):
    #   Layer 1:  CNN_out(648) → 128   →  82,944 synapses
    #   Layer 2:  128          →  64   →   8,192 synapses
    #   Layer 3:  64           →   3   →     192 synapses
    #   TOTAL:                         →  91,328 synapses
    #
    # The YAML evolves 5 flat arrays each of length 91,328:
    #   param['param']['A']  – shape (91328,)
    #   param['param']['B']  – shape (91328,)
    #   param['param']['C']  – shape (91328,)
    #   param['param']['D']  – shape (91328,)
    #   param['param']['lr'] – shape (91328,)
    #
    # evaluate_hebb.py expects hebb_coeffs as a single flat numpy array:
    #   [ A_layer1 | A_layer2 | A_layer3 |
    #     B_layer1 | B_layer2 | B_layer3 |
    #     C_layer1 | C_layer2 | C_layer3 |
    #     D_layer1 | D_layer2 | D_layer3 |
    #     lr_layer1| lr_layer2| lr_layer3 ]
    #
    # Since A/B/C/D/lr arrays in the YAML are already ordered
    # [layer1_synapses … layer2_synapses … layer3_synapses] (row-major
    # flattening of each weight matrix), we just concatenate the five
    # coefficient arrays in order to build the full 456,640-element vector.
    # =========================================================================

    outP = dict()
    outP['raw_scores'] = []

    # --- Architecture constants (must match policies.py) ---------------------
    LAYER_SIZES   = [(648, 128), (128, 64), (64, 3)]   # (in, out) per FC layer
    N_COEFFS      = 5                                   # A, B, C, D, lr  (columns in hebb matrix)
    N_SYNAPSES_PER_LAYER = [i * o for i, o in LAYER_SIZES]
    TOTAL_SYNAPSES = sum(N_SYNAPSES_PER_LAYER)          # 91,328  (rows in hebb matrix)
    CNN_N_PARAMS   = 1362                               # conv1: 6×3×3×3=162, conv2: 8×6×5×5=1200

    # --- Path configuration --------------------------------------------------
    # Adjust HEBB_EVAL_SCRIPT and CNN_PARAMS_PATH to match your repo layout.
    task_dir = os.path.dirname(os.path.abspath(__file__))
    HEBB_EVAL_SCRIPT = os.path.join(task_dir, 'evaluate_hebb.py')
    HEBB_RULE        = 'ABCD_lr'
    ENVIRONMENT      = 'CarRacing-v0'
    INIT_WEIGHTS     = 'uni'
    eval_episodes = 3
    seed_base = 12345
    eval_render = False

    # Optional evaluator controls from YAML (main.HebEval)
    try:
        if isinstance(args.paramFile, str) and os.path.exists(args.paramFile):
            with open(args.paramFile, 'rt') as _f:
                _yaml_cfg = yaml.safe_load(_f) or {}
            _heb_eval_cfg = ((_yaml_cfg.get('main', {}) or {}).get('HebEval', {}) or {})
            if 'eval_episodes' in _heb_eval_cfg and _heb_eval_cfg['eval_episodes'] is not None:
                eval_episodes = int(_heb_eval_cfg['eval_episodes'])
            if 'seed_base' in _heb_eval_cfg and _heb_eval_cfg['seed_base'] is not None:
                seed_base = int(_heb_eval_cfg['seed_base']) + int(time.time() * 1000) % 1000000 + os.getpid()
            if 'render' in _heb_eval_cfg and _heb_eval_cfg['render'] is not None:
                eval_render = bool(_heb_eval_cfg['render'])
    except Exception as _cfg_exc:
        print(f'WARNING: failed to parse HebEval config from YAML ({_cfg_exc}); using defaults.', flush=True)

    eval_episodes = max(1, int(eval_episodes))
    # -------------------------------------------------------------------------

    # --- Extract and validate coefficient arrays from the gene ---------------
    def _get_array(key):
        """Pull a coefficient array out of the gene dict and return as float32."""
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

    # --- Extract and validate CNN weights from the gene ----------------------
    cnn_arr = np.asarray(param['param']['cnn'], dtype=np.float32).ravel()
    if cnn_arr.size != CNN_N_PARAMS:
        raise ValueError(
            f"Gene key 'cnn' has {cnn_arr.size} values; "
            f"expected {CNN_N_PARAMS} (conv1=162 + conv2=1200)."
        )

    # --- Assemble (91328, 5) matrix: each row = one synapse, cols = [A,B,C,lr,D]
    # To match hebbian_update_ABCD_lr_D_in which uses idx 3 for lr multiplier and idx 4 for D offset.
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
    print(
        f"CNN params: shape={cnn_arr.shape}  "
        f"∈[{cnn_arr.min():.3f}, {cnn_arr.max():.3f}]",
        flush=True
    )

    # --- Write with torch.save (matches the format evaluate_hebb.py loads) ---
    unique_suffix = f"{os.getpid()}_{time.time_ns()}"
    hebb_coeffs_path = os.path.join(
        taskPath,
        f'heb_coeffs_{args.agent_idx}_{args.agent_test_num}_{unique_suffix}.dat'
    )
    import torch
    cnn_params_path = os.path.join(
        taskPath,
        f'cnn_params_{args.agent_idx}_{args.agent_test_num}.dat'
    )
    torch.save(hebb_coeffs, hebb_coeffs_path)
    torch.save(cnn_arr, cnn_params_path)
    _debug_stage(args, 'task_impl.hebb_saved')

    # --- Build and launch evaluate_hebb.py -----------------------------------
    cmd = [
        'xvfb-run', '-a', '-s', '-screen 0 1400x900x24 +extension RANDR',
        '--', sys.executable, HEBB_EVAL_SCRIPT,
        '--environment', ENVIRONMENT,
        '--hebb_rule', HEBB_RULE,
        '--path_hebb', hebb_coeffs_path,
        '--path_coev', cnn_params_path,
        '--init_weights', INIT_WEIGHTS,
        '--render', 'True' if eval_render else 'False',
        '--eval_episodes', str(eval_episodes),
        '--seed_base', str(seed_base),
    ]
    print(f'evaluate_hebb_script={HEBB_EVAL_SCRIPT}', flush=True)
    print('Running: ' + ' '.join(cmd), flush=True)

    raw_score = None
    failure_reason = None
    try:
        _debug_stage(args, 'task_impl.evaluate_hebb.start')
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,    # 10-minute hard timeout; adjust for your hardware
        )
        _debug_stage(args, f'task_impl.evaluate_hebb.returncode.{result.returncode}')
        stdout = result.stdout
        stderr = result.stderr
        print('evaluate_hebb stdout:\n' + stdout, flush=True)
        if stderr:
            print('evaluate_hebb stderr:\n' + stderr, flush=True)

        if result.returncode != 0:
            print(
                f'WARNING: evaluate_hebb.py exited with code {result.returncode}. Applying failure penalty.',
                flush=True,
            )
            raw_score = FAILURE_RAW_SCORE
            failure_reason = f'evaluate-hebb-return-code-{result.returncode}'
        else:

            # Parse score from evaluate_hebb.py output.
            # Primary format: "Episode cumulative rewards  305"
            # Fallback formats also handled for robustness.
            import re
            patterns = [
                r'[Ee]pisode\s+cumulative\s+rewards?\s+(-?[\d.]+)',  # "Episode cumulative rewards  305"
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
                failure_reason = 'evaluate-hebb-score-parse-failed'

    except subprocess.TimeoutExpired:
        print('WARNING: evaluate_hebb.py timed out. Applying failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = 'evaluate-hebb-timeout'
    except Exception as _e:
        print(f'WARNING: evaluate_hebb.py exception: {_e}. Applying failure penalty.', flush=True)
        raw_score = FAILURE_RAW_SCORE
        failure_reason = f'evaluate-hebb-exception-{_e}'
    finally:
        try:
            os.remove(hebb_coeffs_path)
        except Exception:
            pass
        try:
            os.remove(cnn_params_path)
        except Exception:
            pass


    outP['raw_scores'].append(raw_score)
    print(f'Raw CarRacing score: {raw_score}', flush=True)

    if failure_reason is not None:
        outP['fitnessScore'] = FAILURE_FITNESS_SCORE
        outP['failureReason'] = failure_reason
    else:
        # L2 weight decay on Hebbian coefficients (matches evolution_strategy_hebb.py)
        l2_penalty = -0.01 * np.mean(hebb_coeffs**2)
        penalized_score = raw_score + l2_penalty
        print(f'L2 penalty: {l2_penalty:.6f}  penalized score: {penalized_score:.4f}', flush=True)

        # CarRacing is "higher is better"; maximise when evolutionTarget == 1
        if args.evolutionTarget == -1:
            outP['fitnessScore'] = float(-1.0 * penalized_score)
        else:
            outP['fitnessScore'] = float(penalized_score)

    #------------------------------------------------------------------------------------------
    # Make sure the `fitnessScore` is in return dict and the fitness is a float number!
    #------------------------------------------------------------------------------------------

    print(f'---- End Task ----- {datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    print(f'---- Task Total Time ----- {(time.time() - start_time)} sec', flush=True)

    # delete alive file and terminate job
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
                    f"INFO: unreadable newMember for agent={args.agent_idx}; json_error={json_exc}; pkl_error={pkl_exc}. Exiting with non-zero to stop empty SLURM requeue chain.",
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
        for test in range(args.total_test_with_same_param):
            #-----------------------
            outP = run_task(params_config[gene_num], args)
            #-----------------------
            if not isinstance(outP, dict):
                outP = _fallback_outcome(args, reason='main-nondict-task-output')

            reason = outP.get('failureReason', None)
            if reason not in [None, '', 'None']:
                failure_reasons.append(str(reason))
                fitness.append(None)  # failed seed excluded from aggregation; raw_scores keeps the sentinel
            else:
                try:
                    score = float(outP.get('fitnessScore', FAILURE_FITNESS_SCORE))
                    if not math.isfinite(score):
                        raise ValueError('non-finite score')
                    fitness.append(score)
                except Exception:
                    failure_reasons.append('main-invalid-fitnessscore')
                    fitness.append(None)

            test_raw_scores = outP.get('raw_scores', [])
            if isinstance(test_raw_scores, list):
                raw_scores.extend(s for s in test_raw_scores if s is not None)
            elif test_raw_scores is not None:
                raw_scores.append(test_raw_scores)

        params_config[gene_num]['fitnessScore'] = fitness
        params_config[gene_num]['raw_scores'] = raw_scores
        if len(failure_reasons) > 0:
            params_config[gene_num]['failureReason'] = sorted(set(failure_reasons))

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
            '--parallel', 'True' if args.parallel else 'False',
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,
            ]

    if args.slurm:
        # make sure the log directory exists
        os.makedirs(args.path + '/scripts/', exist_ok=True)
        temp_file = args.path + '/scripts/Agent_idx_' + str(args.agent_idx) + '_Task_' + str(args.agent_test_num) 
        temp_file += '_Epoch_' + str(args.epoch) + '.sh'

        job_name  = args.path.split('/')[-1]
        if job_name == '':
            job_name = args.path.split('/')[-2]
        script  = SlurmScript(
            jobName=('T_' if self else 'A_') + job_name,
            jobTime=params['main']['Task_resource']['time'] if self else params['main']['Main_resource']['time'], # 00:30:00, 01:00:00, 02:00:00, 04:00:00
            jobMemory=params['main']['Task_resource']['memory'] if self else params['main']['Main_resource']['memory'], # 32768, 65536, 131072, 262144
            output_path=args.path + (params['main']['Task_resource']['path'] if self else params['main']['Main_resource']['path']),
            jobCPUs=params['main']['Task_resource']['cpu'] if self else params['main']['Main_resource']['cpu'],
            # theadsPerCore=params['main']['Task_resource']['theadsPerCore'] if self else params['main']['Main_resource']['theadsPerCore'],
            jobGPUs=params['main']['Task_resource']['gpu'] if self else params['main']['Main_resource']['gpu'],
            excludeNodes=params['main']['Task_resource']['exclude_nodes'] if self else params['main']['Main_resource']['exclude_nodes'],
            condaEnv=params['main']['Task_resource']['conda_env'] if self else params['main']['Main_resource']['conda_env'],
            partition=params['main']['Task_resource']['partition'] if self else params['main']['Main_resource']['partition'],
            singularity_image=params['main']['Task_resource']['singularity_image'] if self else params['main']['Main_resource']['singularity_image'],            
            command="python " + str(run_process).replace('[','').replace(']','').replace('\', \'',' ').replace('\'',''),
            # scriptName=args.path + '/scripts/' + 'Run_Task_idx-' if self else 'Run_Agent_idx-' + temp_file,
            nextTask=temp_file,
        )
        
        # create a file with the slurm script
        with open(temp_file, 'w') as file:
            file.write(script)
            file.flush()
        # SubmitJob(script, temp_file)

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
    
    ##### consider deleteing lok on log file for best gene
        
    print('Done signal_handler', flush=True)
    sys.exit(0)

if __name__ == "__main__":
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='FindSeq')       # pass to the next process
    parser.add_argument("--agent_test_num", type=int, default=-1)       # receved by the child process
    parser.add_argument("--agent_idx", type=int, default=-1)            # modify by the process
    parser.add_argument("--agent_counter", type=str, default='-1.0')        # modified by the next process
    parser.add_argument("--total_test_with_same_param", type=int, default=1) # pass to the next process
    parser.add_argument("--total_testsGene_with_same_agent", type=int, default=1)  # pass to the next
    parser.add_argument("--gpu", type=str, default='False')        # pass to the next process
    parser.add_argument("--slurm", type=str, default='False')      # pass to the next process
    parser.add_argument("--parallel", type=str, default='True')      # NO need to pass to the next process or to myself
    parser.add_argument("--callBackScript", type=str, default=None)
    parser.add_argument("--evolutionTarget", type=int, default=1)      # 1 = Max, -1 = Min 
    parser.add_argument("--paramFile", type=str, default=None)      # 1 = Max, -1 = Min
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
            print(f"INFO: {msg}; exiting with non-zero to stop empty SLURM requeue chain.", flush=True)
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
