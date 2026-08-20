import os, sys, json, subprocess, signal, argparse, pickle, zstandard
import shlex
import glob
import multiprocessing as mp
import time
import tempfile
import math
from datetime import datetime
import yaml
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


def reconstruct_full_hebb_params(gene_record: dict, params_config: dict):
    """
    Reconstruct the full Hebbian parameter payload from a LoRA gene record.

    Used by saveFounder() to write a companion _full_params.pt alongside the
    best gene. Operation is deterministic: W0 is regenerated from a fixed seed.

    Args:
        gene_record: Gene record dict with gene_record['param'] containing
                     beta_X scalars and X_lY_u / X_lY_v LoRA arrays.
        params_config: Full YAML config dict. Reads main.HebLoRA.{rank, w0_seed}.

    Returns:
        dict with:
            - 'hebb_coeffs': np.ndarray float32, shape (123712, 5)
    """
    import numpy as np

    # Architecture constants (must match evaluate_hebb_cogames.py)
    LAYER_SIZES    = [(900, 128), (128, 64), (64, 5)]
    N_COEFFS       = 5
    TOTAL_SYNAPSES = sum(d_in * d_out for d_in, d_out in LAYER_SIZES)   # 123712
    COEFF_FAMILIES = ['A', 'B', 'C', 'D', 'lr']

    # LoRA config from YAML
    _heb_lora_cfg = (params_config.get('main', {}) or {}).get('HebLoRA', {}) or {}
    lora_rank = int(_heb_lora_cfg.get('rank',    8))
    w0_seed   = int(_heb_lora_cfg.get('w0_seed', 12345))

    if lora_rank <= 0:
        raise ValueError(f'Invalid LoRA rank={lora_rank}.')

    if 'param' not in gene_record or not isinstance(gene_record['param'], dict):
        raise ValueError("gene_record missing 'param' dictionary.")

    gene_dict = gene_record['param']

    def _get_beta(family_name):
        key = f'beta_{family_name}'
        if key not in gene_dict:
            raise ValueError(f"Missing gene key '{key}'.")
        v = np.asarray(gene_dict[key], dtype=np.float32).ravel()
        if v.size != 1:
            raise ValueError(f"Gene key '{key}' must be scalar; got size={v.size}.")
        return float(v[0])

    def _get_lora(family_name, layer_idx, d_in, d_out):
        ln = layer_idx + 1
        uk, vk = f'{family_name}_l{ln}_u', f'{family_name}_l{ln}_v'
        if uk not in gene_dict:
            raise ValueError(f"Missing gene key '{uk}'.")
        if vk not in gene_dict:
            raise ValueError(f"Missing gene key '{vk}'.")
        u_arr = np.asarray(gene_dict[uk], dtype=np.float32).ravel()
        v_arr = np.asarray(gene_dict[vk], dtype=np.float32).ravel()
        exp_u, exp_v = d_in * lora_rank, lora_rank * d_out
        if u_arr.size != exp_u:
            raise ValueError(f"'{uk}' size {u_arr.size} != expected {exp_u}.")
        if v_arr.size != exp_v:
            raise ValueError(f"'{vk}' size {v_arr.size} != expected {exp_v}.")
        return u_arr.reshape(d_in, lora_rank), v_arr.reshape(lora_rank, d_out)

    family_flat_arrays = {}
    for family_idx, family_name in enumerate(COEFF_FAMILIES):
        beta_val = _get_beta(family_name)
        layer_flats = []
        for layer_idx, (d_in, d_out) in enumerate(LAYER_SIZES):
            u_mat, v_mat = _get_lora(family_name, layer_idx, d_in, d_out)
            layer_seed = int(w0_seed + family_idx * 1000 + layer_idx)
            layer_rng  = np.random.default_rng(layer_seed)
            sigma_w0   = 1.0 / np.sqrt(float(d_in))
            w0         = layer_rng.normal(loc=0.0, scale=sigma_w0,
                                          size=(d_in, d_out)).astype(np.float32)
            delta       = np.matmul(u_mat, v_mat).astype(np.float32)
            layer_flats.append((beta_val * w0 + delta).astype(np.float32).reshape(-1))
        family_arr = np.concatenate(layer_flats, axis=0)
        if family_arr.size != TOTAL_SYNAPSES:
            raise ValueError(
                f"Family '{family_name}' size={family_arr.size}; expected {TOTAL_SYNAPSES}."
            )
        family_flat_arrays[family_name] = family_arr

    hebb_coeffs = np.stack([family_flat_arrays[f] for f in ['A', 'B', 'C', 'lr', 'D']], axis=1)
    assert hebb_coeffs.shape == (TOTAL_SYNAPSES, N_COEFFS), \
        f"Expected shape ({TOTAL_SYNAPSES}, {N_COEFFS}), got {hebb_coeffs.shape}"
    return {'hebb_coeffs': hebb_coeffs}


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
    # HebbianMetaLearning  –  CoGames tutorial.scout, ABCD_lr (LoRA)
    # =========================================================================
    #
    # For each coefficient family f ∈ {A,B,C,D,lr} and each FC layer l:
    #   W_f,l = beta_f * W0_f,l + U_f,l @ V_f,l
    #
    # W0_f,l is frozen deterministic noise generated from w0_seed.
    # sigma(W0) = 1 / sqrt(d_in)
    #
    # Layer shapes (rank=8 by default):
    #   l1: (900,128)  U:(900,8)=7200  V:(8,128)=1024
    #   l2: (128,64)   U:(128,8)=1024  V:(8,64)=512
    #   l3: (64,5)     U:(64,8)=512    V:(8,5)=40
    # =========================================================================

    outP = dict()
    outP['raw_scores'] = []

    # --- Architecture constants (must match evaluate_hebb_cogames.py) --------
    LAYER_SIZES    = [(900, 128), (128, 64), (64, 5)]
    N_COEFFS       = 5
    N_SYNAPSES_PER_LAYER = [i * o for i, o in LAYER_SIZES]
    TOTAL_SYNAPSES = sum(N_SYNAPSES_PER_LAYER)            # 123,712
    COEFF_FAMILIES = ['A', 'B', 'C', 'D', 'lr']

    # --- Path configuration --------------------------------------------------
    task_dir = os.path.dirname(os.path.abspath(__file__))
    HEBB_EVAL_SCRIPT = os.path.join(task_dir, 'evaluate_hebb_cogames.py')
    HEBB_RULE        = 'ABCD_lr'
    MISSION          = 'tutorial.scout'
    INIT_WEIGHTS     = 'ka_uni'
    FITNESS_STAT     = 'episode_reward'

    # --- Read non-evolved task config from YAML ------------------------------
    lora_rank    = 8
    w0_seed      = 12345
    eval_episodes = 3
    seed_base    = 12345
    eval_render  = False
    num_agents_override = None
    fitness_v5_weights_str = ''
    try:
        if isinstance(args.paramFile, str) and os.path.exists(args.paramFile):
            with open(args.paramFile, 'rt') as _f:
                _yaml_cfg = yaml.safe_load(_f) or {}
            _heb_lora_cfg = ((_yaml_cfg.get('main', {}) or {}).get('HebLoRA', {}) or {})
            _heb_eval_cfg = ((_yaml_cfg.get('main', {}) or {}).get('HebEval', {}) or {})
            if 'rank' in _heb_lora_cfg and _heb_lora_cfg['rank'] is not None:
                lora_rank = int(_heb_lora_cfg['rank'])
            if 'w0_seed' in _heb_lora_cfg and _heb_lora_cfg['w0_seed'] is not None:
                w0_seed = int(_heb_lora_cfg['w0_seed'])
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
    except Exception as _cfg_exc:
        print(f'WARNING: failed to parse HebLoRA/HebEval config from YAML ({_cfg_exc}); using defaults.', flush=True)

    if lora_rank <= 0:
        raise ValueError(f'Invalid LoRA rank={lora_rank}. Rank must be > 0.')

    eval_episodes = max(1, int(eval_episodes))
    print(f'HebLoRA config: rank={lora_rank}, w0_seed={w0_seed}', flush=True)

    # --- Build family arrays from beta * frozen W0 + LoRA -------------------
    if 'param' not in param or not isinstance(param['param'], dict):
        raise ValueError("Gene payload missing 'param' dictionary.")

    gene_dict = param['param']

    def _get_beta_scalar(family_name):
        beta_key = f'beta_{family_name}'
        if beta_key not in gene_dict:
            raise ValueError(f"Missing required gene key '{beta_key}'.")
        beta_value = np.asarray(gene_dict[beta_key], dtype=np.float32).ravel()
        if beta_value.size != 1:
            raise ValueError(f"Gene key '{beta_key}' must be scalar; got size={beta_value.size}.")
        return float(beta_value[0])

    def _get_lora_factors(family_name, layer_idx, d_in, d_out):
        layer_num = layer_idx + 1
        u_key = f'{family_name}_l{layer_num}_u'
        v_key = f'{family_name}_l{layer_num}_v'

        if u_key not in gene_dict:
            raise ValueError(f"Missing required gene key '{u_key}'.")
        if v_key not in gene_dict:
            raise ValueError(f"Missing required gene key '{v_key}'.")

        u_arr = np.asarray(gene_dict[u_key], dtype=np.float32).ravel()
        v_arr = np.asarray(gene_dict[v_key], dtype=np.float32).ravel()

        expected_u = d_in * lora_rank
        expected_v = lora_rank * d_out

        if u_arr.size != expected_u:
            raise ValueError(
                f"Gene key '{u_key}' has {u_arr.size} values; expected {expected_u} "
                f"for shape ({d_in}, {lora_rank})."
            )
        if v_arr.size != expected_v:
            raise ValueError(
                f"Gene key '{v_key}' has {v_arr.size} values; expected {expected_v} "
                f"for shape ({lora_rank}, {d_out})."
            )

        return u_arr.reshape(d_in, lora_rank), v_arr.reshape(lora_rank, d_out)

    family_flat_arrays = {}
    beta_values = {}
    for family_idx, family_name in enumerate(COEFF_FAMILIES):
        beta_val = _get_beta_scalar(family_name)
        beta_values[family_name] = beta_val

        layer_flats = []
        for layer_idx, (d_in, d_out) in enumerate(LAYER_SIZES):
            u_mat, v_mat = _get_lora_factors(family_name, layer_idx, d_in, d_out)

            layer_seed = int(w0_seed + family_idx * 1000 + layer_idx)
            layer_rng = np.random.default_rng(layer_seed)
            sigma_w0 = 1.0 / np.sqrt(float(d_in))
            w0 = layer_rng.normal(loc=0.0, scale=sigma_w0, size=(d_in, d_out)).astype(np.float32)

            delta = np.matmul(u_mat, v_mat).astype(np.float32)
            layer_weights = (beta_val * w0 + delta).astype(np.float32)
            layer_flats.append(layer_weights.reshape(-1))

        family_arr = np.concatenate(layer_flats, axis=0)
        if family_arr.size != TOTAL_SYNAPSES:
            raise ValueError(
                f"Constructed family '{family_name}' has size={family_arr.size}; "
                f"expected {TOTAL_SYNAPSES}."
            )
        family_flat_arrays[family_name] = family_arr

    _debug_stage(args, 'task_impl.gene_lora_reconstructed')

    # --- Assemble (123712, 5) matrix: cols = [A, B, C, lr, D] ----------------
    hebb_coeffs = np.stack([family_flat_arrays[f] for f in ['A', 'B', 'C', 'lr', 'D']], axis=1)
    assert hebb_coeffs.shape == (TOTAL_SYNAPSES, N_COEFFS), \
        f"Expected shape ({TOTAL_SYNAPSES}, {N_COEFFS}), got {hebb_coeffs.shape}"

    print(
        f"Hebb coeffs assembled: shape={hebb_coeffs.shape}  "
        f"A∈[{family_flat_arrays['A'].min():.3f}, {family_flat_arrays['A'].max():.3f}]  "
        f"B∈[{family_flat_arrays['B'].min():.3f}, {family_flat_arrays['B'].max():.3f}]  "
        f"C∈[{family_flat_arrays['C'].min():.3f}, {family_flat_arrays['C'].max():.3f}]  "
        f"D∈[{family_flat_arrays['D'].min():.3f}, {family_flat_arrays['D'].max():.3f}]  "
        f"lr∈[{family_flat_arrays['lr'].min():.5f}, {family_flat_arrays['lr'].max():.5f}]  "
        f"betas={{A:{beta_values['A']:.4f},B:{beta_values['B']:.4f},C:{beta_values['C']:.4f},D:{beta_values['D']:.4f},lr:{beta_values['lr']:.4f}}}",
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
    ]
    if num_agents_override is not None:
        cmd += ['--num_agents', str(num_agents_override)]
    if fitness_v5_weights_str:
        cmd += ['--fitness_v5_weights', fitness_v5_weights_str]
    if FITNESS_STAT == 'composite_v5_role_event':
        cmd += ['--components_sidecar', sidecar_path]
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
                failure_reason = 'evaluate-hebb-score-parse-failed'

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

    outP['raw_scores'].append(raw_score)
    if components_v5 is not None:
        outP['components_v5'] = components_v5
    print(f'Raw CoGames score: {raw_score}', flush=True)

    if failure_reason is not None:
        outP['fitnessScore'] = FAILURE_FITNESS_SCORE
        outP['failureReason'] = failure_reason
    else:
        # L2 penalty normalized by squared max reconstructed weight magnitude
        # so its scale matches the raw fitness signal (~0.01-0.05). See
        # task_cogames_hebb.py for rationale.
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
