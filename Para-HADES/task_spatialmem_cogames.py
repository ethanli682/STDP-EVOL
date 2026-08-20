import os, sys, json, subprocess, signal, argparse, time, pickle, zstandard, shutil, tempfile
from datetime import datetime
from slurmHPCHelper import *
from GA_utils_misc_func import pickle_loads_compat
import yaml

global args
global taskPath
global params_config

import numpy as np

# ============================================================================
# task_spatialmem_cogames.py
# ----------------------------------------------------------------------------
# Para-HADES task module that evolves the weights of a CoGameAgent policy net
# ("the gene") by scoring each candidate gene through a STANDALONE CoGameAgent
# scorer subprocess. The two repositories stay DECOUPLED: this module never
# imports CoGameAgent training code -- it only writes the gene to a temp .npy,
# runs the scorer, and reads a single fitnessScore back.
#
# Contract this module drives (provided by CoGameAgent, treated as given):
#   python3 -m scripts.stage_b.parahades_score \
#       --gene <gene.npy> --net spatialmem|flat --fitness ceiling|league \
#       --board machina_1 --seeds 42,43,44,45,46 --steps 10000 --n-miners 4 \
#       --out <fitness.json>
#   writes: {"fitnessScore": <float>, "detail": {...}, "gene_len": <int>, "ok": true}
#
# Everything except run_task() is modeled 1:1 on task_ANN.py so the GA.py
# master/worker two-phase loop and preemption/requeue behaviour are unchanged.
# ============================================================================


# ---- scorer configuration ---------------------------------------------------
# Read from the task YAML under a top-level `scorer:` block. Every field has a
# default matching the CoGameAgent §1 contract, so a minimal YAML still works.
_SCORER_DEFAULTS = {
    'python':     'python3',
    'module':     'scripts.stage_b.parahades_score',
    'repo_path':  '/cluster/tufts/levinlab/hhazan01/git_repo/CoGameAgent',
    # Extra dirs prepended to PYTHONPATH for the scorer (e.g. vendored cluster libs).
    'extra_pythonpath': '/cluster/tufts/levinlab/hhazan01/git_repo/CoGameAgent/.cluster_pylibs',
    'net':        'spatialmem',      # spatialmem (928764) | flat (448652)
    'fitness':    'ceiling',         # ceiling | league  (see work order §3)
    'board':      'machina_1',
    'seeds':      '42,43,44,45,46',
    'steps':      10000,
    'n_miners':   4,
    # Which key inside newMember[i]['param'] holds the flat gene vector.
    # Must match the single variable key declared in the YAML `param:` block.
    'gene_var':   'genome',
    'timeout_s':  0,                 # 0 = no timeout
}


def _load_scorer_cfg(args):
    cfg = dict(_SCORER_DEFAULTS)
    try:
        with open(args.paramFile, 'rt') as f:
            y = yaml.load(f, Loader=yaml.CSafeLoader)
        if isinstance(y, dict) and isinstance(y.get('scorer'), dict):
            cfg.update({k: v for k, v in y['scorer'].items() if v is not None})
    except Exception as e:
        print(f'[task_spatialmem_cogames] WARN could not read scorer cfg from '
              f'{args.paramFile}: {e} -- using defaults', flush=True)
    return cfg


def run_task(param, args):
    global taskPath

    # show I am alive!
    try:
        with open(taskPath + '/' + str(args.agent_test_num) + '.live.lock', 'wt') as f:
            f.write('.')
    except Exception:
        print('Problem creating agent ' + str(args.agent_idx) + ', task number' + str(args.agent_test_num) +
              ' lock file: ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock', flush=True)
        return

    print(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()

    # -----------------------------------------------------------------------
    cfg = _load_scorer_cfg(args)

    # (a) pull the candidate gene vector out of the decoded newMember record.
    #     GA.py stores geneTOparam(gene)[1] as newMember[i]['param'], keyed by the
    #     YAML variable key (cfg['gene_var']). It is a flat list in WEIGHT space
    #     (already mapped from [-1,1] back through the bounds by geneTOparam).
    gene_var = cfg['gene_var']
    try:
        gene_vec = np.asarray(param['param'][gene_var], dtype=np.float32).reshape(-1)
    except Exception as e:
        print(f'[task_spatialmem_cogames] ERROR reading param["param"]["{gene_var}"]: {e}', flush=True)
        _drop_lock(args)
        return {'fitnessScore': None}

    # (b) stage the gene into a private temp workspace and define the output path.
    workdir = tempfile.mkdtemp(prefix=f'scorer_a{args.agent_idx}_t{args.agent_test_num}_', dir=taskPath)
    gene_path = os.path.join(workdir, 'gene.npy')
    out_path = os.path.join(workdir, 'fitness.json')
    np.save(gene_path, gene_vec)

    # (c) subprocess the CoGameAgent scorer with PYTHONPATH = repo + extra libs.
    cmd = [
        str(cfg['python']), '-m', str(cfg['module']),
        '--gene', gene_path,
        '--net', str(cfg['net']),
        '--fitness', str(cfg['fitness']),
        '--board', str(cfg['board']),
        '--seeds', str(cfg['seeds']),
        '--steps', str(cfg['steps']),
        '--n-miners', str(cfg['n_miners']),
        '--out', out_path,
    ]
    env = os.environ.copy()
    pp_parts = [p for p in [cfg.get('repo_path'), cfg.get('extra_pythonpath'), env.get('PYTHONPATH')] if p]
    env['PYTHONPATH'] = ':'.join(pp_parts)
    # ES workers run single-threaded (project convention for Salimans-ES workers).
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'

    fitness_score = None
    detail = None
    gene_len = None
    timeout = float(cfg.get('timeout_s') or 0) or None
    try:
        proc = subprocess.run(
            cmd, cwd=cfg['repo_path'], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            print(f'[task_spatialmem_cogames] scorer exit={proc.returncode}\n'
                  f'--- scorer stdout (tail) ---\n{proc.stdout[-2000:]}\n'
                  f'--- scorer stderr (tail) ---\n{proc.stderr[-2000:]}', flush=True)
        else:
            with open(out_path, 'rt') as f:
                data = json.load(f)
            if not data.get('ok', False):
                print(f'[task_spatialmem_cogames] scorer reported ok=false: {data}', flush=True)
            gene_len = data.get('gene_len')
            detail = data.get('detail')
            if data.get('fitnessScore') is not None:
                fitness_score = float(data['fitnessScore'])
    except subprocess.TimeoutExpired:
        print(f'[task_spatialmem_cogames] scorer TIMEOUT after {timeout}s', flush=True)
    except Exception as e:
        print(f'[task_spatialmem_cogames] scorer subprocess error: {e}', flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    outP = dict()
    outP['fitnessScore'] = fitness_score       # None on failure -> GA skips this member
    outP['gene_len'] = gene_len
    if detail is not None:
        outP['detail'] = detail

    print(f'Task fitnessScore={fitness_score} gene_len={gene_len}', flush=True)
    print(f'---- End Task ----- {datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    print(f'---- Task Total Time ----- {(time.time() - start_time)} sec', flush=True)

    _drop_lock(args)
    return outP


def _drop_lock(args):
    try:
        os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
    except Exception:
        pass


def main(args):
    global taskPath
    global params_config

    taskPath = args.path + "/running/" + str(args.agent_idx)
    os.makedirs(taskPath, exist_ok=True)

    # load the gene from saved tasks
    try:
        with open(args.path + "/running/" + str(args.agent_idx) + '.newMember.json', 'rt') as f:
            params_config = json.load(f)
        format = 'json'
    except Exception:
        with zstandard.open(args.path + "/running/" + str(args.agent_idx) + '.newMember.pkl', 'rb') as f:
            params_config = pickle_loads_compat(f.read())
        format = 'pkl'

    for gene_num in range(len(params_config)):
        fitness = []
        for test in range(args.total_test_with_same_param):
            # -----------------------
            outP = run_task(params_config[gene_num], args)
            # -----------------------
            if outP['fitnessScore'] is not None:
                fitness.append(outP['fitnessScore'])

        if len(fitness) > 0:
            # Use mean across repeated tests to provide a single scalar fitness
            outP['fitnessScore'] = float(np.mean(fitness))

        if outP is not None:
            for k, v in outP.items():
                params_config[gene_num][k] = v

    # write resoults
    if format == 'json':
        with open(taskPath + '/' + str(args.agent_test_num) + '.resoult.json', 'wt') as f:
            json.dump(params_config, f)
    else:
        with zstandard.open(taskPath + '/' + str(args.agent_test_num) + '.resoult.pkl', 'wb') as f:
            f.write(pickle.dumps(params_config))

    if args.parallel or args.slurm:
        run(args, self=False, params=params_config[-1].copy())

    # delete the gene file
    os.system('rm ' + args.path + "/running/" + str(args.agent_idx) + '.newMember.' + format)

    print('Done Task: ' + str(args.agent_test_num) + ' for agent: ' + str(args.agent_idx), flush=True)
    return


def run(args, self=False, params=None):
    # OVERloading agent_counter
    args.agent_counter = str(args.agent_counter) + '.' + str(args.epoch)

    if self:
        run_process = [
            __file__,
            "--callBackScript", '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"', ''),
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
            '"' + args.callBackScript + '"' if args.parallel else args.callBackScript.replace('"', ''),
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

        job_name = args.path.split('/')[-1]
        if job_name == '':
            job_name = args.path.split('/')[-2]
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
            singularity_image=params['main']['Task_resource'].get('singularity_image', '/cluster/tufts/levinlab/hhazan01/singularity/delayW.sif') if self else params['main']['Main_resource'].get('singularity_image', '/cluster/tufts/levinlab/hhazan01/singularity/delayW.sif'),
            # Inject the CoGameAgent repo (+ vendored cluster libs) so the scorer
            # subprocess can import the game code inside the container.
            extra_pythonpath=params['main']['Task_resource'].get('extra_pythonpath', '') if self else params['main']['Main_resource'].get('extra_pythonpath', ''),
            command="python " + str(run_process).replace('[', '').replace(']', '').replace('\', \'', ' ').replace('\'', ''),
            nextTask=temp_file,
        )

        # create a file with the slurm script
        with open(temp_file, 'w') as file:
            file.write(script)
            file.flush()

    elif args.parallel:
        run_process = str(['python3'] + run_process).replace('[', '').replace(']', '').replace('\'', '').replace(',', ' ')
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


signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
signal.signal(signal.SIGINT, signal_handler_SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='population')
    parser.add_argument("--agent_test_num", type=int, default=-1)
    parser.add_argument("--agent_idx", type=int, default=-1)
    parser.add_argument("--agent_counter", type=str, default='-1.0')
    parser.add_argument("--total_test_with_same_param", type=int, default=1)
    parser.add_argument("--total_testsGene_with_same_agent", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='False')
    parser.add_argument("--slurm", type=str, default='False')
    parser.add_argument("--parallel", type=str, default='True')
    parser.add_argument("--callBackScript", type=str, default=None)
    parser.add_argument("--evolutionTarget", type=int, default=1)      # 1 = Max, -1 = Min
    parser.add_argument("--paramFile", type=str, default=None)

    args = parser.parse_args()

    args.gpu = True if args.gpu == 'True' or args.gpu == 'true' else False
    args.slurm = True if args.slurm == 'True' or args.slurm == 'true' else False
    args.parallel = True if args.parallel == 'True' or args.parallel == 'true' else False

    if isinstance(args.paramFile, str) and args.paramFile.strip().lower() in ['', 'none', 'null']:
        args.paramFile = None
    if args.paramFile is None:
        task_root, _ = os.path.splitext(os.path.abspath(__file__))
        default_yaml = task_root + '.yaml'
        default_yml = task_root + '.yml'
        args.paramFile = default_yaml
        if (not os.path.exists(default_yaml)) and os.path.exists(default_yml):
            args.paramFile = default_yml

    if args.callBackScript is not None:
        args.callBackScript = args.callBackScript.replace('"', '')

    # OVERloading agent_counter
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.slurm:
        args.parallel = False

    main(argparse._copy_items(args))
