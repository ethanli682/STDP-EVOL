global args
global params_config
import argparse
import copy
import json, yaml
from datetime import datetime
import pickle, zstandard
import multiprocessing as mp
import shutil
import signal
import subprocess
import os, sys
import numpy as np
import glob
import time
from tqdm import tqdm
import math
import random

from slurmHPCHelper import *
from GA_utils_misc_func import fast_converter, pickle_loads_compat
from GA_class import *
from GA_class_RevFF import *
from GA_utils_history_loader import *
from GA_utils_plotting import *
from GA_utils_diversity_calc import *
from GA_utils_framed_records import FramedRecordWriter, framed_paths_from_csv
                 

dispatch_skipped_no_candidates = 0


def save(obj, file_name):
    with open(file=file_name, mode='wb')as f:
        pickle.dump(obj=obj, file=f, protocol=pickle.HIGHEST_PROTOCOL)
     
def load(file_name):
    with open(file=file_name, mode='rb') as f:
       obj = pickle.load(file=f)
    return obj


def _atomic_write_json(path, obj):
    tmp_path = path + f".tmp.{os.getpid()}.{int(time.time()*1e6)}"
    with open(tmp_path, 'wt') as f:
        json.dump(obj, f, default=fast_converter)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_zstd_pickle(path, obj):
    tmp_path = path + f".tmp.{os.getpid()}.{int(time.time()*1e6)}"
    with open(tmp_path, 'wb') as raw_f:
        with zstandard.open(raw_f, 'wb') as f:
            f.write(pickle.dumps(obj))
        raw_f.flush()
        os.fsync(raw_f.fileno())
    os.replace(tmp_path, path)


def _write_new_member(path_prefix, payload):
    """Write the candidate payload for one agent as JSON, always.

    newMember files are the hand-off from the orchestrator to the task and are
    short-lived (the task deletes its own after loading), so the compact pickle
    form bought little and made the genes unreadable without a loader script.
    `ea.geneFormat` still governs the *history* (.log.framed.bin), which stays
    pickled. Every task_*.py reader already tries .json before falling back to
    .pkl, so nothing task-side needs to change.
    """
    _atomic_write_json(path_prefix + '.newMember.json', payload)
    # A run started before this change (or resumed from an older checkpoint) can
    # leave a stale .newMember.pkl next to the new .json. The readers prefer the
    # .json, but drop the pickle so the two can never disagree.



def _claim_run_directory(args):
    """Bind a run directory to one task, and refuse to let a different one in.

    --path/--taskFilename/--paramFile used to default to
    ../Para-HADES_Tasks/test_directory + task_FindSeq, so a GA.py launched
    without them quietly started a FindSeq run inside whatever directory another
    experiment was already using. Because every generated .sh re-sbatches itself
    (nextTask=temp_file), that stray chain then kept resurfacing for days. The
    defaults are gone now; this is the second line of defense for a directory
    that gets reused by hand.
    """
    identity = {
        'taskFilename': os.path.basename(str(args.taskFilename)),
        'paramFile': os.path.basename(str(args.paramFile)),
    }
    marker = args.path + '/task.json'
    if os.path.exists(marker):
        try:
            with open(marker, 'rt') as f:
                existing = json.load(f)
        except Exception:
            existing = None
        if existing and existing != identity:
            raise SystemExit(
                "GA - ABORT: run directory '" + args.path + "' belongs to "
                + str(existing.get('taskFilename')) + " / " + str(existing.get('paramFile'))
                + ", but this invocation is " + identity['taskFilename'] + " / "
                + identity['paramFile'] + ". Use a different --path, or delete "
                + marker + " if the reuse is deliberate."
            )
        return
    _atomic_write_json(marker, identity)


def _load_post_fitness_hook(params_config, paramFile):
    """Optional generic plug-in for task-specific fitness post-processing.

    If the task yaml specifies:

      post_fitness_hook:
        module:   <importable module name in the task dir>
        function: <callable in that module>

    return the resolved callable; otherwise return None. The hook is
    invoked once per valid reportParam in Part 2, after fitness has been
    validated and reduced and before the log is written:

      hook(reportParam, *, agent_idx, log_path, evolutionTarget,
           params_config, **future_kwargs)

    The hook may mutate reportParam in place (e.g. modify fitnessScore,
    add diagnostic fields). It owns its own configuration (env vars,
    yaml subkeys, etc.) — GA.py stays task-agnostic.
    """
    if not isinstance(params_config, dict):
        return None
    spec = params_config.get('post_fitness_hook')
    if not spec or not isinstance(spec, dict):
        return None
    try:
        task_dir = os.path.dirname(os.path.abspath(paramFile))
        if task_dir not in sys.path:
            sys.path.insert(0, task_dir)
        import importlib
        mod = importlib.import_module(spec['module'])
        return getattr(mod, spec['function'])
    except Exception as exc:
        print(f"GA - post_fitness_hook load failed: {exc}", flush=True)
        return None


def _build_task_candidate_skeleton(params_config, agent_idx, epoch, iteration):
    candidate = {
        'fitnessScore': None,
        'epoch': epoch,
        'loss': 0.0,
        'iteration': iteration,
        'candidate_agent_idx': int(agent_idx),
        'build_in_param': {},
    }
    if isinstance(params_config, dict) and ('main' in params_config):
        candidate['main'] = params_config['main']
    return candidate


def _estimate_pickle_bytes(obj):
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1


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


def safe_pickle_loads_isolated(payload, timeout_sec=30):
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

def run(args, self=False, exc=False, params=None):
    global dispatch_skipped_no_candidates
    
    # OVERloading agent_counter
    args.agent_counter = str(args.agent_counter) + '.' + str(args.epoch)
    
    if self:
        run_process = [ 
            '"' + __file__ + '"' if args.parallel else __file__.replace('"',''),
            '--part', str(args.part),
            '--path', '"' + args.path + '"' if args.parallel else args.path,
            '--agent_idx', str(args.agent_idx),
            '--agent_counter', str(args.agent_counter),
            '--agent_test_num', str(args.agent_test_num),
            '--total_test_with_same_param', str(args.total_test_with_same_param),
            '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
            '--gpu', 'True' if args.gpu else 'False',
            '--slurm', 'True' if args.slurm else 'False',
            '--parallel', 'True' if args.parallel else 'False',
            '--taskFilename', '"' + args.taskFilename + '"' if args.parallel else args.taskFilename,
            '--evolutionTarget', str(args.evolutionTarget),
            '--paramFile', args.paramFile,            
            ]
    else:
        run_process = [ 
                '"' + args.taskFilename + '"' if args.parallel else args.taskFilename.replace('"',''),
                '--callBackScript', '"' + __file__ + '"' if args.parallel else __file__,
                '--path', '"' + args.path + '"' if args.parallel else args.path,
                '--agent_idx', str(args.agent_idx),   
                '--agent_counter', str(args.agent_counter),
                '--agent_test_num', str(args.agent_test_num),
                '--total_test_with_same_param', str(args.total_test_with_same_param),
                '--total_testsGene_with_same_agent', str(args.total_testsGene_with_same_agent),
                '--gpu', 'True' if args.gpu else 'False',
                '--slurm', 'True' if args.slurm else 'False',
                '--parallel', 'True' if args.parallel else 'False',
                '--evolutionTarget', str(args.evolutionTarget),
                '--paramFile', args.paramFile,
                ]

    if (not self):
        candidate_json = args.path + "/running/" + str(args.agent_idx) + '.newMember.json'
        candidate_pkl = args.path + "/running/" + str(args.agent_idx) + '.newMember.pkl'
        if (not os.path.exists(candidate_json)) and (not os.path.exists(candidate_pkl)):
            dispatch_skipped_no_candidates += 1
            print(
                "GA - Dispatch - Skip launch for agent " + str(args.agent_idx) +
                " (no candidates: " + candidate_json + " | " + candidate_pkl + ")",
                flush=True,
            )
            print(
                "GA - Part 2 - SkipCounter no_candidate_dispatch=" + str(dispatch_skipped_no_candidates),
                flush=True,
            )
            return

    if args.slurm:        
        # make sure the log directory exists
        os.makedirs(args.path + '/scripts/', exist_ok=True)
        temp_file = args.path + '/scripts/Agent_idx_' + str(args.agent_idx) + '_Task_' + str(args.agent_test_num) 
        temp_file += '_Epoch_' + str(args.epoch) + '.sh'

        job_name  = args.path.split('/')[-1]
        if job_name == '':
            job_name = args.path.split('/')[-2]

        script  = SlurmScript(
            jobName=('A_' if self else 'T_') + job_name,
            jobTime=params['main']['Main_resource']['time'] if self else params['main']['Task_resource']['time'], # 00:30:00, 01:00:00, 02:00:00, 04:00:00
            jobMemory=params['main']['Main_resource']['memory'] if self else params['main']['Task_resource']['memory'], # 32768, 65536, 131072, 262144
            output_path=args.path + (params['main']['Main_resource']['path'] if self else params['main']['Task_resource']['path']),
            jobCPUs= str(params['main']['Main_resource']['cpu'] if self else params['main']['Task_resource']['cpu']),
            # theadsPerCore= str(params['main']['Main_resource']['theadsPerCore'] if self else params['main']['Task_resource']['theadsPerCore']),
            jobGPUs=(params['main']['Main_resource']['gpu'] if self else params['main']['Task_resource']['gpu']),
            excludeNodes= params['main']['Main_resource']['exclude_nodes'] if self else params['main']['Task_resource']['exclude_nodes'],
            condaEnv=params['main']['Main_resource']['conda_env'] if self else params['main']['Task_resource']['conda_env'],
            partition=params['main']['Main_resource']['partition'] if self else params['main']['Task_resource']['partition'],
            # run_mode gates singularity (default, OLD cluster) vs conda (NEW cluster).
            # Absent from the yaml -> 'singularity' -> unchanged behavior for every existing run.
            run_mode=params['main']['Main_resource'].get('run_mode', 'singularity') if self else params['main']['Task_resource'].get('run_mode', 'singularity'),
            command="python3 " + str(run_process).replace('[','').replace(']','').replace('\', \'',' ').replace('\'',''),
            nextTask=temp_file,
        )
    
        if exc:
            SubmitJob(script, temp_file)
        else:
            with open(temp_file, 'wt') as f:
                f.write(script)
                f.flush()
    else:
        temp = run_process
        run_process = ['python3']
        run_process.extend(temp)
        print("---CMD----> " + " ".join(run_process), flush=True)
        if args.parallel:
            run_process = " ".join(run_process)
            subprocess.Popen(run_process, shell=True)
        else:
            # wait for task to compiter 
            task = subprocess.Popen(run_process)
            task.wait()


def signal_handler_SIGTERM(sig, frame):
    global params_config

    pid = os.getpid()
    sig_args = globals().get('args', None)
    sig_params = globals().get('params_config', None)

    print(
        f'SIGTERM received (signal={sig}) pid={pid} '
        f'args_set={sig_args is not None} params_set={sig_params is not None} '
        f'paramFile={getattr(sig_args, "paramFile", None)} '
        f'part={getattr(sig_args, "part", None)} '
        f'agent_idx={getattr(sig_args, "agent_idx", None)} '
        f'agent_counter={getattr(sig_args, "agent_counter", None)} '
        f'epoch={getattr(sig_args, "epoch", None)}',
        flush=True,
    )

    try:
        # Fallback: load params from YAML if params_config not yet set in globals
        if sig_params is None and sig_args is not None and getattr(sig_args, 'paramFile', None):
            try:
                with open(sig_args.paramFile, 'rt') as f:
                    sig_params = yaml.load(f, Loader=yaml.CSafeLoader)
                print(f'SIGTERM handler: pid={pid} loaded params fallback from {sig_args.paramFile}', flush=True)
            except Exception as e:
                print(f'SIGTERM handler: pid={pid} params fallback failed: {e}', flush=True)

        if sig_args is None or sig_params is None:
            print(f'SIGTERM handler: pid={pid} missing runtime context, cannot requeue', flush=True)
        else:
            run(args=copy.copy(sig_args), self=True, exc=True, params=sig_params.copy())
            print(f'SIGTERM handler: pid={pid} requeue submitted', flush=True)
    except Exception as e:
        print(f'SIGTERM handler: pid={pid} error: {e}', flush=True)

    print(f'SIGTERM handler: pid={pid} exiting', flush=True)
    # Use os._exit to terminate immediately and avoid atexit callback errors
    # (sys.exit raises SystemExit which can be swallowed or cause torch cleanup crashes)
    os._exit(0)


def _install_signal_handlers():
    """Register SIGTERM/SIGINT/SIGUSR1 handlers. Called from __main__ AFTER args
    is parsed so the handler always has a valid args Namespace in globals().
    """
    signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
    signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
    signal.signal(signal.SIGINT, signal_handler_SIGTERM)


def _dispatch_evolution_algorithm(args, ea, build_in_params, number_of_candidate_genes, verbose=3, genePopulation=None):
    """Pick one algorithm from args.evolutionType (comma-separated) and return candidate genes.

    Shared by Part 1 (resume with sufficient pool) and Part 2. Keeps the algorithm dispatch
    in one place so adding/removing an algorithm does not require two edits.
    """
    evolution_types = [t.strip() for t in args.evolutionType.split(',')]
    selected = random.choice(evolution_types)
    if len(evolution_types) > 1:
        print(f"GA - Dispatch - Agent - {args.agent_idx} - Randomly selected evolution type: {selected} (from: {args.evolutionType})", flush=True)

    tag = f"GA - Dispatch - Agent - {args.agent_idx} - Using {selected} to generate new genes"

    if selected == 'GA':
        from GA_class_GA import createCandidateGene_GeneticAlgorithm as fn
    elif selected == 'CMA_ES':
        from GA_class_CMA_ES import createCandidateGene_CMA_ES as fn
    elif selected == 'SNES':
        from GA_class_SNES import createCandidateGene_SNES as fn
    elif selected == 'OpenAI_ES':
        from GA_class_OpenAI_ES import createCandidateGene_OpenAI_ES as fn
    elif selected == 'RevFF_predictor_MinHeap':
        from GA_class_RevFF_MinHeap import createCandidateGeneUsingFFPredictor_MinHeap as fn
    elif selected == 'RevFF_predictor_chunk':
        from GA_class_RevFF import createCandidateGeneUsingFFPredictor_chunk as fn
    elif selected == 'MultiHeadPredictor':
        from GA_class_MultiHeadPredictor import createCandidateGeneUsingMultiHeadPredictor_chunk as fn
    elif selected == 'CondVAE':
        from Alternative_Approach.ConditionalVAE import createCandidateGeneUsingMultiHeadPredictor_chunk as fn
    elif selected == 'HADES_Gemini':
        from GA_class_CondEvo_Gemini import createCandidateGeneUsingHADES as fn
    elif selected == 'HADES_Claude':
        from GA_class_CondEvo_Claude import createCandidateGeneUsingHADESlution as fn
    elif selected == 'DiffEvo_Gemini':
        from GA_class_DiffEvo_Gemini import createCandidateGene_Diffusion as fn
    elif selected == 'DiffEvo_Claude':
        from GA_class_DiffEvo_Claude import createCandidateGene_DiffusionEvolution as fn
    elif selected == 'DiffEvo_HADES_Ben':
        from condevo.GA_class_DiffEvo_HADES_Ben_Claude import createCandidateGene_CondEvo_HADES_Ben as fn
    elif selected == 'DiffEvo_CHARLES_Ben':
        from condevo.GA_class_DiffEvo_CHARLS_Ben_Claude import createCandidateGene_CondEvo_CHARLES_Ben as fn
    elif selected == 'EA_Heb':
        from GA_class_EA_Heb import createCandidateGene_EA_Heb as fn
    else:
        print(f"GA - Dispatch - Agent - {args.agent_idx} - Error - Unknown evolution type: {selected} (from: {args.evolutionType})", flush=True)
        sys.exit(1)

    print(tag, flush=True)
    kwargs = dict(
        args=args,
        EA_Class=ea,
        build_in_params=build_in_params,
        number_of_candidate_genes=number_of_candidate_genes,
        verbose=verbose,
    )
    # HADES/DiffEvo families maintain their own population state and don't accept genePopulation.
    _no_pop_algos = {'HADES_Gemini', 'HADES_Claude', 'DiffEvo_Gemini', 'DiffEvo_Claude', 'DiffEvo_HADES_Ben', 'DiffEvo_CHARLES_Ben'}
    if selected not in _no_pop_algos:
        kwargs['genePopulation'] = genePopulation
    return fn(**kwargs)


def main(args):
    global params_config
    while(True):

        if args.part == 1: # init and spawn process
            print("GA - Part 1 - Start", flush=True)
            # make automatic backup of the directory code in the running directory
            os.makedirs(args.path, exist_ok=True)
            _claim_run_directory(args)
            os.makedirs(args.path + "/founders", exist_ok=True)
            os.makedirs(args.path + "/graphs", exist_ok=True)

            # log the excution command and the parameters
            with open(args.path + '/command.txt', 'wt') as f:
                f.write(' '.join(sys.argv) + '\n')
                f.write(' '.join([str(k) + '=' + str(v) for k,v in vars(args).items()]) + '\n')
                f.flush()
            os.system(
                'tar --exclude=logs_* --exclude=MNIST_data  --exclude=Run_* --exclude=.git ' \
                '--exclude=.vscode --exclude=RunDTask* -czvf ' + args.path + 
                '/' + datetime.now().strftime("Code_%d-%m-%Y-%H-%M-%S-%f") + '.tar.gz  .' 
                      )
            
            args.agent_counter = 0
            args.epoch = 0

            with open(args.paramFile, 'rt') as f:
                params_config = yaml.load(f, Loader=yaml.CSafeLoader)

            try:
                params_ML_file = args.paramFile.replace('.yaml', '_ML.yaml').replace('.yml', '_ML.yml')
                with open(params_ML_file, 'rt') as f:
                    params_ML = yaml.load(f, Loader=yaml.CSafeLoader)
            except:
                params_ML = None

            if 'ML' in params_config and params_config['ML'] is not None:
                if params_ML is None:
                    params_ML = params_config['ML']
                    params_config.pop('ML', None)
                else:
                    print("Error: Both paramFile and paramFile_ML exist. Using paramFile_ML for ML parameters.", flush=True)
                    exit()

            tmp_populationSize = args.populationSize
            args.populationSize = args.populationSize +  args.num_of_agents * args.total_testsGene_with_same_agent
            ea = difEvoInit(vars_and_bounds=params_config['param'], params_ML=params_ML, args=args, geneMin=args.geneMin)
            args.populationSize = tmp_populationSize
            del tmp_populationSize

            # Register full-params reconstruction hook if the task YAML specifies it
            _founder_cfg = params_config.get('founder', {}) or {}
            _fp_spec = _founder_cfg.get('full_params_fn', None)
            if _fp_spec:
                import importlib as _il
                import sys as _sys, os as _os
                _task_dir = _os.path.dirname(_os.path.abspath(args.paramFile))
                if _task_dir not in _sys.path:
                    _sys.path.insert(0, _task_dir)
                _mod = _il.import_module(_fp_spec['module'])
                _raw_fn = getattr(_mod, _fp_spec['function'])
                ea.full_params_fn = lambda rec, _fn=_raw_fn, _cfg=params_config: _fn(rec, _cfg)

            print("GA - Part 1 - Save Objec", flush=True)

            save(obj=[
                        params_config, ea.params_ML, args.evolutionTarget, ea.geneLength, args.populationSize, args.numOFgenerations, 
                        args.evolutionType, ea.geneMin,
                    ], file_name=args.path + '/ga_running_obj.pkl')

            print("GA - Part 1 - Loading Gene Population", flush=True)

            # Load all population
            loader = ChunkedGeneHistoryLoader(
                savePath=args.path,
                agent_id=-1,
                geneFormat=ea.geneFormat,
                required_keys_and_types=['fitnessScore', 'epoch', 'iteration', 'loss'],
                shuffle=True
            )
            genePopulation = loader.loadAllGeneHistory(
                args, allEpochs=True
            )
            
            print(f"GA - Part 1 - Gene size = {len(genePopulation)}", flush=True)

            print("GA - Part 1 - Check previous epoch", flush=True)
            
            # check if there are genes from the last round            
            none_genePopulation = [g for g in genePopulation if g.get('fitnessScore') is None]

            max_epoch = args.epoch
            max_iter = args.agent_counter
            for i,p in enumerate(genePopulation):
                if 'epoch' in p and 'iteration' in p and not np.isnan(p['iteration']):
                    if i==0:
                        print("GA - Part 1 - 1", flush=True)
                        max_epoch = p['epoch']
                        max_iter = p['iteration']
                    else:
                        if p['epoch'] > max_epoch or (p['epoch'] == max_epoch and p['iteration'] > max_iter):
                            print("GA - Part 1 - 2", flush=True)
                            max_epoch = p['epoch']
                            max_iter = p['iteration']

            print("GA - Part 1 - 2 Applying changes from the last run", flush=True)
            args.epoch = max_epoch
            args.agent_counter = max_iter
     
            args.part = 2
            agents = np.random.choice(args.num_of_agents, args.num_of_agents, replace=False)   
            
            print("GA - Part 1 - Spawn Process", flush=True)

            # We need the total count of agents to divide work evenly
            total_genes = len(genePopulation) - len(none_genePopulation)
            minimum_population = max(int(args.populationSize * 0.2), 100)

            print(f"GA - Part 1 - Total genes with fitness: {total_genes}", flush=True)

            # ── Decide gene source for initial spawn ──
            # Under-populated pool → random genes to bootstrap.
            # Sufficient pool → use the evolution algorithm so the resume continues from
            # where the last run left off (same gene pool as Part 2 will use next cycle).
            args._allEpoch_gene_count = total_genes  # pass all-epoch count to Part 2 gate
            use_evolution = total_genes >= minimum_population

            if use_evolution:
                print(f"GA - Part 1 - Resuming: population ({total_genes}) >= minimum ({minimum_population}). Using evolution algorithm for initial candidates.", flush=True)
                total_tasks_per_agent = args.total_testsGene_with_same_agent
            elif total_genes > 0:
                if args.all_in_first_run == 1:
                    total_tasks_per_agent = max(math.ceil(total_genes / args.num_of_agents), 1)
                else:
                    total_tasks_per_agent = args.total_testsGene_with_same_agent
            else:
                total_tasks_per_agent = args.total_testsGene_with_same_agent if args.all_in_first_run == 0 else max(minimum_population // args.num_of_agents, 1)

            print(f"GA - Part 1 - Total tasks per agent: {total_tasks_per_agent} (source: {'evolution' if use_evolution else 'random'})", flush=True)

            # When using evolution, keep genePopulation in memory and reuse it across all
            # agents so the history is loaded once (not 50×). Filter out None-fitness
            # entries that were used only for epoch detection.
            reusable_gene_population = None
            if use_evolution:
                reusable_gene_population = [g for g in genePopulation if g.get('fitnessScore') is not None]
                print(f"GA - Part 1 - Reusing cached gene population across all agents (size={len(reusable_gene_population)})", flush=True)
            del genePopulation

            for args.agent_idx in tqdm(agents):
                param = []
                temp_copy_params_config = params_config.copy()

                if use_evolution:
                    build_in_params_default = ea.bulidin_params()
                    gene_candidates = _dispatch_evolution_algorithm(
                        args, ea,
                        build_in_params=build_in_params_default,
                        number_of_candidate_genes=total_tasks_per_agent,
                        genePopulation=reusable_gene_population,
                    )
                    for idx in range(total_tasks_per_agent):
                        gene = gene_candidates[idx]
                        param.append(_build_task_candidate_skeleton(temp_copy_params_config, args.agent_idx, args.epoch, args.agent_counter))
                        temp_param_list = dict()
                        temp_param_list, param[-1]['param'] = ea.geneTOparam(gene.copy() if hasattr(gene, 'copy') else gene, param_list=temp_param_list)
                        param[-1]['gene'] = gene
                        param[-1]['loss'] = 0.0
                        param[-1]['build_in_param'] = ea.bulidin_params()
                else:
                    for idx in range(total_tasks_per_agent):
                        gene = ea.generateRandomGene()
                        param.append(_build_task_candidate_skeleton(temp_copy_params_config, args.agent_idx, args.epoch, args.agent_counter))
                        temp_param_list = dict()
                        temp_param_list, param[-1]['param'] = ea.geneTOparam(gene.copy(), param_list=temp_param_list)
                        build_in_params = ea.bulidin_params()
                        build_in_param, param[-1]['param'] = ea.extractBuildInParamFromGene(parameters=param[-1]['param'], build_in_params=build_in_params)
                        param[-1]['gene'] = gene
                        param[-1]['loss'] = 0.0
                        param[-1]['build_in_param'] = build_in_param

                # save mutation (This part remains unchanged)
                if len(param) == 0:
                    print(f"GA - Part 1 - Agent - {args.agent_idx} - ERROR: param is empty (total_tasks_per_agent=0?), skipping write", flush=True)
                    continue

                # Clean any stale state from a prior crashed run for this agent before spawning.
                agent_running_dir = args.path + "/running/" + str(args.agent_idx)
                shutil.rmtree(agent_running_dir, ignore_errors=True)
                os.makedirs(agent_running_dir, exist_ok=True)

                est_bytes = _estimate_pickle_bytes(param)
                print(
                    f"GA - Part 1 - Agent - {args.agent_idx} - Candidate payload entries={len(param)} est_pickle_bytes={est_bytes}",
                    flush=True,
                )
                _write_new_member(args.path + "/running/" + str(args.agent_idx), param)

                run(args=copy.copy(args), self=False, exc=True, params=temp_copy_params_config)
                if args.parallel:
                    time.sleep(5)  # slight delay to avoid overwhelming the system

            del param
            del temp_copy_params_config

            print("GA - Part 1 - Done", flush=True)
            if args.parallel or args.slurm:
                return

        if args.part == 2: # individual run # summrize individual run
            _claim_run_directory(args)
            agents = [args.agent_idx] if (args.parallel or args.slurm) else np.random.choice(args.num_of_agents, args.num_of_agents, replace=False)
            has_any_valid_fitness = False

            # Load the orchestrator state once per Part 2 invocation rather than per agent.
            [
                params_config, params_ML, args.evolutionTarget, geneLength, args.populationSize, args.numOFgenerations,
                args.evolutionType, geneMin,
            ] = load(file_name=args.path + '/ga_running_obj.pkl')

            for args.agent_idx in agents:

                print("GA - Part 2 - Agent - " + str(args.agent_idx) + " - Start", flush=True)

                ea = difEvoInit(vars_and_bounds=params_config['param'], params_ML=params_ML, args=args, geneMin=geneMin)
                ea.geneLength = geneLength

                # Register full-params reconstruction hook if the task YAML specifies it
                _founder_cfg = params_config.get('founder', {}) or {}
                _fp_spec = _founder_cfg.get('full_params_fn', None)
                if _fp_spec:
                    import importlib as _il
                    import sys as _sys, os as _os
                    _task_dir = _os.path.dirname(_os.path.abspath(args.paramFile))
                    if _task_dir not in _sys.path:
                        _sys.path.insert(0, _task_dir)
                    _mod = _il.import_module(_fp_spec['module'])
                    _raw_fn = getattr(_mod, _fp_spec['function'])
                    ea.full_params_fn = lambda rec, _fn=_raw_fn, _cfg=params_config: _fn(rec, _cfg)

                # check if all repeted files complited. 
                print("GA - Part 2 - Agent - " + str(args.agent_idx) + " - Check if all repeted files complited", flush=True)
                agent_running_dir = args.path + "/running/" + str(args.agent_idx)
                json_reports = glob.glob(agent_running_dir + "/*.resoult.json")
                pkl_reports = glob.glob(agent_running_dir + "/*.resoult.pkl")

                reports = []
                json_stems = set()
                for report_path in json_reports:
                    report_name = os.path.basename(report_path)
                    stem = report_name[:-len('.resoult.json')]
                    json_stems.add(stem)
                    reports.append(report_path)

                for report_path in pkl_reports:
                    report_name = os.path.basename(report_path)
                    stem = report_name[:-len('.resoult.pkl')]
                    if stem not in json_stems:
                        reports.append(report_path)

                reports.sort()
                    
                print("GA - Part 2 - Agent - Number of Reports = " + str(len(reports)), flush=True)

                # summerize resoults
                framed_writer = None
                if ea.geneFormat == 'pickle':
                    csv_log_path = args.path + '/' + str(args.agent_idx) + '.log.csv'
                    data_path, index_path = framed_paths_from_csv(csv_log_path)
                    framed_writer = FramedRecordWriter(
                        data_path=data_path,
                        index_path=index_path,
                        codec='pickle',
                        compression_level=3,
                    )
                returned_build_in_params = []
                skipped_unreadable_reports = 0
                skipped_malformed_records = 0
                skipped_invalid_fitness_records = 0
                valid_fitness_records = 0

                # Optional task-supplied fitness post-processor (see
                # _load_post_fitness_hook). Loaded once per Part 2.
                _post_fitness_hook = _load_post_fitness_hook(
                    params_config, args.paramFile)
                _post_fitness_log_path = (args.path + '/'
                                          + str(args.agent_idx) + '.log.csv')

                for reportFile in reports:
                    if reportFile.endswith('.resoult.json'):
                        try:
                            with open(reportFile, 'rt') as f:
                                reportParams = json.load(f)
                        except Exception as e:
                            skipped_unreadable_reports += 1
                            print(f"GA - Part 2 - Skip unreadable report: {reportFile} ({e})", flush=True)
                            continue
                    elif reportFile.endswith('.resoult.pkl'):
                        try:
                            with zstandard.open(reportFile, 'rb') as f:
                                payload = f.read()
                            reportParams = safe_pickle_loads_isolated(payload, timeout_sec=20)
                        except Exception as e:
                            skipped_unreadable_reports += 1
                            print(f"GA - Part 2 - Skip unreadable report: {reportFile} ({e})", flush=True)
                            continue
                    else:
                        print(f"GA - Part 2 - Skip unknown report format: {reportFile}", flush=True)
                        continue

                    if isinstance(reportParams, dict):
                        reportParams = [reportParams] # this is a fix, I dont know if its the intended behaviour but it works -C
                    elif not isinstance(reportParams, list):
                        skipped_malformed_records += 1
                        print(f"GA - Part 2 - Skip malformed report payload: {reportFile}", flush=True)
                        continue

                    for rep_n, reportParam in enumerate(reportParams):
                        if not isinstance(reportParam, dict):
                            skipped_malformed_records += 1
                            continue

                        fitness_value = reportParam.get('fitnessScore', None)
                        fitness_valid = False
                        _individual_valid_scores = []
                        if fitness_value is not None:
                            try:
                                raw_list = fitness_value if isinstance(fitness_value, list) else [fitness_value]
                                valid_scores = [v for v in raw_list if v is not None]
                                failed_count = len(raw_list) - len(valid_scores)
                                if failed_count > 0 and len(valid_scores) < len(raw_list):
                                    print(f"GA - Part 2 - fitnessScore: skipping {failed_count}/{len(raw_list)} None (failed) seeds", flush=True)
                                if valid_scores:
                                    fitness_arr = np.array(valid_scores, dtype=float)
                                    fitness_mean = float(np.mean(fitness_arr))
                                    if not np.isnan(fitness_mean):
                                        fitness_std = float(np.std(fitness_arr))
                                        reportParam['fitnessScore'] = float(str(fitness_mean))
                                        reportParam['fitnessScore_std'] = float(str(fitness_std))
                                        fitness_valid = True
                                        _individual_valid_scores = [float(s) for s in valid_scores]
                            except Exception:
                                fitness_valid = False

                        if not fitness_valid:
                            skipped_invalid_fitness_records += 1
                            continue

                        valid_fitness_records += 1

                        if 'loss' in reportParam and reportParam['loss'] is not None:
                            reportParam['loss'] = float(str(np.mean(reportParam['loss'])))
                        else:
                            reportParam['loss'] = None

                        reportParam['epoch'] = args.epoch
                        reportParam['iteration'] = args.agent_counter

                        if 'build_in_param' not in reportParam:
                            skipped_malformed_records += 1
                            reportParam['build_in_param'] = ea.bulidin_params()

                        returned_build_in_params.append(reportParam['build_in_param'])

                        # Optional task-supplied fitness post-processor.
                        # Hook may mutate reportParam in place (modify
                        # fitnessScore, add diagnostic fields). Errors
                        # are logged but do not abort the report write.
                        if _post_fitness_hook is not None:
                            try:
                                _post_fitness_hook(
                                    reportParam,
                                    agent_idx=args.agent_idx,
                                    log_path=_post_fitness_log_path,
                                    evolutionTarget=args.evolutionTarget,
                                    params_config=params_config,
                                )
                            except Exception as _pfh_exc:
                                print(
                                    f"GA - Part 2 - Agent - {args.agent_idx} "
                                    f"- post_fitness_hook error: {_pfh_exc}",
                                    flush=True,
                                )

                        # Write ONE entry per candidate using mean fitnessScore.
                        # Previously individual seed scores were expanded into separate
                        # log entries, causing a single lucky outlier seed to promote the
                        # gene to elite status N times — corrupting all optimizer histories.
                        _write_entry = reportParam.copy()
                        # fitnessScore is already the mean; raw_scores retains per-seed data.

                        if ea.geneFormat == 'json':
                            save_dict = _write_entry.copy()
                            save_dict.pop('main', None)
                            with open(args.path + '/' + str(args.agent_idx) + '.log.csv', 'at') as f:
                                f.write(json.dumps(save_dict, default=fast_converter) + '\n')

                        elif ea.geneFormat == 'pickle':
                            # 1. Save the Heavy Pickle Data
                            data_to_save = _write_entry.copy()
                            framed_writer.append(data_to_save)

                            # 2. Save the Lightweight Text Summary (as JSON)
                            param_to_save = _write_entry.copy()
                            for k in ['gene', 'param', 'main']:
                                param_to_save.pop(k, None)
                            with open(args.path + '/' + str(args.agent_idx) + '.log.csv', 'at') as f:
                                f.write(json.dumps(param_to_save, default=fast_converter) + '\n')
                            del data_to_save, param_to_save

                        if ea.geneFormat == 'pickle':
                            print("GA - Part 2 - Agent - " + str(args.agent_idx) + " - Report nu: " + str(rep_n) + " - Save updated gene (1 entry, mean fitness)", flush=True)

                if framed_writer is not None:
                    framed_writer.close()

                print(
                    f"GA - Part 2 - Agent - {args.agent_idx} - SkipCounter unreadable_reports={skipped_unreadable_reports} malformed_records={skipped_malformed_records} invalid_fitness_records={skipped_invalid_fitness_records} no_candidate_dispatch={dispatch_skipped_no_candidates}",
                    flush=True,
                )

                if valid_fitness_records > 0:
                    has_any_valid_fitness = True
                else:
                    print(
                        f"GA - Part 2 - Agent - {args.agent_idx} - No valid fitnessScore found; log not updated and iteration will not advance for this cycle.",
                        flush=True,
                    )
                                        
                print("GA - Part 2 - Agent - " + str(args.agent_idx) + " - Pass 0", flush=True)

                genePopulationHistory = ea.saveFounder(args)
                genePopulationHistory_size = len(genePopulationHistory)

                # deterministic plotting after warmup (not random)
                if args.agent_counter > 5:

                    # Keep records with valid fitness; diversity can be missing.
                    valid_history = []
                    for g in genePopulationHistory:
                        try:
                            fs = g.get('fitnessScore', None)
                            if fs in [None, 'None']:
                                continue
                            fsf = float(fs)
                            if np.isnan(fsf):
                                continue
                            valid_history.append(g)
                        except Exception:
                            continue

                    # sort by iteration for stable deterministic plots
                    valid_history = sorted(valid_history, key=lambda k: k.get('iteration', -1))

                    if len(valid_history) > 10:
                        iterations = np.array([d.get('iteration', 0) for d in valid_history])
                        fitness_vals = np.array([float(v) if (v := d.get('fitnessScore')) is not None else np.nan for d in valid_history])
                        loss_vals = np.array([float(v) if (v := d.get('loss')) is not None else np.nan for d in valid_history])

                        diversity_vals = []
                        valid_diversity_count = 0
                        for d in valid_history:
                            dv = d.get('Average pairwise distances', np.nan)
                            try:
                                dvf = float(dv)
                                if np.isnan(dvf):
                                    diversity_vals.append(np.nan)
                                else:
                                    diversity_vals.append(dvf)
                                    valid_diversity_count += 1
                            except Exception:
                                diversity_vals.append(np.nan)

                        plot_data = {
                            'fitness Score': fitness_vals,
                            'loss': loss_vals,
                            'iteration': iterations,
                        }

                        names = [
                            ['fitness Score'],
                        ]

                        if valid_diversity_count >= 2:
                            plot_data['Average pairwise distances'] = np.array(diversity_vals)
                            names.extend([
                                ['fitness Score', 'Average pairwise distances'],
                                ['Average pairwise distances'],
                            ])

                        if not np.all(np.isnan(loss_vals)):
                            names.extend([['loss']])
                            names.extend([['fitness Score', 'loss']])

                        plotTable(args, plot_data, names)
                        del plot_data
                
                
                del genePopulationHistory
                
                print(f'E {args.epoch} C {args.agent_counter} A {args.agent_idx} ', flush=True)

                # inflate returned_build_in_params if its smaller than total_testsGene_with_same_agent
                if len(returned_build_in_params) < args.total_testsGene_with_same_agent:
                    for _ in range(args.total_testsGene_with_same_agent - len(returned_build_in_params)):
                        returned_build_in_params.append(ea.bulidin_params())

                # filter returned_build_in_params to only keep the ones for the current evolution configs
                _evolution_types = [t.strip() for t in args.evolutionType.split(',')]
                tmp_returned_build_in_params = list()
                for returned_build_in_param in returned_build_in_params:
                    filtered_params = {}
                    for _evo_type in _evolution_types:
                        if _evo_type in returned_build_in_param:
                            filtered_params[_evo_type] = returned_build_in_param[_evo_type]
                        else:
                            filtered_params[_evo_type] = {}

                    if 'loss' in returned_build_in_param:
                        filtered_params['loss'] = returned_build_in_param['loss']

                    tmp_returned_build_in_params.append(filtered_params)
                returned_build_in_params = tmp_returned_build_in_params
                
                # Use the larger of current-epoch count and all-epoch count from Part 1.
                # Algorithms use allEpochs=True internally, so blocking them based on
                # current-epoch count alone causes spurious random fallback on resume
                # or early in a new epoch.
                _effective_history_size = max(
                    genePopulationHistory_size,
                    getattr(args, '_allEpoch_gene_count', 0),
                )
                if _effective_history_size < max(int(args.populationSize*0.1), 100):
                    print(f"GA - Part 2 - Agent - {args.agent_idx} - Gene population ({genePopulationHistory_size} current-epoch, {_effective_history_size} effective) < threshold. Generating new random genes.", flush=True)
                    gene_candidates = []
                    for ng in range(args.total_testsGene_with_same_agent):
                        gene_candidates.append(ea.generateRandomGene())
                        returned_build_in_params[ng] = ea.bulidin_params()

                else:
                    gene_candidates = _dispatch_evolution_algorithm(
                        args, ea,
                        build_in_params=returned_build_in_params[0],
                        number_of_candidate_genes=args.total_testsGene_with_same_agent,
                    )


                newParam = []
                for ng in range(args.total_testsGene_with_same_agent):
                    newParam.append(_build_task_candidate_skeleton(params_config, args.agent_idx, args.epoch, args.agent_counter))
                    temp_param_list = dict()
                    temp_param_list, newParam[-1]['param'] = ea.geneTOparam(gene_candidates[ng], param_list=temp_param_list)
                    newParam[-1]['build_in_param'] = returned_build_in_params[ng]
                    # Keep native gene representation (same pattern as Part 1) to avoid huge list payloads.
                    newParam[-1]['gene'] = gene_candidates[ng]
                    if 'loss' in returned_build_in_params[0]:
                        newParam[-1]['loss'] = returned_build_in_params[0]['loss']
                    else:
                        newParam[-1]['loss'] = 0.0                

                # add to the last newParam the deversityScore
                deversityMatrix = calculateDiversityMatrix(args=args, geneFormat=ea.geneFormat)
                # save the gene
                if deversityMatrix is not None:
                    for k,v in deversityMatrix.items():
                        if isinstance(v, (int, float, np.integer, np.floating, bool)):
                            newParam[0][k] = float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v

                # deleate files in the agent running directory 
                shutil.rmtree(args.path + "/running/" + str(args.agent_idx), ignore_errors=True)

                # save mutation
                if len(newParam) == 0:
                    print(f"GA - Part 2 - Agent - {args.agent_idx} - ERROR: newParam is empty, skipping write to avoid corrupt newMember file", flush=True)
                    continue
                os.makedirs(args.path + "/running/" + str(args.agent_idx), exist_ok=True)
                est_bytes = _estimate_pickle_bytes(newParam)
                print(
                    f"GA - Part 2 - Agent - {args.agent_idx} - Candidate payload entries={len(newParam)} est_pickle_bytes={est_bytes}",
                    flush=True,
                )
                _write_new_member(args.path + "/running/" + str(args.agent_idx), newParam)
                
                print("GA - Part 2 - Agent - " + str(args.agent_idx) + " - Spwanning agents -  Pass 3", flush=True)

                # run the next task
                if args.parallel or args.slurm:
                    if valid_fitness_records > 0:
                        args.agent_counter += 1
                    if args.agent_counter < args.numOFgenerations:
                        run(args=copy.copy(args), self=False, params=params_config.copy())
                    else:
                        ea.finlized(args)
                    return

        # increase the agent counter for serial run only when valid fitness was recorded
        if has_any_valid_fitness:
            args.agent_counter += 1
        else:
            print(
                "GA - Part 2 - Serial run - No valid fitnessScore found; iteration counter unchanged.",
                flush=True,
            )
    
        print("GA - Part 2 - Agent - " + str(args.agent_idx) + " Done - Pass 4", flush=True)
        if args.agent_counter >= args.numOFgenerations:
            break

        if (args.parallel or args.slurm):
            break
        else:
            run(args=copy.copy(args), self=False, params=params_config.copy())
            None
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_of_agents", type=int, default=2)
    parser.add_argument("--all_in_first_run", type=int, default=1) # if 1, all ageents will test all the initial genes on the first run
    parser.add_argument("--total_test_with_same_param", type=int, default=1) # pass to the next process
    parser.add_argument("--total_testsGene_with_same_agent", type=int, default=5) # Hoe many genes to test in each agant
    parser.add_argument("--populationSize", type=int, default=5000)
    parser.add_argument("--numOFgenerations", type=int, default=2000)
    parser.add_argument("--geneMin", type=float, default=-1.0)        # not pass to the next process,  genes betwen geneMin and -geneMin or 0 to 0/1
    parser.add_argument("--gpu", type=str, default='False')        # pass to the next process
    parser.add_argument("--slurm", type=str, default='False')      # pass to the next process
    parser.add_argument("--parallel", type=str, default='False')      # pass to the next process

    parser.add_argument("--evolutionType", type=str, default='CondVAE') # GA, CMA_ES, SNES, etc. Comma-separated for multiple (e.g. GA,CMA_ES,SNES) - randomly picks one per agent
    
    # Required. These used to default to ../Para-HADES_Tasks/test_directory +
    # task_FindSeq, which meant a GA.py started without them ran the WRONG TASK
    # into a directory another experiment already owned, silently.
    parser.add_argument("--path", type=str, default=None)          # pass to the next process
    parser.add_argument("--paramFile", type=str, default=None)     # pass to the next process
    parser.add_argument("--taskFilename", type=str, default=None)  # pass to the next process
    parser.add_argument("--evolutionTarget", type=int, default=1)      # 1 = Max, -1 = Min

    # DO NOT CHANGE-----Internal var-------------
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--agent_idx", type=int, default=-1)            # modify by the process
    parser.add_argument("--agent_test_num", type=int, default=-1)       # receved by the child process
    parser.add_argument("--agent_counter", type=str, default='-1.0')        # modified by the next process
    # ------------------------------------------

    args = parser.parse_args()

    args.agent_idx = None if args.agent_idx == -1 else args.agent_idx
    args.gpu = True if args.gpu == 'True' or args.gpu == 'true' else False
    args.slurm = True if args.slurm == 'True' or args.slurm == 'true' else False
    args.parallel = True if args.parallel == 'True' or args.parallel == 'true' else False

    if (args.num_of_agents * args.total_testsGene_with_same_agent) > args.populationSize:
       args.populationSize = args.num_of_agents * args.total_testsGene_with_same_agent

    if args.path is None:
        parser.error("--path is required (e.g. --path ../Para-HADES_Tasks/dotTracing_run1). "
                     "It no longer defaults to a shared test directory.")
    if args.taskFilename is None and args.paramFile is None:
        parser.error("--taskFilename is required (e.g. --taskFilename task_dotTracing.py). "
                     "It no longer defaults to task_FindSeq.py.")
    if args.taskFilename is None:
        _root, _ = os.path.splitext(args.paramFile.replace('"', ''))
        args.taskFilename = _root + '.py'

    if args.taskFilename is not None:
        args.taskFilename = args.taskFilename.replace('"', '')

        if isinstance(args.paramFile, str) and args.paramFile.strip().lower() in ['', 'none', 'null']:
            args.paramFile = None

        if args.paramFile is None:
            task_root, _ = os.path.splitext(args.taskFilename)
            default_yaml = task_root + '.yaml'
            default_yml = task_root + '.yml'
            args.paramFile = default_yaml
            if (not os.path.exists(default_yaml)) and os.path.exists(default_yml):
                args.paramFile = default_yml


    if args.slurm: 
        args.parallel = False

    print(f"GA - Task: {args.taskFilename} | Params: {args.paramFile} | Path: {args.path}",
          flush=True)

    # OVERloading agent_counter
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])
    # args.method = ['jensenshannon'] # use for gene similarity only
    args.method = ['euclidean'] # use for gene similarity only

    # if args.total_test_with_same_param > 1 and args.total_testsGene_with_same_agent > 1:
    #     raise ValueError('total_test_with_same_param AND total_testsGene_with_same_agent can not be both greater than 1 = Program is not ready for both!')

    if args.parallel:
        parallel_process = '4'
        os.environ['OPENBLAS_NUM_THREADS'] = parallel_process
        os.environ['MKL_NUM_THREADS'] = parallel_process
        os.environ['NUMEXPR_NUM_THREADS'] = parallel_process
        os.environ['OMP_NUM_THREADS'] = parallel_process
        os.environ['VECLIB_MAXIMUM_THREADS'] = parallel_process

        # set threads to 1 to avoid issues with multiprocessing
        parallel_process = int(parallel_process)
        torch.set_num_threads(parallel_process)
        torch.set_num_interop_threads(parallel_process)
        torch.set_num_threads(parallel_process)


    # Install signal handlers only now — args is fully parsed and normalized, so
    # the handler can never fire with args=None. params_config is filled in by
    # main() (or by the handler's YAML fallback when SIGTERM arrives early).

    # for testing locally, comment out sigterm
    # _install_signal_handlers()

    print(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()
    main(args)
    print(f'---- End Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    print(f'---- Task Total Time -----T {(time.time() - start_time)} sec', flush=True)

