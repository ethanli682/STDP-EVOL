import os, sys, json, subprocess, signal, argparse, pickle, zstandard
import traceback
from datetime import datetime
from slurmHPCHelper import *
from GA_utils_misc_func import pickle_loads_compat
import numpy as np

# main dot tracing task in this file
import dot_tracingtask

# Import the unified plotting system
from GA_utils_plotting import UnifiedPlotter
#---------------------


global args
global taskPath
global params_config

import numpy as np

def run_task(param, args):
    global taskPath 

    # [Standard setup code...]
    try:
        with open(taskPath + '/' + str(args.agent_test_num) + '.live.lock', 'wt') as f:
            f.write('.')
    except:
        return

    import time
    print(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    start_time = time.time()

    #------------------------Task Start--------------------------------------
    #calls the file with simulation
    try:
        outP = dot_tracingtask.run_dot_task(param, args)
    except Exception as exc:
        print(f"Task - candidate FAILED: {exc}", flush=True)
        traceback.print_exc()
        outP = {"fitnessScore": None, "error": str(exc)}
    
    #------------------------------------------------------------------------------------------

    
    print(f'---- End Task ----- {datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")}', flush=True)
    print(f'---- Task Total Time ----- {(time.time() - start_time)} sec', flush=True)

    os.system('rm ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')

    return outP    

def main(args):

    global taskPath
    global params_config 

    taskPath = args.path + "/running/" + str(args.agent_idx)
    os.makedirs(taskPath, exist_ok=True)
    
    # load the gene from save tasks
    try:
        with open(args.path + "/running/" + str(args.agent_idx) + '.newMember.json', 'rt') as f:
            params_config = json.load(f)
        format = 'json'
    except:
        with zstandard.open(args.path + "/running/" + str(args.agent_idx) + '.newMember.pkl', 'rb') as f:
            params_config = pickle_loads_compat(f.read())
        format = 'pkl'

    for gene_num in range(len(params_config)):
        fitness = []
        for test in range(args.total_test_with_same_param):
            # Tell the task which candidate this is. Without it every gene and
            # repeat in this process writes the same plot filenames into the same
            # folder, so only the last one survives -- see _outDirFor() in
            # dot_tracingtask.py.
            args.gene_num = gene_num
            args.test_rep = test
            #-----------------------
            outP = run_task(params_config[gene_num], args)
            #-----------------------
            if outP['fitnessScore'] is not None:
                fitness.append(outP['fitnessScore'])    

        if len(fitness) > 0:
            outP['fitnessScore'] = fitness

        if not outP is None:
            for k,v in outP.items():
                params_config[gene_num][k] = v

    # ----------- Unified Plotting System -----------
    # Create plots using the new unified plotting module
    # Note: targets are defined in run_task(), so we reconstruct them here
    try:
        # Reconstruct targets (same as in run_task)
        dim = 10  # Adjust if needed based on your param dimension
        random_state = np.random.get_state()
        np.random.seed(42)
        target_A = np.random.uniform(-500, 500, dim)
        np.random.set_state(random_state)
        x_space = np.linspace(0, 4*np.pi, dim)
        target_B = np.sin(x_space) * 100
        target_C = np.array([100 if i % 2 == 0 else -100 for i in range(dim)])
        target_D = np.linspace(-50, 50, dim)
        target_E = np.floor(np.linspace(0, 10, dim)) * 20
        targets = [target_A, target_B, target_C, target_D, target_E]
        
        plotter_config = {
            'metric_type': 'error',               # We're measuring error/distance
            'better_direction': 'lower',          # Lower distance is better
            'target_labels': ['Random Seed 42', 'Sine Hill', 'High Freq Spikes', 'Linear Slope', 'Stairs'],
            'enable_clustering': True,            # Enable clustering visualization
            'enable_3d_viz': False,               # Don't need 3D for this task
            'enable_individual_targets': False,   # Too many targets for individual plots
            'plot_frequency': 0.1,                # 10% chance to plot (performance)
            'main_plot_title': f'Convergence to {len(targets)} Multimodal Targets',
            'dpi': 150  # Lower DPI for faster generation with many targets
        }
        
        plotter = UnifiedPlotter(args, targets=targets, config=plotter_config)
        plotter.create_all_plots(method='euclidean')
    except Exception as e:
        print(f"Warning: Plotting failed with error: {e}", flush=True)
        print("Continuing without plots...", flush=True)
    # ----------------------------------------------

    # write resoults
    if format == 'json':
        with open(taskPath + '/' + str(args.agent_test_num)  + '.resoult.json', 'wt') as f:
            json.dump(params_config, f)
    else:
        with zstandard.open(taskPath + '/' + str(args.agent_test_num)  + '.resoult.pkl', 'wb') as f:
            f.write(pickle.dumps(params_config))

    if args.parallel or args.slurm:
        run(args, self=False, params=params_config[-1].copy())

    # delete the gene file
    os.system('rm ' + args.path + "/running/" + str(args.agent_idx) + '.newMember.json')

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
            jobTime=params['main']['Task_resource']['time'] if self else params['main']['Main_resource']['time'], 
            jobMemory=params['main']['Task_resource']['memory'] if self else params['main']['Main_resource']['memory'],
            output_path=args.path + (params['main']['Task_resource']['path'] if self else params['main']['Main_resource']['path']),
            jobCPUs=params['main']['Task_resource']['cpu'] if self else params['main']['Main_resource']['cpu'],
            jobGPUs=params['main']['Task_resource']['gpu'] if self else params['main']['Main_resource']['gpu'],
            excludeNodes=params['main']['Task_resource']['exclude_nodes'] if self else params['main']['Main_resource']['exclude_nodes'],
            condaEnv=params['main']['Task_resource']['conda_env'] if self else params['main']['Main_resource']['conda_env'],
            partition=params['main']['Task_resource']['partition'] if self else params['main']['Main_resource']['partition'],
            command="python " + str(run_process).replace('[','').replace(']','').replace('\', \'',' ').replace('\'',''),
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
    print('Done signal_handler', flush=True)
    sys.exit(0)

if __name__ == "__main__":
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

    if args.parallel:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    # if args.slurm:
    #     signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
    #     signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
    #     signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
    #     signal.signal(signal.SIGINT, signal_handler_SIGTERM)

    if not args.callBackScript is None:
        args.callBackScript = args.callBackScript.replace('"','')
        
    # OVERloading agent_counter
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.slurm: 
        args.parallel = False

    main(argparse._copy_items(args))