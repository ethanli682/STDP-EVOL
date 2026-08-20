import torch
import os, sys, json, subprocess, signal, argparse, pickle, zstandard
from datetime import datetime
from slurmHPCHelper import *
from GA_utils_misc_func import pickle_loads_compat


#--------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

import glob
import gc

# Import the unified plotting system
from GA_utils_plotting import UnifiedPlotter


global args
global taskPath
global params_config


import numpy as np

def run_task(param, args):
    global taskPath 

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

    global target
    target = list()
    
    target.append([0.417,725.47,210.25,20.68,987.44,869.78,543.58,493.91,280.08,570.66])
    target.append([91.79,169.75,251.79,44.13,237.67,705.18,631.44,25.76,771.64,995.25])
    target.append([393.07,922.62,863.66,190.81,923.32,215.29,95.89,918.8,243.76,84.4])
    target.append([141.2,356.77,648.62,545.81,809.65,373.28,168.02,623.77,693.75,847.49])
    target.append([704.08,253.62,669.37,88.24,910.41,360.67,916.08,879.48,539.6,27.44])
    # target.append([704.08,253.62,669.37,88.24,910.41,360.67,916.08,879.48,539.6,27.44])
    # target.append([100.81,883.95,264.32,556.83,727.61,966.16,948.24,0.65,202.88,705.58])
    # target.append([615.48,98.39,402.52,159.85,159.19,386.55,455.2,90.56,999.56,827.5])
    # target.append([59.41,675.95,601.8,129.86,895.76,4.46,247.65,785.6,774.06,712.61])
    # target.append([522.78,222.07,95.62,533.02,183.55,107.89,221.06,91.13,931.32,610.84])

    # target.append([1.332, 2.044, 0.399, 0.295, -0.571, -0.981, -0.623, 0.804, -0.338, 0.561, 0.627, 0.09, 0.565, 0.513, 0.163, 0.496, 0.922, 0.616, 0.584, 0.602, 0.961, -0.783, 0.558, 0.978, -0.746, -0.078, 0.352, 0.331, -0.736, -0.115])
    # target.append([-0.418, 1.795, -0.232, -0.906, -0.714, -0.762, -0.833, -0.703, -0.994, 0.096, -0.209, 0.426, 0.534, -0.775, 0.009, -0.891, 0.091, 0.776, 0.678, -0.43, -0.207, -0.182, -0.977, 0.816, 0.321, -0.649, -0.699, 0.467, 0.007, 0.581])
    # target.append([0.897, -1.847, -0.126, 0.405, 0.241, 0.572, 0.757, -0.333, -0.221, 0.129, -0.145, -0.585, 0.122, 0.74, 0.495, -0.075, -0.412, 0.986, 0.525, -0.947, 0.376, -0.615, -0.333, -0.836, -0.006, 0.923, 0.215, -0.651, 0.53, 0.782])

    # target.append([1.332, 2.044, 0.399, ])
    # target.append([-0.418, 1.795, -0.232, ])
    # target.append([0.897, -1.847, -0.126,])


    outP = dict()
    outP['train'] = [0,0,0,0,0,0,0,]
    outP['tests'] = list()
    
    # for tar in target:
    #     outP['tests'].append(0)
    #     i = 0
    #     for k in param['param'].keys():
    #         outP['tests'][-1] += abs(param['param'][k] - tar[i])
    #         i += 1
    # outP['fitnessScore'] = 1 - float(str(zero_is_the_best(min(np.array(outP['tests'])))))
    
    # source=list()
    # for k in param['param'].keys():
    #     source.append(param['param'][k])
    # source = np.array(source)
    # for tar in target:
    #     outP['tests'].append(float(str(np.abs(np.corrcoef(source,tar)[0,1]))))
 
    # outP['fitnessScore'] = 1 - float(str(min(np.array(outP['tests']))))

    from scipy.spatial import distance
    source=list()
    for k in param['param'].keys():
        print(param['param'][k], flush=True)
        v = param['param'][k]
        if isinstance(v, list):
            source.extend(v)
        else:
            source.append(v)
    source = np.array(source, dtype=float)
    source = np.round(source, 3)
    # method = args.method[0] if len(args.method) == 1 else args.method[1]     
    # method_type = ['euclidean', 'minkowski', 'cityblock','seuclidean','sqeuclidean','cosine','correlation','hamming','jaccard','jensenshannon','chebyshev','canberra','braycurtis','mahalanobis','yule','matching','dice','kulsinski','rogerstanimoto','russellrao','sokalmichener','sokalsneath']
    method = 'euclidean' #'sqeuclidean'

    for tar in target:
        tmp = distance.cdist(source.reshape(1,-1), np.array(tar).reshape(1,-1), method)[0][0]
        # tmp = np.tanh(tmp)
        # tmp = np.exp(-tmp)
        # outP['tests'].append(float(str(np.abs(tmp))))
        outP['tests'].append(float(str(tmp)))
        
    if args.evolutionTarget == -1:
        outP['fitnessScore'] = float(str((np.array(outP['tests']).min()))) 
    else:
        outP['fitnessScore'] = float(str(-1 * (np.array(outP['tests']).min()))) 
    
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
    try:
        plotter_config = {
            'metric_type': 'similarity',          # We're measuring similarity/distance
            'better_direction': 'lower',          # Lower distance is better
            'enable_clustering': True,            # Enable clustering visualization
            'enable_3d_viz': True,                # Enable 2D/3D gene pool plots
            'enable_individual_targets': True,    # Create individual target plots
            'plot_frequency': 0.5,                # 50% chance to plot (performance)
            'main_plot_title': 'Evolution: Similarity to Closest Target',
            'dpi': 300
        }
        
        plotter = UnifiedPlotter(args, targets=target, config=plotter_config)
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
            jobTime=params['main']['Task_resource']['time'] if self else params['main']['Main_resource']['time'], # 00:30:00, 01:00:00, 02:00:00, 04:00:00
            jobMemory=params['main']['Task_resource']['memory'] if self else params['main']['Main_resource']['memory'], # 32768, 65536, 131072, 262144
            output_path=args.path + (params['main']['Task_resource']['path'] if self else params['main']['Main_resource']['path']),
            jobCPUs=params['main']['Task_resource']['cpu'] if self else params['main']['Main_resource']['cpu'],
            # theadsPerCore=params['main']['Task_resource']['theadsPerCore'] if self else params['main']['Main_resource']['theadsPerCore'],
            jobGPUs=params['main']['Task_resource']['gpu'] if self else params['main']['Main_resource']['gpu'],
            excludeNodes=params['main']['Task_resource']['exclude_nodes'] if self else params['main']['Main_resource']['exclude_nodes'],
            condaEnv=params['main']['Task_resource']['conda_env'] if self else params['main']['Main_resource']['conda_env'],
            partition=params['main']['Task_resource']['partition'] if self else params['main']['Main_resource']['partition'],
            singularity_image=params['main']['Task_resource'].get('singularity_image', '/cluster/tufts/levinlab/hhazan01/singularity/delayW.sif') if self else params['main']['Main_resource'].get('singularity_image', '/cluster/tufts/levinlab/hhazan01/singularity/delayW.sif'),
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

    if args.slurm: 
        args.parallel = False

    main(argparse._copy_items(args))
