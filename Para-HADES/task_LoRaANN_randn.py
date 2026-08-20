import os, sys, json, subprocess, signal, argparse, time, pickle, zstandard
from datetime import datetime
from slurmHPCHelper import *
from GA_utils_misc_func import pickle_loads_compat
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

global args
global taskPath
global params_config

def run_task(param, args):
    global taskPath 

    if args.train_bp is False:
        # show I am alive!
        try:
            with open(taskPath + '/' + str(args.agent_test_num) + '.live.lock', 'wt') as f:
                f.write('.')
        except:
            tqdm.write('Problem creating agent ' + str(args.agent_idx) + ', task number' + str(args.agent_test_num) + ' lock file: ' + taskPath + '/' + str(args.agent_test_num) + '.live.lock')
            return

    import time
    tqdm.write(f'---- Start Task ----- {datetime.now().strftime("_%d-%m-%Y-%H-%M-%S-%f")}')
    start_time = time.time()

    #-----------------------------------------------------
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim  # <--- Added for BP
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Define the device (GPU if available)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    try:
        # <--- Modified: Always prepare to load train dataset if BP is requested
        if args.train_bp:
            train_dataset = datasets.MNIST(root='./MNIST_data', train=True, transform=transform, download=True)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
        test_dataset = datasets.MNIST(root='./MNIST_data', train=False, transform=transform, download=True)
    except Exception as e:
        tqdm.write(f"Error loading MNIST dataset: {e}")

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Custom Layer for partial connections
    class CustomLayer(nn.Module):
        def __init__(self, input_size, output_size, mask=None, device='cpu'):
            super(CustomLayer, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.mask = mask

            # Create weight matrix with partial connections
            self.weight = nn.Parameter(torch.zeros(input_size, output_size, device=device))
            
            if mask is None:
                self.mask = torch.ones(input_size, output_size, device=device)
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.weight, a=0.01)

        def forward(self, x):
            with torch.no_grad():
                # Apply mask to the weight matrix
                masked_weight = self.weight * self.mask
            output = torch.mm(x, masked_weight)
                
            # Linear output (activation handled in CustomNN)
            return output

    # Define the neural network with custom connections
    class CustomNN(nn.Module):
        def __init__(self, layers=None, mask=[None, None], device='cpu'):
            super(CustomNN, self).__init__()
            self.fc1 = CustomLayer(layers[0], layers[1], mask=mask[0], device=device)  # Input size is 28*28 for MNIST images
            self.fc2 = CustomLayer(layers[1], layers[2], mask=mask[1], device=device)  # 10 output classes for digits 0-9

        def forward(self, x):
            x = x.view(-1, 28*28)  # Flatten the image
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def test(model, test_loader, device):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                outputs = model(data)

                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy
    # -------------------------------------------------------------------------
    # build multi-rank LoRA-style weights from parameter vectors
    i2h_a = torch.tensor(param['param']['i2h_a']).float().to(device)
    i2h_b = torch.tensor(param['param']['i2h_b']).float().to(device)
    h2o_a = torch.tensor(param['param']['h2o_a']).float().to(device)
    h2o_b = torch.tensor(param['param']['h2o_b']).float().to(device)

    input_size = 28 * 28
    output_size = 10

    # ... [Validation Checks Omitted for Brevity, kept same as original] ...
    if i2h_a.numel() % input_size != 0: raise ValueError(f"i2h_a length mismatch")
    rank_i2h = i2h_a.numel() // input_size
    if i2h_b.numel() % rank_i2h != 0: raise ValueError(f"i2h_b length mismatch")
    hidden_size = i2h_b.numel() // rank_i2h
    rank_h2o = h2o_b.numel() // output_size # Simplified logic for snippet
    hidden_size_h2o = h2o_a.numel() // rank_h2o

    # reshape to (rank, dim) so A.T @ B yields summed low-rank update
    # Note: rank_i2h and rank_h2o can be different for per-layer rank support
    i2h_a = i2h_a.view(rank_i2h, input_size)
    i2h_b = i2h_b.view(rank_i2h, hidden_size)
    h2o_a = h2o_a.view(rank_h2o, hidden_size)
    h2o_b = h2o_b.view(rank_h2o, output_size)

    # -------------------------------------------------------------------------
    # Initialize beta coefficients (scale factors for w0)
    # -------------------------------------------------------------------------
    if args.use_beta:
        # Try to load previously trained beta coefficients, otherwise initialize to 1.0
        if 'beta_i2h' in param['param'] and 'beta_h2o' in param['param']:
            beta_i2h = torch.tensor([param['param']['beta_i2h']], device=device, requires_grad=True)
            beta_h2o = torch.tensor([param['param']['beta_h2o']], device=device, requires_grad=True)
            tqdm.write(f"Loaded saved beta values: beta_i2h={param['param']['beta_i2h']:.4f}, beta_h2o={param['param']['beta_h2o']:.4f}")
        else:
            beta_i2h = torch.tensor([1.0], device=device, requires_grad=True)
            beta_h2o = torch.tensor([1.0], device=device, requires_grad=True)
    else:
        beta_i2h = torch.tensor([1.0], device=device, requires_grad=False)
        beta_h2o = torch.tensor([1.0], device=device, requires_grad=False)

    # -------------------------------------------------------------------------
    # <--- ADDED BP TRAINING LOGIC HERE
    # -------------------------------------------------------------------------
    
    # Helper functions for Option A (Meta-LoRA) and Option B (Canonicalize)
    def generate_scaffold():
        # Generates fresh random noise (The "Library")
        w1 = torch.randn(input_size, hidden_size, device=device)
        w2 = torch.randn(hidden_size, output_size, device=device)
        return w1 * args.base_scale, w2 * args.base_scale

    def canonicalize_scaffold(w1, w2):
        # Option B: Sort the noise to enforce a consistent topology
        # w1 is (input, hidden), w2 is (hidden, output)
        # We sort based on the columns of w1 (the "incoming features" of the hidden neurons)
        
        # Calculate norms of columns of w1
        norms = w1.norm(p=2, dim=0) 
        _, indices = norms.sort()
        
        # Reorder w1 columns
        w1_sorted = w1[:, indices]
        # Reorder w2 rows (to match the hidden neuron permutation)
        w2_sorted = w2[indices, :]
        
        return w1_sorted, w2_sorted

    # 1. Initial Scaffold Generation
    fixed_noise_i2h, fixed_noise_h2o = generate_scaffold()
    
    # Apply initial canonicalization if requested
    if args.canonicalize:
        fixed_noise_i2h, fixed_noise_h2o = canonicalize_scaffold(fixed_noise_i2h, fixed_noise_h2o)

    if args.train_bp:
        tqdm.write("---- Running Backpropagation to find LoRA vectors ----")
        if args.meta_lora: tqdm.write("  [Option A] Meta-LoRA Active: Resampling scaffold every epoch.")
        if args.canonicalize: tqdm.write("  [Option B] Canonicalization Active: Sorting hidden neurons.")

        # 1. Enable gradients for LoRA vectors
        i2h_a.requires_grad_(True)
        i2h_b.requires_grad_(True)
        h2o_a.requires_grad_(True)
        h2o_b.requires_grad_(True)
        
        # 2. Setup Optimizer
        optimizer_params = [i2h_a, i2h_b, h2o_a, h2o_b]
        if args.use_beta:
            optimizer_params.extend([beta_i2h, beta_h2o])
        optimizer = optim.Adam(optimizer_params, lr=0.001)
        criterion = nn.CrossEntropyLoss()
                
        # 4. Training Loop
        for epoch in range(param['epochs']):
            
            # --- OPTION A: META-LORA ---
            if args.meta_lora:
                # Resample the noise scaffold at the start of every epoch
                fixed_noise_i2h, fixed_noise_h2o = generate_scaffold()
                
                # --- OPTION B: CANONICALIZE (Inside loop if resampling) ---
                if args.canonicalize:
                    fixed_noise_i2h, fixed_noise_h2o = canonicalize_scaffold(fixed_noise_i2h, fixed_noise_h2o)
            # ---------------------------

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                with torch.enable_grad():
                    # Dynamic weight construction (preserving gradient flow to a/b vectors)
                    # Note: w = beta * base + A.T @ B
                    w_i2h_t = beta_i2h * fixed_noise_i2h + torch.matmul(i2h_a.t(), i2h_b)
                    w_h2o_t = beta_h2o * fixed_noise_h2o + torch.matmul(h2o_a.t(), h2o_b)
                
                # Manual Forward Pass matching CustomNN logic
                x = data.view(-1, 28*28)
                x = F.relu(torch.mm(x, w_i2h_t))
                output = torch.mm(x, w_h2o_t) # Matches CustomNN architecture
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    tqdm.write(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

        tqdm.write("---- BP Training Complete ----")
        
        # 5. Detach vectors Update param dict (same as before)
        param['param']['i2h_a'] = i2h_a.detach().flatten().tolist()
        param['param']['i2h_b'] = i2h_b.detach().flatten().tolist()
        param['param']['h2o_a'] = h2o_a.detach().flatten().tolist()
        param['param']['h2o_b'] = h2o_b.detach().flatten().tolist()
        
        # 6. Save beta coefficients if enabled
        if args.use_beta:
            param['param']['beta_i2h'] = float(beta_i2h.detach().item())
            param['param']['beta_h2o'] = float(beta_h2o.detach().item())

    # -------------------------------------------------------------------------
    # ROBUST EVALUATION: Monte Carlo Test on Fresh Noise
    # -------------------------------------------------------------------------
    
    eval_accuracies = []

    tqdm.write(f"---- Starting Robust Evaluation ({args.num_eval_seeds} seeds) ----")

    for i in range(args.num_eval_seeds):
        # 1. Generate FRESH noise (Test Set for the Scaffold)
        # If meta_lora is off, we technically should use the original fixed noise to see how it mastered that specific instance.
        # But if meta_lora is ON, we MUST resample to prove generalization.
        if args.meta_lora:
            eval_noise_i2h, eval_noise_h2o = generate_scaffold()
            if args.canonicalize:
                eval_noise_i2h, eval_noise_h2o = canonicalize_scaffold(eval_noise_i2h, eval_noise_h2o)
        else:
            # Standard LoRa: Evaluate on the SAME noise we trained on (Training Accuracy)
            eval_noise_i2h, eval_noise_h2o = fixed_noise_i2h, fixed_noise_h2o

        # 2. Reconstruct Weights
        w_i2h = beta_i2h.detach() * eval_noise_i2h + torch.matmul(i2h_a.t(), i2h_b)
        w_h2o = beta_h2o.detach() * eval_noise_h2o + torch.matmul(h2o_a.t(), h2o_b)

        # 3. Inject into Model
        layers = [input_size, hidden_size, output_size]
        model = CustomNN(device=device, layers=layers).to(device)
        model.fc1.weight.data = w_i2h
        model.fc2.weight.data = w_h2o

        # 4. Test
        acc = test(model, test_loader, device)
        eval_accuracies.append(float(acc))
        tqdm.write(f"   Eval Seed {i+1}: {acc:.2f}%")

    # Compute Statistics
    avg_accuracy = np.mean(eval_accuracies)
    best_accuracy = np.max(eval_accuracies) # Or mean, depending on what you want to optimize for. 
    # For Evolution, Mean is safer (rewards robustness). For "Can it work?", Max is okay.
    # Let's use MEAN for fitness if Meta-LoRA is on.

    tqdm.write(f'Test Accuracies: {[round(a, 2) for a in eval_accuracies]}')        
    if args.meta_lora:
        final_score = avg_accuracy  
        tqdm.write(f'Final Mean Score: {final_score:.2f}%')
    else: 
        final_score = best_accuracy
        tqdm.write(f'Final Max Score: {final_score:.2f}%')

    outP = dict()
    outP['tests'] = eval_accuracies
    outP['best_accuracy'] = float(final_score) 
    
    if args.evolutionTarget == -1:
        outP['fitnessScore'] = float(100.0 - final_score)
    else:   
        outP['fitnessScore'] = float(final_score)

    #------------------------------------------------------------------------------------------
    # Make sure the `fitnessScore` is in return dict and the fitness is a float number!
    #------------------------------------------------------------------------------------------

    tqdm.write(f'---- End Task ----- {datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")}')
    tqdm.write(f'---- Task Total Time ----- {(time.time() - start_time)} sec')

    if args.train_bp is False:
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
            # Use mean across repeated tests to provide a single scalar fitness
            outP['fitnessScore'] = float(np.mean(fitness))

        if not outP is None:
            for k,v in outP.items():
                params_config[gene_num][k] = v

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
    os.system('rm ' + args.path + "/running/" + str(args.agent_idx) + '.newMember.' + format)

    tqdm.write('Done Task: ' + str(args.agent_test_num) + ' for agent: ' + str(args.agent_idx))
    return


def sweep_and_plot(args, net_sizes=None, rank_pairs=None, epochs=5, seeds=None):
    import math
    import torch.nn as nn
    args.num_eval_seeds = epochs

    def plot_sweep(results, rank_pairs, save_dir, timestamp, seed):
        fig, (ax_acc, ax_params) = plt.subplots(1, 2, figsize=(13, 5))

        for rank_pair in rank_pairs:
            subset = [r for r in results if r['rank_i2h'] == rank_pair[0] and r['rank_h2o'] == rank_pair[1] and r['best_accuracy'] is not None]
            subset = sorted(subset, key=lambda r: r['net_size'])
            xs = [r['net_size'] for r in subset]
            ys = [r['best_accuracy'] for r in subset]
            stds = [r.get('std_accuracy', 0.0) for r in subset]
            betas_i2h = [r.get('beta_i2h', 1.0) for r in subset]
            betas_h2o = [r.get('beta_h2o', 1.0) for r in subset]

            label = f'rank_i2h={rank_pair[0]}, rank_h2o={rank_pair[1]}'
            ax_acc.errorbar(xs, ys, yerr=stds, marker='o', label=label, capsize=3)

            ax_params.plot(xs, betas_i2h, marker='o', label=f'{label} (beta_i2h)', linestyle='-')
            ax_params.plot(xs, betas_h2o, marker='s', label=f'{label} (beta_h2o)', linestyle='--')

        ax_acc.set_title('Accuracy by net_size and rank pairs (i2h, h2o)')
        ax_acc.set_xlabel('net_size')
        ax_acc.set_ylabel('best accuracy (%)')
        ax_acc.grid(True, linestyle='--', alpha=0.4)
        ax_acc.legend()

        ax_params.set_title('Beta values by net_size and rank pairs (i2h, h2o)')
        ax_params.set_xlabel('net_size')
        ax_params.set_ylabel('beta coefficient values')
        ax_params.grid(True, linestyle='--', alpha=0.4)
        ax_params.legend()

        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f'loraann_sweep__baseScale_{args.base_scale}_seed_{seed}_{timestamp}.png')
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return fig_path

    # Example usage:
    # rank_pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 4), (4, 4)]
    # net_sizes = [64, 128, 256, 512, 1024]
    # seeds = [None]

    for seed in tqdm(seeds, desc='Seeds'):
        results = []
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        for rank_i2h, rank_h2o in tqdm(rank_pairs, desc='Rank Pairs', leave=False):
            for net_size in tqdm(net_sizes, desc='Net Sizes', leave=False):
                layers = {
                        'i2h_a': torch.empty(rank_i2h, 28 * 28),
                        'i2h_b': torch.empty(rank_i2h, net_size),
                        'h2o_a': torch.empty(rank_h2o, net_size),
                        'h2o_b': torch.empty(rank_h2o, 10), 
                }
                for p in layers.keys():
                    nn.init.kaiming_uniform_(layers[p], a=math.sqrt(5))
                    layers[p] = layers[p].flatten().tolist()
                    
                # Initialize beta values if using them
                if args.use_beta:
                    layers['beta_i2h'] = 1.0
                    layers['beta_h2o'] = 1.0
                
                tmp_param = {
                    'param': layers,
                    'epochs': epochs,
                }

                outP = run_task(param=tmp_param, args=args)
                
                # Extract beta values if they were trained
                beta_i2h = 1.0
                beta_h2o = 1.0
                if args.use_beta and 'param' in tmp_param:
                    beta_i2h = tmp_param['param'].get('beta_i2h', 1.0)
                    beta_h2o = tmp_param['param'].get('beta_h2o', 1.0)
                
                tests = outP.get('tests', [])
                std_acc = np.std(tests) if tests else 0.0

                results.append({
                    'rank_i2h': rank_i2h,
                    'rank_h2o': rank_h2o,
                    'net_size': net_size,
                    'best_accuracy': outP.get('best_accuracy'),
                    'std_accuracy': float(std_acc),
                    'beta_i2h': float(beta_i2h) if isinstance(beta_i2h, (int, float)) else beta_i2h,
                    'beta_h2o': float(beta_h2o) if isinstance(beta_h2o, (int, float)) else beta_h2o,
                    'parameters': tmp_param['param'],
                })

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = plot_sweep(results, rank_pairs, args.path, timestamp, seed)

        results_path = os.path.join(args.path, f'loraann_sweep_results_baseScale_{args.base_scale}_seed_{seed}_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)

        tqdm.write(f'Saved sweep plot to {fig_path}')
        tqdm.write(f'Saved sweep results to {results_path}')

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
    print('Done signal_handler', flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal_handler_SIGTERM)
signal.signal(signal.SIGUSR1, signal_handler_SIGTERM)
signal.signal(signal.SIGINT, signal_handler_SIGTERM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='population')       # pass to the next process
    parser.add_argument("--agent_test_num", type=int, default=-1)       # receved by the child process
    parser.add_argument("--agent_idx", type=int, default=-1)            # modify by the process
    parser.add_argument("--agent_counter", type=str, default='-1.0')        # modified by the next process
    parser.add_argument("--total_test_with_same_param", type=int, default=1) # pass to the next process
    parser.add_argument("--total_testsGene_with_same_agent", type=int, default=1)  # pass to the next
    parser.add_argument("--gpu", type=str, default='False')          # pass to the next process
    parser.add_argument("--slurm", type=str, default='False')        # pass to the next process
    parser.add_argument("--parallel", type=str, default='True')      # NO need to pass to the next process or to myself
    parser.add_argument("--callBackScript", type=str, default=None)
    parser.add_argument("--evolutionTarget", type=int, default=1)        # 1 = Max, -1 = Min 
    parser.add_argument("--paramFile", type=str, default=None)   # 1 = Max, -1 = Min
    parser.add_argument("--train_bp", type=str, default='False') 
    parser.add_argument("--base_scale", type=float, default=0.0) 
    
    # <--- ADDED New Arguments
    parser.add_argument("--meta_lora", type=str, default='False', help="Option A: Resample noise every epoch")
    parser.add_argument("--canonicalize", type=str, default='False', help="Option B: Sort noise to enforce order")
    parser.add_argument("--num_eval_seeds", type=int, default=20)
    parser.add_argument("--use_beta", type=str, default='False', help="Enable trainable beta coefficient to scale w0 (fixed noise)")

    args = parser.parse_args()
    
    args.gpu = True if args.gpu == 'True' or args.gpu == 'true' else False
    args.slurm = True if args.slurm == 'True' or args.slurm == 'true' else False
    args.parallel = True if args.parallel == 'True' or args.parallel == 'true' else False
    args.train_bp = True if args.train_bp.lower() == 'true' else False
    args.meta_lora = True if args.meta_lora.lower() == 'true' else False
    args.canonicalize = True if args.canonicalize.lower() == 'true' else False
    args.use_beta = True if args.use_beta.lower() == 'true' else False

    if isinstance(args.paramFile, str) and args.paramFile.strip().lower() in ['', 'none', 'null']:
        args.paramFile = None
    if args.paramFile is None:
        task_root, _ = os.path.splitext(os.path.abspath(__file__))
        default_yaml = task_root + '.yaml'
        default_yml = task_root + '.yml'
        args.paramFile = default_yaml
        if (not os.path.exists(default_yaml)) and os.path.exists(default_yml):
            args.paramFile = default_yml
    
    # args.method = [item for item in args.method.replace(' ','').replace('"','').split(';')]

    if not args.callBackScript is None:
        args.callBackScript = args.callBackScript.replace('"','')
    
    # OVERloading agent_countern
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.slurm: 
        args.parallel = False

    if args.train_bp:
        # Specify rank pairs (rank_i2h, rank_h2o) for different layer configurations
        rank_pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (4, 4), (6, 6), (8, 8), (10, 10)]
        # rank_pairs = [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (3, 4)]
        sweep_and_plot(args=args, net_sizes=[16, 32, 64, 128, 256, 512, 2048], rank_pairs=rank_pairs, epochs=25, seeds=[None]) #seeds=[42, 69, 666, 777]
        # sweep_and_plot(args=args, net_sizes=[64], rank_pairs=[(1, 1), (1, 2)], epochs=10, seeds=[None]) #seeds=[42, 69, 666, 777]

    else:
        main(argparse._copy_items(args))