import os, sys, json, subprocess, signal, argparse, time, pickle, zstandard
from datetime import datetime
from slurmHPCHelper import *
from GA_utils_misc_func import pickle_loads_compat

global args
global taskPath
global params_config

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

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

    #-----------------------------------------------------
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Define the device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    try:
        # train_dataset = datasets.MNIST(root=args.path + '/../MNIST_data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./MNIST_data', train=False, transform=transform, download=True)
    except Exception as e:
        print(f"Error loading MNIST dataset: {e}", flush=True)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
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
            # self.bias = nn.Parameter(torch.zeros(output_size, device=device))

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
            # output += self.bias

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

    # load weights from the gene (auto-handle stored layout)
    w_i2h = torch.tensor(param['param']['i2h']).float().to(device)
    if w_i2h.dim() == 1:
        hidden = w_i2h.numel() // (28 * 28)
        w_i2h = w_i2h.view(28 * 28, hidden)
    elif w_i2h.shape[0] == 28 * 28:
        # already input x hidden
        pass
    elif w_i2h.shape[1] == 28 * 28:
        # stored as hidden x input (PyTorch Linear default); transpose to input x hidden
        w_i2h = w_i2h.t()
    else:
        raise ValueError(f"Unexpected i2h shape: {w_i2h.shape}")

    w_h2o = torch.tensor(param['param']['h2o']).float().to(device)
    if w_h2o.dim() == 1:
        hidden = w_i2h.shape[1]
        w_h2o = w_h2o.view(hidden, 10)
    elif w_h2o.shape[1] == 10:
        # already hidden x 10
        pass
    elif w_h2o.shape[0] == 10:
        # stored as 10 x hidden; transpose to hidden x 10
        w_h2o = w_h2o.t()
    else:
        raise ValueError(f"Unexpected h2o shape: {w_h2o.shape}")

    # Initialize the model, loss function, and optimizer
    model = CustomNN(device=device, layers=[w_i2h.shape[0], w_i2h.shape[1], w_h2o.shape[1]]).to(device)
    
    # load weights from the gene
    model.fc1.weight.data = w_i2h
    model.fc2.weight.data = w_h2o

    del w_i2h, w_h2o


    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Training the model
    # import torch.optim as optim
    # def train(model, train_loader, optimizer, criterion, device):
    #     model.train()
    #     correct = 0
    #     total = 0
    #     running_loss = 0.0

    #     for data, target in train_loader:
    #         data, target = data.to(device), target.to(device)

    #         # Forward pass
    #         outputs = model(data)
    #         loss = criterion(outputs, target)

    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # Track the accuracy
    #         _, predicted = torch.max(outputs, 1)
    #         total += target.size(0)
    #         correct += (predicted == target).sum().item()

    #         running_loss += loss.item()

    #     accuracy = 100 * correct / total
    #     avg_loss = running_loss / len(train_loader)

    #     return avg_loss, accuracy

    # Testing the model
    # def test(model, test_loader, criterion, device):
    def test(model, test_loader, device):
        model.eval()
        correct = 0
        total = 0
        # test_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                outputs = model(data)
                # loss = criterion(outputs, target)
                # test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        # avg_loss = test_loss / len(test_loader)
        # return avg_loss, accuracy
        return accuracy

    # Training loop
    # num_epochs = 5
    # for epoch in range(1, num_epochs + 1):
        # train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    # test_loss, test_accuracy = test(model, test_loader, criterion, device)
    test_accuracy = test(model, test_loader, device)

    print(f'Test Accuracy: {test_accuracy:.2f}%', flush=True)
        # print(f'Epoch {epoch}/{num_epochs}, '
            # f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
            # f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    
    outP = dict()
    outP['tests'] = float(test_accuracy)
    if args.evolutionTarget == -1:
        outP['fitnessScore'] = float(100.0 - test_accuracy)
    else:   
        outP['fitnessScore'] = float(test_accuracy) 
    
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
    
    # args.method = [item for item in args.method.replace(' ','').replace('"','').split(';')]

    if not args.callBackScript is None:
        args.callBackScript = args.callBackScript.replace('"','')
    
    # OVERloading agent_counter
    tmp = args.agent_counter.split('.')
    args.agent_counter = int(tmp[0])
    args.epoch = int(tmp[1])

    if args.slurm: 
        args.parallel = False

    main(argparse._copy_items(args))
