import argparse
import numpy as np
import glob, os
from tqdm import tqdm
import yaml
from datetime import datetime as dt


def load_logs(args, data=None):
    
    #check if all results are it, if not quit!
    files = [f for f in glob.glob(args.log_path + '/*.out')]

    if data is None:
        data = {}
        data['running time'] = list()
        data['CUDA error'] = list()
        data['unclear'] = list()
        data['killed and comback'] = list()
        data['killed in action'] = list()

    for file in tqdm(files):
        task_number = int(file.split('.')[1])
        machine = False
        gpu = False
        gpu_sn = False
        gpu_prp = False
        job_continue = False
        killed = False
        total_time = False

        with open(file, 'rt') as f:
            wholFileData = f.readlines()
            
        for line_counter,line in enumerate(wholFileData):
            if line_counter==1:
                machine = line.split('\n')[0]
            else:
                line = line.replace('\n','')

            if "Submitted batch job" in line:
                job_continue = True

            elif "DUE TO PREEMPTION" in line:
                killed = True
            
            elif "Task Total Time" in line:
                total_time = float(line.replace('---- Task Total Time ----- ','').replace(' sec',''))                                    
                            
            elif "CUDA error:" in line:
                if not gpu_sn in data['CUDA error']:
                    data['CUDA error'][gpu_sn]=dict()
                
                error = line.replace('CUDA error: ','')

                if not error in data['CUDA error'][gpu_sn]:
                    data['CUDA error'][gpu_sn][error]=list()
                
                data['CUDA error'][gpu_sn][error].append({
                    'Task Number': task_number, 
                    'GPU': gpu, 
                    'Machine': machine, 
                    'GPU number': gpu_prp,
                    'restart job': job_continue,
                })

        if killed and job_continue:
            data['killed and comback'].append({
                'Task Number': task_number, 
                'GPU': gpu, 
                'Machine': machine, 
                'GPU number': gpu_prp,
                'restart job': job_continue,
            })

        elif killed and not job_continue:
            data['killed in action'].append({
                'Task Number': task_number, 
                'GPU': gpu, 
                'Machine': machine, 
                'GPU number': gpu_prp,
                'restart job': job_continue,
            })

        elif not killed and not job_continue:
            data['unclear'].append({
                'Task Number': task_number, 
                'GPU': gpu, 
                'Machine': machine, 
                'GPU number': gpu_prp,
                'restart job': job_continue,
            })
        
        if total_time:
            data['running time'].append({
                'Task Number': task_number, 
                'GPU': gpu, 
                'Machine': machine, 
                'GPU sn': gpu_sn,
                'Running Time': total_time,
            })
                        
   #  temp_file = dt.now().strftime(args.log_path + "_logs_%d-%m-%Y-%H-%M-%S")+'.7z'
   #  os.system('7z a ' + temp_file + ' '+ args.log_path + '/')
    
   #  # delete log file
   #  for file in tqdm(files):
   #      os.remove(file)  

    for k in data.keys():
        # save yaml
        yaml_file = args.log_report_path + '/' + args.log_path + '_logs_' + k + '.yaml'
        with open(yaml_file, 'tw') as f:
            yaml.dump(data[k], f)
        
    return data



def parce_performance(args, data):
   savef = args.log_report_path + '/' + args.log_path + '_logs.csv'

   gpu = dict()
   for machin in data['running time']:
    for item in data['running time'][machin]:
      if not item['GPU sn'] in gpu:
         gpu[item['GPU sn']] = list()
      gpu[item['GPU sn']].append(item['Running Time'])


   # save csv
   with open(savef, 'tw') as f:
      for k,v in gpu.items():
         f.write(str(k))
         for value in v:
            f.write(',' + str(value))
         f.write('\n')


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="logs_main")
    parser.add_argument("--log_report_path", type=str, default="reports")
    
    args = parser.parse_args()

    # check if directory exists
    try:
        os.makedirs(args.log_report_path)
    except:
        None    
      
    
    data = load_logs(args)
        
    # parce_performance(args, data)

