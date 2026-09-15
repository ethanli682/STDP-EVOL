import os

# to be ran on tufts hpc cluster with only conda env
# This function retuen a text with the slurm script
def SlurmScript(jobName, 
                jobTime = '00-23:55:55', 
                jobMemory='32768', 
                output_path='logs_', 
                jobCPUs=32, jobGPUs=0,
                excludeNodes = [],
                condaEnv = 'Para-HADES',
                partition = 'preempt',
                command = 'echo "Hello World!"',
                scriptName = 'script.sh',
                nextTask = '',
                extra_pythonpath = '',
                run_mode = 'conda',
                conda_sh = 'source /cluster/home/eli08/miniconda3/etc/profile.d/conda.sh',
                ):

      run_mode = (str(run_mode).strip().lower() or 'singularity')
      use_conda = (run_mode == 'conda')
      
      returnText = '#!/bin/bash\n'
      returnText += '#SBATCH --job-name=' + jobName + '\n'
      returnText += '#SBATCH --time=' + jobTime + '\n'
      returnText += '#SBATCH --mem=' + jobMemory + '\n'

      # make sure that the directory exists
      if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
      returnText += '#SBATCH --output=' + output_path + "/" + jobName + '.%j.out\n'
      # returnText += '#SBATCH --error=' + output_path + "/" + jobName + '.%j.err\n'
      # returnText += '#SBATCH --nodes=' + str(jobNodes) + "\n"
      # returnText += '#SBATCH --ntasks-per-node=' + str(jobTasks) + "\n"
      returnText += '#SBATCH --cpus-per-task=' + str(jobCPUs) + '\n'
      # returnText += '#SBATCH --threads-per-core=' + str(theadsPerCore) + '\n'
      
      if jobGPUs > 0:
         returnText += '#SBATCH --gres=gpu:' + str(jobGPUs) + '\n'
         returnText += '#SBATCH --partition=gpu\n'
      else:
         returnText += '#SBATCH --partition=' + partition + '\n'
      returnText += '#SBATCH --export=ALL \n'
      # returnText += '#SBATCH --constraint=rhel8 \n'    
           

      #exclude the following nodes
      if len(excludeNodes) > 0:
            returnText += '#SBATCH --exclude=' + excludeNodes + '\n'

      # use trap at https://hpc-discourse.usc.edu/t/signalling-a-job-before-time-limit-is-reached/314/3
      # to send a signal will be sent 30 seconds before the job is killed
      # NOTE: the this need to be the end of #SBATCH
      # returnText += '#SBATCH --signal=B:USR1@30\n'
      # returnText += '#SBATCH --signal=B:SIGINT@30\n'
      returnText += '#SBATCH --signal=B:SIGTERM@30\n'

      # Clean STOP switch: if a sentinel file exists in the experiment directory,
      # neither the preemption trap nor the normal-completion path resubmits the
      # successor, so the self-perpetuating chain can be halted without SIGKILL.
      # The sentinel path is derived from nextTask (<exp>/scripts/<script>.sh ->
      # <exp>/STOP); when nextTask is empty it stays empty and no guard is added,
      # preserving the previous behavior exactly.
      stop_sentinel = ''
      if nextTask:
            stop_sentinel = os.path.dirname(os.path.dirname(nextTask.rstrip('/'))) + '/STOP'

      # Kill-safe preemption trap. On SIGTERM (sent 30s before kill) we terminate
      # the container child and launch the successor, then exit promptly. The old
      # trap did `kill; sbatch; wait "${PID}"; handler` -- the `wait` after `kill`
      # blocked forever when the child was stuck in uninterruptible I/O on /cluster,
      # leaving the batch shell in SLURM state COMPLETING (zombie job), and `handler`
      # was undefined dead code. `exit 0` inside the trap ends the script, so the
      # normal-path resubmit below does NOT also run -- exactly one successor either
      # way (preempted OR completed normally), never two.
      if nextTask and stop_sentinel:
            returnText += ("trap 'echo \"Signal received!\"; kill \"${PID}\" 2>/dev/null; "
                           "if [ -f \"" + stop_sentinel + "\" ]; then "
                           "echo \"STOP sentinel present; not resubmitting successor.\"; "
                           "else sbatch " + nextTask + "; fi; exit 0' SIGTERM\n")
      elif nextTask:
            returnText += ("trap 'echo \"Signal received!\"; kill \"${PID}\" 2>/dev/null; "
                           "sbatch " + nextTask + "; exit 0' SIGTERM\n")
      else:
            returnText += "trap 'echo \"Signal received!\"; kill \"${PID}\" 2>/dev/null; exit 0' SIGTERM\n"
      returnText += '\n'

      #-------------------------------------------------
      
      # activate modules 
      # returnText += 'module load gcc/11.2.0 \n'
      # returnText += 'module unload gcc/7.3.0 \n'

      # for debugging and diagnostics logging 
      returnText += 'echo "-------------1---------------"\n'
      returnText += 'echo "Job is running on node(s): $SLURM_JOB_NODELIST "\n'
      returnText += 'echo "Job is running on host: $SLURMD_NODENAME "\n'
      returnText += 'echo "Job ID is: $SLURM_JOB_ID"\n'
      returnText += 'echo "Job name is: $SLURM_JOB_NAME"\n'
      returnText += 'echo "-------------2---------------"\n'
      returnText += 'if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not found (CPU node)"; fi\n'
      returnText += 'echo "-------------3---------------"\n'
      returnText += 'printenv CUDA_VISIBLE_DEVICES\n'
      returnText += 'echo "-------------4---------------"\n'
      returnText += '\n'
      
      # deactivate dispaly
      returnText += 'unset DISPLAY\n'
      returnText += 'export QT_QPA_PLATFORM="offscreen"\n'
      returnText += 'export MPLBACKEND="agg"\n'
      returnText += '\n'

      # # activate conda environment
      # returnText += 'echo "------Conda Env-----" \n'
      # returnText += 'source  /cluster/tufts/levinlab/hhazan01/miniconda3/etc/profile.d/conda.sh \n'
      # returnText += 'eval "$(conda shell.bash hook)" \n'
      # returnText += 'conda activate ' + condaEnv + '\n'
      # returnText += '\n'
      # returnText += 'echo "------Env Exp-----" \n'
      # returnText += 'export LD_LIBRARY_PATH=/cluster/tufts/levinlab/hhazan01/miniconda3/envs/delayW/lib:$LD_LIBRARY_PATH\n'
      # returnText += 'ldd /cluster/tufts/levinlab/hhazan01/miniconda3/envs/delayW/lib/python3.10/site-packages/numpy/_core/_multiarray_umath.cpython-310-x86_64-linux-gnu.so | grep libstdc++'
      # returnText += '\n'
      # returnText += 'echo "------Lib-----" \n'
      # returnText += 'echo $LD_LIBRARY_PATH  \n'
      # returnText += 'echo "-----------" \n'
      
      # job start message + timestamp
      returnText += 'echo "Job started!"\n'
      returnText += 'date\n'
      returnText += 'echo "-----------------------------"\n'

      # Debug SLURM variables before singularity
      returnText += 'echo "------SLURM Env Debug-----"\n'
      returnText += 'echo "SLURM_MEM_PER_CPU=$SLURM_MEM_PER_CPU"\n'
      returnText += 'echo "SLURM_MEM=$SLURM_MEM"\n'
      returnText += 'echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"\n'
      returnText += 'echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"\n'
      returnText += 'echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"\n'
      returnText += 'echo "SLURM_NTASKS=$SLURM_NTASKS"\n'
      returnText += 'echo "-----------------------------"\n'
      
      # Execution block. Ensure project root is always importable.
      project_root = os.path.dirname(os.path.abspath(__file__))
      nv_flag = '--nv ' if int(jobGPUs) > 0 else ''
      # extra_pythonpath (default '') lets a task inject additional import roots
      # (e.g. a decoupled scorer's repo + vendored cluster libs) without touching
      # the shared default for every other task.
      pythonpath_val = project_root if not extra_pythonpath else f'{project_root}:{extra_pythonpath}'

      # NEW-cluster conda-native path. Activate the env and run the command
      # directly (no container). SLURM_* vars are already in this shell's
      # environment (no --env passthrough needed). PYTHONPATH + thread caps
      # are set exactly as the singularity path set them, for parity.
      returnText += 'echo "------Conda Exec-----"\n'
      returnText += f'source {conda_sh}\n'
      returnText += f'conda activate {condaEnv}\n'
      returnText += f'export PYTHONPATH={pythonpath_val}\n'
      returnText += 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n'
      returnText += 'export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n'
      returnText += 'export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n'
      returnText += f'echo "Running command in conda env {condaEnv}: {command}"\n'
      returnText += command + ' &\n'

      returnText += 'PID="$!"\n'
      returnText += 'wait ${PID}\n'
      returnText += '\n'      
      
      # Check exit status
      returnText += 'EXIT_STATUS=$?\n'
      returnText += 'if [ $EXIT_STATUS -eq 0 ]; then\n'
      returnText += '    echo "Python script completed successfully"\n'
      
      # job end message + timestamp
      returnText += '    echo "Job ended!"\n'
      returnText += '    date\n'
      returnText += '    echo "-----------------------------"\n'

      # submit the successor job to slurm (unless the STOP sentinel is present).
      if nextTask and stop_sentinel:
            returnText += '    if [ -f "' + stop_sentinel + '" ]; then\n'
            returnText += '        echo "STOP sentinel present; not resubmitting successor."\n'
            returnText += '    else\n'
            returnText += '        sbatch ' + nextTask + '\n'
            returnText += '    fi\n'
      elif nextTask:
            returnText += '    sbatch ' + nextTask + '\n'
      # returnText += '    rm ' + nextTask + '\n'

      returnText += 'else\n'
      returnText += '    echo "Python script failed with exit status: $EXIT_STATUS"\n'
      returnText += 'fi\n'
      returnText += '\n'


      # # delete this script
      # returnText += 'rm ' + scriptName + '\n'
      # returnText += '\n'

      return returnText

# This function submit a job to the slurm
def SubmitJob(slurmScript, jobName):
      # create a file with the slurm script
      with open(jobName, 'w') as file:
            file.write(slurmScript)
      
      # submit the job to the slurm
      os.system('sbatch ' + jobName)





