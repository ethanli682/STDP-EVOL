import os
import sys
import argparse
import time
import glob
from tqdm import tqdm

def find_files_without_phrase(directory, phrase):
   files_without_phrase = []
   current_time = time.time()
   # time_ago = current_time - 2 * 60 * 60  # Two hours in seconds
   # time_ago = current_time - 15 * 60  # 15 min in seconds
   time_ago = current_time - 2 * 60  # 2 min in seconds

   for root, _, files in os.walk(directory):
      for file in tqdm(files):
         if file.endswith('.out'):
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) < time_ago:
               with open(file_path, 'r') as f:
                  content = f.read()
                  if phrase not in content:
                     second_line = content.split('\n')[1].split('--')
                     job = {}
                     for parm in second_line:
                        if parm.count('path') > 0:
                           job["type"]= parm.split('_')[1]

                        if parm.count('agent_idx') > 0:
                           job["idx"]= parm.split(' ')[1]
                        
                        if parm.count('agent_counter') > 0:
                           job["counter"]= parm.split(' ')[1]

                     files_without_phrase.append([file_path, job])
   return files_without_phrase

def main():
   parser = argparse.ArgumentParser(description="Search for files without a specific phrase.")
   parser.add_argument('--directorys', type=str, help='Path to the directory to search')
   parser.add_argument('--output', type=str, help='Path to the output directory', default='LogsWithErrors')
   parser.add_argument('--phrase', type=str, help='The phrase to search for', default='Submitted batch job')
   args = parser.parse_args()

   logs_dir_pattren = "logs_*"

   dirs = args.directorys.split(',')
   pid = list()
   for dir in dirs:
      dir = os.path.abspath(dir)

      if not os.path.exists(dir):
         print(f"Could not find the directory {dir}")
         continue

      # search logs directories acording to the pattern
      logs_dirs = glob.glob(f"{dir}/{logs_dir_pattren}")

      if len(logs_dirs) == 0:
         print(f"Could not find any logs directories in {dir}")
         continue

      for logs_dir in logs_dirs:
         files_without_phrase = find_files_without_phrase(logs_dir, args.phrase)
         
         if len(files_without_phrase) > 0:
            print(f"Found {len(files_without_phrase)} files without the phrase '{args.phrase}' in {logs_dir}")

            # copy the files to another directory
            tmp_dir = logs_dir.split('/')[-1]
            if tmp_dir[-1] == '/':
               tmp_dir = tmp_dir[:-1]
            
            # another_dit = dir + "/" + tmp_dir + "_failed_jobs"
            another_dit = args.output + "/" + tmp_dir + "_failed_jobs"

            os.makedirs(another_dit, exist_ok=True)
            for file in files_without_phrase:
               os.system(f"cp {file[0]} {another_dit}")
               pid.append(file[1].split('.')[-2])
      
   if len(pid) > 0:
      node_info = list()
      for p in pid:
         node_info = os.popen("sacct --format=nodelist -j 12101767").read().replace(" ","").split("\n")[-2]
      
      # find how many unique nodes are in the list
      unique_nodes = list(set(node_info))
      with open(f"{args.output}/failed_jobs_nodes.txt", 'w') as f:
         for node in unique_nodes:
            f.write(f"{node}\n")      



if __name__ == "__main__":
   main()