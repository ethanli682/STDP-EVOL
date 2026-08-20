import matplotlib.pyplot as plt
import argparse
import numpy as np
import yaml
from tqdm import tqdm          
from scipy.spatial import distance
import glob, os
import subprocess


def plotty(file, titleName, data, cNum): 
   # Create Plot
   fig, ax1 = plt.subplots() 
   ax1.set_xlabel('Generation') 
   ax1.set_ylabel(titleName +' Distance')
   color = ['green', 'red', 'black','lightcoral', 'darkorange', 'olive', 'teal', 'violet', 
         'skyblue', 'lightgreen', 'lightcoral',]

   max_data = 0
   for i,d in enumerate(data):
      # Define Data
      d = np.array(d).squeeze()
      x = np.arange(0, d.shape[0])  
      ax1.scatter(x, d, label="Target " + str(i), s=0.5, c=color[i%cNum], alpha=0.3) 
      max_data = np.max([max_data, d.max()])
      
   # Show plot
   plt.savefig(file,dpi=300)
   plt.close()


def task(args):
  target = list()
  target.append([322.32,996.94,264.14,870.27,701.57,66.56,450.7,517.85,674.03,609.08])
  target.append([708.75,838.64,978.18,139.75,64.07,400.19,718.36,425.16,655.41,578.39])
#   target.append([417.07,725.47,210.25,20.68,987.44,869.78,543.58,493.91,280.08,570.66])
#   target.append([91.79,169.75,251.79,44.13,237.67,705.18,631.44,25.76,771.64,995.25])
#   target.append([393.07,922.62,863.66,190.81,923.32,215.29,95.89,918.8,243.76,84.4])
#   target.append([141.2,356.77,648.62,545.81,809.65,373.28,168.02,623.77,693.75,847.49])
#   target.append([704.08,253.62,669.37,88.24,910.41,360.67,916.08,879.48,539.6,27.44])
#   target.append([100.81,883.95,264.32,556.83,727.61,966.16,948.24,0.65,202.88,705.58])
#   target.append([615.48,98.39,402.52,159.85,159.19,386.55,455.2,90.56,999.56,827.5])
#   target.append([59.41,675.95,601.8,129.86,895.76,4.46,247.65,785.6,774.06,712.61])
#   target.append([522.78,222.07,95.62,533.02,183.55,107.89,221.06,91.13,931.32,610.84])
   
#   method = ['euclidean', 'minkowski', 'cityblock','seuclidean','sqeuclidean','cosine','correlation','hamming','jaccard','jensenshannon','chebyshev','canberra','braycurtis','mahalanobis','yule','matching','dice','kulsinski','rogerstanimoto','russellrao','sokalmichener','sokalsneath']
  method = ['euclidean', 'cosine','correlation', 'jensenshannon', 'sqeuclidean']
  buildInFields = ['mutationRate', 'numOfMutation', 'mutationMin', 'mutationMax']

  plotData = dict()
  logs = glob.glob(args.dirc + '/*.log.csv')
  for i,l in enumerate(tqdm(logs)):
     geneData = list()
     with open(l, 'rt') as f:
        for line in f:
           yamlData = yaml.load(line, Loader=yaml.FullLoader)
           v_list = list()
           for k,v in yamlData['param'].items():
            if buildInFields.count(k)==0:
               v_list.append(v)
           geneData.append(v_list)

     geneData = np.array(geneData, dtype=np.float32).squeeze()
     geneData = np.round(geneData, 2)
     
     for m in method:
         if not(m in plotData):
            plotData[m] = list()
         for t in target:
            plotData[m].append(distance.cdist(geneData, np.array(t).reshape(1,-1), m))

  for m in method:
    plotty(args.dirc + "/../" +args.file + '.' + str(args.num) + '.' + m + '.jpg', m, plotData[m], len(target))


def main(args):
   directorys = glob.glob(args.path + '/*')
   for d in directorys:
      if not os.path.isdir(d): 
         continue
      tqdm.write(f"Directory - {d}")
      file=d.split('/')[-1].split('.')[0]
      num = int(d.split('/')[-1].split('.')[1])
      run_process = "python3 " + __file__ + " --num " + str(num) + " --file " + file + " --dirc " + d
      subprocess.Popen(run_process, shell=True)  
      print(run_process)
      
      # # ==debug==
      # args.dirc = d
      # task(args)

   # task(args)
      
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--path", type=str, default='.')       # pass to the next process
   parser.add_argument("--num", type=int, default=0)       # pass to the next process
   parser.add_argument("--file", type=str, default='plot')       # pass to the next process
   # parser.add_argument("--dirc", type=str, default=None)       # pass to the next process
   parser.add_argument("--dirc", type=str, default='Vectors') 
   
   args = parser.parse_args()   
    
   # if not (args.num is None) and not (args.file is None) and not (args.dirc is None):
   task(args)
   # else:
   #  print("----------------------------------------------")
   #  print(args)
   #  print("----------------------------------------------")
   #  main(args)
    
