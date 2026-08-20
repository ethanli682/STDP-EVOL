import argparse
import numpy as np


def plotty(file, data_1, data_1_std, data_2):
   # Import Library
   import matplotlib.pyplot as plt 
   
   # Define Data
   x = np.arange(0, data_1.shape[0])
   
   # Create Plot
   fig, ax1 = plt.subplots() 
   
   ax1.set_xlabel('Semi-Generation') 
   ax1.set_ylabel('Deversity', color = 'red')
   ax1.set_ylim([0, 1])
   ax1.scatter(x, data_1, color = 'red', s=0.5, alpha=0.5) 
   ax1.scatter(x, data_1, color = 'violet', s=(data_1_std)*50, alpha=0.1) 

   ax1.tick_params(axis ='y', labelcolor = 'red') 
   
   # Adding Twin Axes
   ax2 = ax1.twinx() 
   ax2.set_ylim([0, max(1, data_2.max())])
   ax2.set_ylabel('Fitness', color = 'blue') 
   ax2.scatter(x, data_2, color = 'blue', s=0.5, alpha=0.1) 
   ax2.tick_params(axis ='y', labelcolor = 'blue') 
   
   # Show plot
   plt.savefig(file, dpi=300)
   plt.close()

def main(args):
   import glob, os
   directorys = glob.glob(args.path + '/*')
   for d in directorys:
      if not os.path.isdir(d): 
         continue

      file=d.split('/')[-1].split('.')[0]
      num = int(d.split('/')[-1].split('.')[1])

      data = np.loadtxt(d+'/diversity.csv', delimiter=",")
      plotty(args.path +'/' + file + '.' + str(num) +'.jpg', data[:,2], data[:,3], data[:,4])

      # twoDigit = None
      # oneDigit = None
      # minPoint = None
      # maxPoint = None
      # # find first 2 digit and 1 digit
      # for t in range(data.shape[0]):
      #    if (twoDigit is None) and data[t,2]<100:
      #       twoDigit = [num, t]
         
      #    if (oneDigit is None) and data[t,2]<10:
      #       oneDigit = [num, t]
         
      #    if minPoint is None:
      #       minPoint = [num, t, data[t,2]]
      #    elif data[t,2] < minPoint[2]:
      #       minPoint = [num, t, data[t,2]]
         
      #    if maxPoint is None:
      #       maxPoint = [num, t, data[t,2]]
      #    elif data[t,2] > maxPoint[2]:
      #       maxPoint = [num, t, data[t,2]]

      # if not (oneDigit is None):
      #    with open(args.path + '/' + file + '.' + str(num) + '_oneDigit.csv','at') as f:
      #       f.write(str(oneDigit).replace('[','').replace(']','') + '\n')
      
      # if not (twoDigit is None):
      #    with open(args.path + '/' + file + '.' + str(num) + '_twoDigits.csv','at') as f:
      #       f.write(str(twoDigit).replace('[','').replace(']','') + '\n')

      # if not (minPoint is None):
      #    with open(args.path + '/' + file + '.' + str(num) + '_minPoint.csv','at') as f:
      #       f.write(str(minPoint).replace('[','').replace(']','') + '\n')

      # if not (maxPoint is None):
      #    with open(args.path + '/' + file + '.' + str(num) + '_maxPoint.csv','at') as f:
      #       f.write(str(maxPoint).replace('[','').replace(']','') + '\n')

      
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--path", type=str, default='/mnt/d/Downloads/3/mean/')       # pass to the next process
   
   args = parser.parse_args()
   main(args)
