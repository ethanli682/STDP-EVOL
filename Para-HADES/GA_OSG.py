import argparse
import numpy as np
import yaml
import os, sys
import glob
import time
from datetime import datetime
import pickle
import signal
import subprocess


global args

class difEvoInit():
    def __init__(self, vars_and_bounds, mutation=0.3, crossover=0.7, populationSize=10, savePath=None, evolutionTarget = 1, agent=None):
        self.vars_and_bounds = vars_and_bounds
        self.mutation = mutation
        self.crossover = crossover
        if populationSize < 2:
            print("Population size too small")
            return -1
        self.populationSize = populationSize
        self.evolutionTarget = evolutionTarget
        
        if savePath is None:
            print("Save file name is missing", flush=True)
            return -1
        else:
            self.savePath = savePath
            
            if agent is None:
                print("GA - initialization!", flush=True)
                self.geneLength = 0
                self.geneLength += str(vars_and_bounds.values()).count('type')
                genePopulation = self.generateRandomGenePopulation()
                for i, p in enumerate(genePopulation):
                    with open(self.savePath + '/founder.' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f_") + str(int(np.random.rand() *10000000000000000000) ) + '.yaml', 'wt') as f:
                        yaml.dump(p, f)            
            elif agent < 0:
                    print('Agent number is invalid')
                    return -1
            else:
                self.agent_idx = agent    
                
    def generateRandomGenePopulation(self):
        value = list()
        for i in range(self.populationSize):
            value.append(dict())
            value[-1]['gene'] = (np.random.rand(self.geneLength)).tolist()
            value[-1]['fitness'] = None
        return value
    
    def loadGenePopulation(self):
        files = glob.glob(self.savePath + '/founder.*.yaml')
        population = list()
        for file in files:
            try:
                with open(file, 'rt') as f:
                    gene=yaml.load(f, Loader=yaml.FullLoader)
                gene['fname'] = file
                population.append(gene)
            except:
                None
        return population
    
    def geneTOparam(self, gene, param=None, param_list=None):
        # return the align the gene acording to the var boundaries  
        if param_list is None:
            return

        if param is None:
            param = self.vars_and_bounds.copy()

            global end
            global start
            end = 0
            start = 0
                        
        iterable = param.keys() if isinstance(param, dict) else param if isinstance(param, list) else []
        array_shape = []
        array_size = 1
        for k in iterable:
            if k is None: continue
            if not isinstance(k, dict) and not 'type' in param[k]:
               self.geneTOparam(gene=gene, param=param[k], param_list=param_list)
            elif isinstance(k, dict) :
               self.geneTOparam(gene=gene, param=k, param_list=param_list)
            else:
                details = param[k]
                array_shape = [1]
                array_size = 1
                if ('array' in details) and not(details['array'] == 'T'):
                    if (("#dots" in details) and (details["#dots"]>0)):
                        array_size = 3 * details["#dots"]
                    else:
                        array_shape = np.array([p for p in details['array']])
                        for p in array_shape:
                            array_size *= p
                end += array_size

                if 'default' in details and details['default'] is not None:
                    param[k] = details['default']
                else:               
                    if details['type'] == 'float':
                        details['min'] = float(details['min'])
                        details['max'] = float(details['max'])
                        
                        if details['min'] > details['max']:
                            a = details['min']
                            details['min'] = details['max']
                            details['max'] = a
                        
                        param[k] = details['min'] + abs(details['min'] - details['max']) * gene[start:end]
                        param[k] = np.array(param[k], dtype=float).tolist()
                        param_list[details['name']] = param[k]
                    
                    elif details['type'] == 'int':
                        details['min'] = int(details['min'])
                        details['max'] = int(details['max'])
                        
                        if details['min'] > details['max']:
                            a = details['min']
                            details['min'] = details['max']
                            details['max'] = a
                        
                        param[k] = details['min'] + abs(details['min'] - details['max']) * gene[start:end]
                        param[k] = np.array(param[k], dtype=int).tolist()
                        param_list[details['name']] = param[k]
                    
                    elif details['type'] == 'bool':
                        param[k] = gene[start:end] > 0.5
                        param_list[details['name']] = param[k]

                    elif details['type'] == 'image':
                        base_size = 1; array_size = 1
                        array_shape = np.array([p for p in details['array']])
                        for p in array_shape:
                            array_size *= p
                        base_shape = np.array([p for p in details['use_base'].shape])
                        for p in base_shape:
                            base_size *= p

                        if details['#dots'] == 0:
                            # if png 
                            param[k] = np.array(np.round(255 * gene[start:end], 0), dtype=np.uint8)
                            if base_size == array_size:
                                param[k] = param[k].reshape(details['array'])
                            else:
                                tmp = (details['use_base']==255)==False
                                tmp[:,:,1:-1] = False
                                grad_img = param[k]
                                param[k] = np.copy(details['use_base'])
                                param[k][tmp] = grad_img
                        else:
                            # parse genes to create dots on the base shape in the red RGB
                            if base_size == array_size:
                                xSize = details['array'][0]
                                ySize = details['array'][1]
                                param[k] = np.zeros((xSize,ySize,4), dtype=np.uint8)
                                param[k][:,:,3] = 255
                                canvas = np.zeros((xSize,ySize), dtype=np.uint8)
                            else:
                                xSize = details['use_base'].shape[0]
                                ySize = details['use_base'].shape[1]
                                param[k] = np.copy(details['use_base'])
                                canvas = np.copy(details['use_base'][:,:,0])
                            
                            allPixels = np.array(list(range(xSize*ySize))).reshape(xSize,ySize)
                            mask = (details['use_base'][:, :, 0]==255)==False
                            availablePixels = allPixels[mask]

                            for d in range(0, details['#dots']*3, 3): # numOrAvilPixel,r,vmem
                                p,r,vMem = gene[start+d:start+d+3]
                                p = int(round(p*(len(availablePixels)-1),0))
                                p = availablePixels[p]
                                gx,gy = np.where(allPixels==p)
                                r = 1 + int(round(r * (((xSize+ySize)/2)*0.05),0))
                                vMem = 1+int(round(vMem * 254,0))

                                y,x = np.ogrid[-gx:xSize-gx, -gy:ySize-gy]
                                mask = x*x + y*y <= r*r
                                canvas[mask] = vMem

                            param[k][:,:,0] = canvas

                        param[k] = {'image_content':param[k], 'image_path':details['name']}
                        
                    if 'round' in details:
                        param[k] = np.round(param[k], int(details['round']))
                        param_list[details['name']] = param[k]
                    
                    if 'array' in details:
                        if not('file' in k):
                            param[k] = np.reshape(np.array(param[k]), array_shape).tolist()
                    else:
                        if (isinstance(param[k],np.ndarray) and (param[k].shape != ())):
                            typ =str(param[k].dtype)
                            if typ.count('float') > 0:
                                param[k] = float(param[k][0]) 
                                param_list[details['name']] = param[k]
                            elif typ.count('int') > 0:
                                param[k] = int(param[k][0])
                                param_list[details['name']] = param[k]
                            elif typ.count('bool') > 0:
                                param[k] = bool(param[k][0])
                                param_list[details['name']] = param[k]
                            else:
                                param[k] = param[k][0]
                        elif (isinstance(param[k],list) and (len(param[k]) > 0)):
                            param[k] = param[k][0]

                start = end
                
            if end>=len(gene):
                return param_list,param

    
    def createCandidateGene(self):
        while(1):
            genePopulation = self.loadGenePopulation()
            if len(genePopulation) >= self.populationSize:
                break
       
        def mergeMutation(candidats_idx, fitnessProbability):
            candidates=list()
            for c in np.random.choice(candidats_idx, 3, replace = False, p=fitnessProbability):
                candidates.append(np.copy(genePopulation[c]['gene']))
            flip = -1 if np.random.rand() > 0.5 else 1
            mutant_values = (candidates[0] + self.mutation * flip * (candidates[1] - candidates[2])) % 1
            
            cross_points = np.random.rand(geneLength) < self.crossover
            if not np.any(cross_points): # mutate at least one point
                cross_points[np.random.randint(0, geneLength)] = True

            temp_Index = np.random.choice(len(candidates), 1)[0]
            return np.where(cross_points, mutant_values,  np.copy(candidates[temp_Index]))
        
        def multiPointsMutation(gene, points=1):
            # choose number of points in the gene to mutate randomly
            gene_idx = np.array([i for i in range(len(gene))])
            points = np.random.choice(gene_idx, points, replace=False)
            for p in points:
                flip = -1 if np.random.rand() > 0.5 else 1
                # flipII = -1 if np.random.rand() > 0.5 else 1
                # gene[p] += gene[p] * flip * self.mutation + flipII * np.random.rand() * self.mutation
                gene[p] += flip * np.random.rand() * self.mutation
                gene[p] = (gene[p]) % 1
            return(gene)
        
        def crossOver(numOfCrossoverP, numOfParents, candidats_idx, fitnessProbability):
            # make offspring using crossOver on gene
            if numOfParents == 0:
                numOfParents = np.random.randint(1, self.populationSize-1)

            parents=list(np.random.choice(candidats_idx, numOfParents, replace=False, p=fitnessProbability))
            parents = np.array(parents)
            
            numOfCrossoverP = numOfCrossoverP if numOfCrossoverP >= (numOfParents-1) else (numOfParents-1)
            numOfCrossoverP = numOfCrossoverP if numOfCrossoverP < geneLength else geneLength - 1
            crossOverPoints = np.random.choice(geneLength, numOfCrossoverP, replace = False)
            crossOverPoints.sort()
            gene = np.copy(genePopulation[parents[0]]['gene'])
            fromP=0
            for p in range(len(crossOverPoints)):
                toP = crossOverPoints[p]
                if toP == fromP:
                    continue
                gene[fromP:toP] = np.copy(genePopulation[parents[np.random.randint(0, numOfParents)]]['gene'][fromP:toP])
                fromP = toP
            gene[fromP:] = np.copy(genePopulation[parents[np.random.randint(0, numOfParents)]]['gene'][fromP:])
            
            return(gene)
        #------------
        newGeneCandidat_idx = np.random.randint(0, len(genePopulation))
        candidats_idx = list()
        for p in range(len(genePopulation)):
            # if p != newGeneCandidat_idx:
            candidats_idx.append(p)
        
        fitnessProbability = list()
        for cnd in candidats_idx:
            fitnessProbability.append(genePopulation[cnd]['fitness'])
        fitnessProbability = np.array(fitnessProbability, dtype=float)
        
        print(f'fitness : {str(fitnessProbability)}' )

        if np.isnan(fitnessProbability).sum() > 0 or np.isnan(fitnessProbability.sum()) or fitnessProbability.sum() == 0:
            fitnessProbability = np.zeros(len(candidats_idx))
            fitnessProbability[:] = 1/len(candidats_idx)
        else:
            if self.evolutionTarget == 1:
                if (np.sign(fitnessProbability) < 0).sum() == fitnessProbability.shape[0]:
                    fitnessProbability *= -1
                fitnessProbability = (fitnessProbability - fitnessProbability.min()) 
                fitnessProbability += fitnessProbability.mean()
                fitnessProbability = fitnessProbability / fitnessProbability.sum()
            elif self.evolutionTarget == -1:
                fitnessProbability = (fitnessProbability - fitnessProbability.max()) 
                fitnessProbability += fitnessProbability.mean()
                fitnessProbability = fitnessProbability / fitnessProbability.sum()
            
            #fail safe
            if np.isnan(fitnessProbability).sum() > 0 or np.isnan(fitnessProbability.sum()) or fitnessProbability.sum() == 0:
                fitnessProbability = np.zeros(len(candidats_idx))
                fitnessProbability[:] = 1/len(candidats_idx)

        geneLength = len(genePopulation[newGeneCandidat_idx]['gene'])
        new_gene = np.copy(genePopulation[newGeneCandidat_idx]['gene'])
        
        run = np.random.choice(3, p=[0.25, 0.5, 0.25])
        numOfCrossoverP = 1
        numOfParents = 2
        mutationPoints = 1

        # check diversity
        genePool = np.ndarray((len(genePopulation), geneLength), dtype=float)
        for g in range(len(genePopulation)):
            genePool[g, :] = np.array(genePopulation[g]['gene'])
        
        from scipy.spatial import distance
        deversityMatrix = distance.cdist(genePool, genePool, 'cosine')
        normalizedeversityMatrix = 1 - np.abs(deversityMatrix)
        # normalizedeversityMatrix = (deversityMatrix / geneLength)
        # normalizedeversityMatrix = np.sort(normalizedeversityMatrix)[:,1:] 
        # if self.evolutionTarget == -1:
        #     normalizedeversityMatrix = 1 - normalizedeversityMatrix
             
        fitnessProbability = (fitnessProbability * args.fitness + normalizedeversityMatrix.mean(1) * args.deversity)
        fitnessProbability /= fitnessProbability.sum()

        deversity_list = list()
        for i in range(len(genePopulation)):
            for j in range(i+1, len(genePopulation)):
                deversity_list.append(deversityMatrix[i,j])       
        deversity = np.mean(deversity_list)

        if deversity < 0.15:
            # run = np.random.choice(2) + 1 
            run = 2
            # mutationPoints = int(geneLength // 1.5)
            print(f"genes are too similar")
        print(f'Euclidian mean : {str(deversity)}' )
                
        print(f'probability : {str(fitnessProbability)}' )
        print(f"Gene method use = {run}")
        
        if run==0:
            new_gene = crossOver(numOfCrossoverP=numOfCrossoverP, numOfParents=numOfParents, candidats_idx=candidats_idx, fitnessProbability=fitnessProbability)
            if np.random.rand() > 0.5:
                new_gene = multiPointsMutation(new_gene, points=mutationPoints)      
        elif run==1:
            new_gene = mergeMutation(candidats_idx=candidats_idx, fitnessProbability=fitnessProbability)
            if np.random.rand() > 0.5:
                new_gene = multiPointsMutation(new_gene, points=mutationPoints)      
        elif run==2:
            new_gene = multiPointsMutation(new_gene, points=mutationPoints)
       
        return new_gene
                    
    def updateIndividual(self, fitnessScore, agentOutput):
        # set the fitness score of candidate and retuen new set the best candidate
        geneToSave = dict()
        geneToSave['fitness'] = float(str(fitnessScore))
        geneToSave['gene'] = agentOutput['gene'] 
        geneToSave['param'] = agentOutput['param']

        # update best if needed 
        best_flag = False
        file = self.savePath + '/best_log.txt'
        while(1):
            best_flag = False
            while(1):
                population = self.loadGenePopulation()
                if len(population) >= self.populationSize:
                    break

            # find the distance to other founders 
            genePool = np.ndarray((len(population), len(population[0]['gene'])), dtype=float)
            for g in range(len(population)):
                genePool[g, :] = np.array(population[g]['gene'])

            from scipy.spatial import distance
            deversity_newgene = distance.cdist(genePool, np.array(geneToSave['gene']).reshape(1,-1), 'cosine')
            deversity_newgene = 1 - np.abs(deversity_newgene)

            # deversity_newgene = deversity_newgene / len(population[0]['gene'])
            # deversity_newgene = np.sort(deversity_newgene)
            # if self.evolutionTarget == -1:
            #     deversity_newgene = 1 - deversity_newgene
            
            deversity = distance.cdist(genePool, genePool, 'cosine')
            normalizedeversityMatrix = 1 - np.abs(deversity)
            # normalizedeversityMatrix = (deversity / len(population[0]['gene']))
            # normalizedeversityMatrix = np.sort(normalizedeversityMatrix)[:,1:]
            # if self.evolutionTarget == -1:
            #     normalizedeversityMatrix = 1 - normalizedeversityMatrix

            # geneToSave['deversity_correction'] = str(deversity_newgene)

            fitnessScoreORG = fitnessScore
            fitnessScore = fitnessScore * args.fitness + np.mean(deversity_newgene) * args.deversity

            # set the best candidate
            if self.evolutionTarget == 1:
                populationFitness = [-np.inf if population[i]['fitness'] is None else (float(population[i]['fitness'])*args.fitness + normalizedeversityMatrix[i].mean()*args.deversity) for i in range(len(population))]
                bestFitnessScore = np.nanmax(np.array(populationFitness, dtype=float))
            elif self.evolutionTarget == -1:
                populationFitness = [np.inf if population[i]['fitness'] is None else (float(population[i]['fitness'])*args.fitness + normalizedeversityMatrix[i].mean()*args.deversity) for i in range(len(population))]
                bestFitnessScore = np.nanmin(np.array(populationFitness, dtype=float))
            print(f'best: {bestFitnessScore}, current: {fitnessScore}')
                            
            if bestFitnessScore is not None and ((self.evolutionTarget == -1 and fitnessScore <= bestFitnessScore) or (self.evolutionTarget == 1 and fitnessScore >= bestFitnessScore)):
                # appand log file with the best score!
                # Expect that other processes would like to update this file simultansily                
                
                best_flag = True
                # check if lock exist
                if not os.path.exists(file + '.lock'):
                    # open file indvidual#.loc
                    try:
                        with open(file + '.lock', 'wt') as f:
                            f.write(str(self.agent_idx))

                        with open(file + '.lock', 'rt') as f:
                            txt = f.read()
                        
                        if int(txt) == self.agent_idx:    
                            # save individual gene and fitness score!
                            with open(file, 'at') as f:
                                f.write('fitness,' + str(fitnessScoreORG) + ',' + 'gene,' + str(geneToSave['param'])+ '\n')
                                
                            # delete file indvidual#.loc
                            os.system("rm " + file + '.lock')
                            break
                    except:
                        print(" agent " + str(self.agent_idx) + " Unable to lock file")
            
            else: # no changes needed
                break   
        
        # save founder
        from scipy.spatial import distance
        while(1):
            while(1):
                population = self.loadGenePopulation()
                if len(population) >= self.populationSize:
                    break

            # find the distance to other founders 
            genePool = np.ndarray((len(population), len(population[0]['gene'])), dtype=float)
            for g in range(len(population)):
                genePool[g, :] = np.array(population[g]['gene'])

            deversityMatrix = distance.cdist(genePool, genePool, 'cosine')
            normalizedeversityMatrix = 1 - np.abs(deversityMatrix)
            # normalizedeversityMatrix = (deversityMatrix / len(population[0]['gene']))
            # normalizedeversityMatrix = np.sort(normalizedeversityMatrix)[:,1:] 
            # if self.evolutionTarget == -1:
            #     normalizedeversityMatrix = 1 - normalizedeversityMatrix
            deversityMeanReturn = 1 - np.mean(normalizedeversityMatrix)

            print(f'agent {args.agent_idx} args.deversity {args.deversity} args.fitness {args.fitness}', flush=True)


            myIndexs = list()
            nonIndex = list()
            for i,p in enumerate(population):
                if p['fitness'] is None:
                    nonIndex.append([p['fname'], i])
                else:
                    p['fitness'] = float(p['fitness'])
                    if ((self.evolutionTarget == 1 and (p['fitness'] * args.fitness + normalizedeversityMatrix[i].mean() * args.deversity) < fitnessScore) or \
                       (self.evolutionTarget == -1 and (p['fitness'] * args.fitness + normalizedeversityMatrix[i].mean() * args.deversity) > fitnessScore)):
                        myIndexs.append([p['fname'], i])
            
            if len(myIndexs)==0 and len(nonIndex)==0: # no changes needed
                return None, None
    
            if len(nonIndex)>0:
                myIndexs = nonIndex

            if not best_flag and len(myIndexs)==0:
                return None, None                

            lock = False
            for file, idx in myIndexs:
                fname = self.savePath + '/founder.' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f_") + str(int(np.random.rand() *10000000000000000000) )
                ind_geneToSave = dict.copy(population[idx])
                for key, value in geneToSave.items():
                    ind_geneToSave[key] = value
                ind_geneToSave['fname'] = fname +'.yaml'
                
                # open file indvidual#.loc
                # delete the original founder
                r = os.system("rm " + file)
                    
                if r == 0: # if succseful approve the new founder
                    try:                                                  
                        # save individual gene and fitness score!
                        with open(fname + '.temp', 'wt') as f:
                            yaml.dump(ind_geneToSave, f)  
                        os.system("mv " + fname + '.temp ' + fname + '.yaml')
                        lock = True        
                        break
                    except:
                        None
                else: # if not deleition was unsuccses remove the temp founder
                    None
            if lock:
                break
        return deversityMeanReturn, fitnessScore   
        
    def finlized(self,myIndex):
        while(1):
            population = self.loadGenePopulation()
            if len(population) >= self.populationSize:
                break
        Fitness = None
        Gene = None
        for p in population:
            if Fitness is None:
                Fitness = p['fitness']
                Gene = p['gene']
            else:
                if self.evolutionTarget == 1 and Fitness < p['fitness']:
                    Fitness = p['fitness']
                    Gene = p['gene']
                elif self.evolutionTarget == 11 and Fitness > p['fitness']:
                    Fitness = p['fitness']
                    Gene = p['gene']
        
        if Fitness is not None:
            file = self.savePath + '/best_log.txt'
            
            bestGene = self.geneTOparam(gene=Gene)
            temp_param_list = dict()
            temp_param_list, bestGene = self.geneTOparam(bestGene.copy(), param_list=temp_param_list)            
            
            # check if lock exist
            while(1):
                if not os.path.exists(file + '.lock'):
                    # open file indvidual#.loc
                    try:
                        with open(file + '.lock', 'wt') as f:
                            f.write(str(self.agent_idx))
                        
                        time.sleep(1)
                        
                        with open(file + '.lock', 'rt') as f:
                            txt = f.read()
                        
                        if int(txt) == self.agent_idx: 
                            # save individual gene and fitness score!
                            with open(file, 'at') as f:
                                f.write('Final round agent ' + str(myIndex) + ' best fitness is ,' + str(Fitness) + ',' + 'gene index,' + str(bestGene)+ '\n')
                        
                            # delete file indvidual#.loc
                            os.system("rm " + file + '.lock')
                            break
                    except:
                        None
                time.sleep(1)
                    
            while(1):
                try:
                    # save individual gene and fitness score!
                    with open(file, 'at') as f:
                        f.write('Final round agent ' + str(myIndex) + ' best fitness is ,' + str(Fitness) + ',' + 'gene index,' + str(bestGene)+ '\n')
                    break
                except:
                    None
                time.sleep(1)
                
            # delete file indvidual#.loc
            os.system("rm " + file + '.lock')
        

def save(obj, file_name):
    with open(file=file_name, mode='wb')as f:
        pickle.dump(obj=obj, file=f, protocol=pickle.HIGHEST_PROTOCOL)
     
def load(file_name):
    with open(file=file_name, mode='rb') as f:
       obj = pickle.load(file=f)
    return obj    
      
def main(args):

    # TODO:
    ## part 1 - Run from console
    #   1. create OSG DAG scripts for every agent
    #   2. create founder group
    #   3. create the first config for each agent
    #   4. Execute agents 

    ## part 3 
    # 
    # - Run on OSG login node, DAG part 2
    #   1. uncompress the results 
    #   2. evaluate results and replace a founder if necessery
    #   3. check running counter and increment counter and run or exit depend on the running counter
    #
    # - Run on OSG node, DAG part 1
    #   1. trasfer the configuration deatials required for starting task
    #   2. execute task
    #   3. evaluate resoult
    #   4. compress the files trasfer back to the login node

    if args.part == 1: # init and spawn process
        # make automatic backup of the directory code in the running directory
        os.makedirs(args.path, exist_ok=True)
        filename_code = datetime.now().strftime("Code_%d-%m-%Y-%H-%M-%S-%f") + '.tar'
        os.system('tar --exclude=betse.sif -czvf ' + args.path + '/' + filename_code + '  *.*' )
        
        # init GA and create founder group
        with open('param.yaml', 'rt') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        
        evolutionTarget=-1 # 1 = Max, -1 = Min 
            
        ea = difEvoInit(
            vars_and_bounds=params, savePath=args.path, populationSize=args.populationSize, 
            mutation=args.mutation, crossover=args.crossover, evolutionTarget=evolutionTarget
            )
        
        save(obj=[
                    params, evolutionTarget, 
                    args.populationSize, args.mutation, args.crossover, args.numOFgenerations,
                ], file_name=args.path + '/ga_running_obj.pkl')
        
        # save current directory
        current_path = os.getcwd()

        # create OSG DAG scripts
        for agent in range(args.num_of_agents):
            os.makedirs(args.path + '/AgentData/' + str(agent) + '/joblogs/', exist_ok=True)

            # DAG file
            txt='JOB agent'+ str(agent) +' run.sub\n' +\
                'VARS agent'+ str(agent) +' jobnum="'+ str(agent) + '"\n' +\
                'SCRIPT POST agent'+ str(agent) +' ./run.sh '+ str(agent) + '\n' +\
                'RETRY agent'+ str(agent) +' 200\n'

            with open(args.path + '/AgentData/' + str(agent) + '/run.dag', 'wt') as f:
                f.write(txt)

            # SUB file
            #'transfer_output_remaps = "$(jobnum).output.pkl = ' + args.path + '/AgentData/' + str(agent) + '/$(jobnum).output.pkl" \n' +\
            txt='universe     = vanilla \n' + \
                'Requirements = SINGULARITY_CAN_USE_SIF == TRUE \n' +\
                'request_cpus = 1 \n' +\
                'request_memory = 1GB \n' +\
                'request_disk = 5GB \n' +\
                'executable = run.sh\n' +\
                'arguments = $(jobnum)\n' +\
                '\n' +\
                'transfer_input_files = ' +\
                                '../../../betse.sif,' +\
                                filename_code + ',' +\
                                'ga_running_obj.pkl,' +\
                                str(agent) + '.newMember.yaml, ' +\
                                str(agent) + '.screipt.sh, ' +\
                                'run.sh \n' +\
                'transfer_output_files =' +\
                            str(agent) + '.resoult.yaml,' +\
                            str(agent) + '.screipt.sh \n' +\
                '+SingularityImage = "./../../../betse.sif" \n' +\
                '\n' +\
                'error = joblogs/job.$(jobnum).$(Cluster).$(Process)error\n' +\
                'output = joblogs/job.$(jobnum).$(Cluster).$(Process).output\n' +\
                'log = joblogs/job.$(jobnum).$(Cluster).$(Process).log\n' +\
                '\n' +\
                'queue 1\n'
            
            with open(args.path + '/AgentData/' + str(agent) + '/run.sub', 'wt') as f:
                f.write(txt)

            # lunch task wrapper   
            run_process = [
                    'python3', 
                    args.taskFilename,
                    '--callBackScript', __file__,
                    '--path', '.',
                    '--agent_idx', str(agent),   
                    '--agent_counter', '0',
                    '--agent_test_num', '0',
                    '--total_tests_per_agent', str(args.total_tests_per_agent),
                    '--gpu', 'True' if args.gpu else 'False',
                    ]
            txt=''
            for t in run_process:
                txt += t + ' '
            with open(args.path + '/AgentData/' + str(agent) + '/' + str(agent) + '.screipt.sh', 'wt') as f:
                f.write(txt)

            # copy run.sh from parent directory to the agent directory
            os.system('cp run.sh ' + args.path + '/AgentData/' + str(agent) + '/')
            os.system('cp ' + args.path + '/ga_running_obj.pkl ' + args.path + '/AgentData/' + str(agent) + '/')
            os.system('cp ' + args.path + '/' + filename_code + ' ' + args.path + '/AgentData/' + str(agent) + '/')

            # create the first config for each agent
            gene = ea.createCandidateGene()
            param = {}
            temp_param_list = dict()
            temp_param_list, param['param'] = ea.geneTOparam(gene.copy(), param_list=temp_param_list)
            param['gene'] = gene.tolist()
            
            # save mutation
            with open(args.path + "/AgentData/" + str(agent) + '/' + str(agent) + '.newMember.yaml', 'wt') as f:
                yaml.dump(param, f)

            # execute agents 
            os.chdir(args.path + "/AgentData/" + str(agent) + '/')
            
            # OSG
            os.system('condor_submit_dag run.dag') # start OSG with part 2
            # Debug
            # run_process = './run.sh ' + str(agent)
            # subprocess.Popen(run_process, shell=True)    
        
            # revert to original path
            os.chdir(current_path)
                
    if args.part == 3: # summrize individual run           
        [
            params, evolutionTarget,
            args.populationSize, args.mutation, args.crossover, args.numOFgenerations,
        ] = load (file_name=args.path + '/ga_running_obj.pkl')
      
        # check if all repeted files complited. 
        reports = glob.glob("*.resoult.yaml")
        train = list()
        test = list()
        if args.total_tests_per_agent == len(reports):
            # summerize resoults
            for reportFile in reports:
                with open(reportFile, 'rt') as f:
                    param = yaml.load(f, Loader=yaml.FullLoader)
                
                if 'test' in param:
                    test.append(param['test'])

                if 'train' in param:
                    train.append(param['train'])
            
            # for reportFile in reports:    
            #     # clearn resoults
            #     os.system('rm '+ reportFile)

            # load the gene from save tasks
            with open(str(args.agent_idx) + '.newMember.yaml', 'rt') as f:
                param = yaml.load(f, Loader=yaml.FullLoader)
            
            param['train'] = train    
            param['test'] = test
            
            with open('../../' + str(args.agent_idx) + '.log.csv', 'at') as f:
                f.write(str(param) + '\n')
                
            # os.system("rm " + str(args.agent_idx) + '.newMember.yaml')
            
            ea = difEvoInit(
                vars_and_bounds=params, savePath='../../', populationSize=args.populationSize, 
                agent=args.agent_idx, mutation=args.mutation, crossover=args.crossover, evolutionTarget=evolutionTarget)
        
            ea.updateIndividual(fitnessScore=np.mean(test), agentOutput=param)
            fitnessScore=np.mean(test)
            deversityMean, returnfitnessScore = ea.updateIndividual(fitnessScore=fitnessScore, agentOutput=param)
            print(f'agent {args.agent_idx} counter {args.agent_counter} Fitness {np.mean(test)}', flush=True)
            print(f'agent {args.agent_idx} counter {args.agent_counter} Fitness {np.mean(test)}', flush=True)

            # Rerun next generation
            #save diversity
            # check if lock exist
            if not(deversityMean is None):
                # if deversityMean > 0.4:
                #     tmp = args.deversity
                #     args.deversity = args.fitness
                #     args.fitness = tmp
                                        
                n = 500
                if (args.agent_counter //n) %2 ==0:
                    args.deversity = (args.agent_counter %n) * 1/n
                    args.fitness = 1 - args.deversity
                else:
                    args.deversity = (n - (args.agent_counter %n)) * 1/n
                    args.fitness = 1 - args.deversity

                # args.deversity = 1 - deversityMean
                # args.fitness = deversityMean
                
                # args.deversity = deversityMean
                # args.fitness = 1 - deversityMean
                
                while(1):
                    if not os.path.exists(args.path + '/diversity.lock'):
                        # open file indvidual#.loc
                        try:
                            with open(args.path + '/diversity.lock', 'wt') as f:
                                f.write(str(args.agent_idx))
                                                        
                            with open(args.path + '/diversity.lock', 'rt') as f:
                                txt = f.read()
                            
                            if int(txt) == args.agent_idx: 
                                # save individual gene and fitness score!
                                with open(args.path + '/diversity.csv', 'at') as f:
                                    f.write(str(args.agent_counter) + 
                                    ',' + str(deversityMean) +
                                    ',' + str(fitnessScore) +
                                    # ',' + str(args.deversity) +
                                    # ',' + str(args.fitness) + 
                                    '\n')
                            
                                # delete file indvidual#.loc
                                os.system("rm " + args.path + "/diversity.lock")
                                break
                        except:
                            None
                    time.sleep(1)

            args.part = 2

            gene = ea.createCandidateGene()
            param = {}
            temp_param_list = dict()
            temp_param_list, param['param'] = ea.geneTOparam(gene.copy(), param_list=temp_param_list)
            param['gene'] = gene.tolist()
            
            # save mutation
            with open(str(args.agent_idx) + '.newMember.yaml', 'wt') as f:
                yaml.dump(param, f)

            if args.agent_counter == args.numOFgenerations:
                with open('GA.STOP', 'wt') as f:
                    f.write('.')
                sys.exit(0) # return 0 # stop
            else:
                # lunch task wrapper   
                run_process = [
                        'python3', 
                        args.taskFilename,
                        '--callBackScript', __file__,
                        '--path', '.',
                        '--agent_idx', str(args.agent_idx),   
                        '--agent_counter', str(args.agent_counter + 1),
                        '--agent_test_num', '0',
                        '--total_tests_per_agent', str(args.total_tests_per_agent),
                        '--gpu', 'True' if args.gpu else 'False',
                        ]
                txt=''
                for t in run_process:
                    txt += t + ' '
                with open(str(args.agent_idx) + '.screipt.sh', 'wt') as f:
                    f.write(txt)

                with open('GA.Contine', 'wt') as f:
                    f.write('.')

                sys.exit(1) # return 1 # continue

        sys.exit(0) # return 0 # stop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--path", type=str, default=datetime.now().strftime("Exc-_%d-%m-%Y-%H-%M-%S-%f"))       # pass to the next process
    parser.add_argument("--num_of_agents", type=int, default=3)
    parser.add_argument("--agent_idx", type=int, default=-1)            # modify by the process
    parser.add_argument("--agent_test_num", type=int, default=-1)       # receved by the child process
    parser.add_argument("--agent_counter", type=int, default=-1)        # modified by the next process
    parser.add_argument("--total_tests_per_agent", type=int, default=1) # pass to the next process
    parser.add_argument("--populationSize", type=int, default=5)
    parser.add_argument("--numOFgenerations", type=int, default=5)    
    parser.add_argument("--fitness", type=float, default=0.2)
    parser.add_argument("--deversity", type=float, default=0.8)

    parser.add_argument("--mutation", type=float, default=0.3)
    parser.add_argument("--crossover", type=float, default=0.7)
    parser.add_argument("--gpu", type=str, default='False')        # pass to the next process

    parser.add_argument("--taskFilename", type=str, default='task_Task_OSG.py')      # pass to the next process
    
    args = parser.parse_args()
    
    args.gpu = True if args.gpu == 'True' or args.gpu == 'true' else False

    # OSG
    main(args=args)

    # # Debug!
    # args.part=1
    # main(args)

    # args.part=3
    # for args.agent_idx in range(args.num_of_agents):
    #     main(args)

    # args.part=2
    # for args.agent_idx in range(args.num_of_agents):
    #     main(args)