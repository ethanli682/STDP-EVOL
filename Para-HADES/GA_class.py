
import json, pickle, zstandard
from datetime import datetime
import numpy as np
import copy
import glob
import os
import time
from GA_utils_history_loader import *
from GA_utils_scalers import *
# from GA_utility_diversity_calc import *
from GA_utils_misc_func import *
from GA_utils_framed_records import FramedRecordReader


class difEvoInit():
    def __init__(self, vars_and_bounds, params_ML=None, args=None, geneMin=None):
        savePath=args.path; populationSize=args.populationSize
        agent=args.agent_idx; evolutionTarget=args.evolutionTarget; epoch=args.epoch
        self.geneMin = geneMin
        self.params_ML = copy.deepcopy(params_ML)

        # count in recursion the number of times 'type' and 'array' is in the vars_and_bounds
        def countTypeAndArray(vars_and_bounds):
            count = 0
            for k,v in vars_and_bounds.items():
                if isinstance(v, dict):
                    count += countTypeAndArray(v)
                elif k == 'type':
                    count += 1
                elif k == 'array' and vars_and_bounds[k] != None:
                    count -= 1
                    inter_count = 1
                    for v in vars_and_bounds[k]:
                        inter_count *= v
                    count += inter_count
            return count

        def buildParamSlices(vb):
            """
            Walk vars_and_bounds in insertion order and return per-param slice info.
            Each entry: {'name': str, 'coevo_group': str, 'start': int, 'end': int, 'dim': int}
            coevo_group defaults to '_default' unless explicitly set in the YAML via 'coevo_group'.
            All untagged params share a single '_default' group.
            """
            slices = []
            cursor = [0]

            def _walk(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        if 'type' in v or 'array' in v:
                            arr = v.get('array')
                            if arr is not None:
                                dim = 1
                                for s in arr:
                                    dim *= int(s)
                            else:
                                dim = 1
                            group = v.get('coevo_group', '_default')
                            slices.append({
                                'name': k,
                                'coevo_group': group,
                                'start': cursor[0],
                                'end': cursor[0] + dim,
                                'dim': dim,
                                'min': v.get('min', None),
                                'max': v.get('max', None),
                            })
                            cursor[0] += dim
                        else:
                            _walk(v)

            _walk(vb)
            return slices

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
                print("-> initialization!", flush=True)
                
                if self.params_ML is None:
                    self.params_ML = self.bulidin_params()

                self.ML_param_Length = countTypeAndArray(self.params_ML)
                
                self.geneLength = countTypeAndArray(vars_and_bounds)
                self.geneFormat = 'json' if (self.geneLength + self.ML_param_Length) < 50 else 'pickle'

                if epoch >= 0:
                    loader = ChunkedGeneHistoryLoader(
                        savePath=args.path,
                        agent_id=-1,
                        geneFormat=self.geneFormat,
                        required_keys_and_types=['fitnessScore', 'iteration', 'epoch'],
                        shuffle=True
                    )
                    genePopulation = loader.loadAllGeneHistory(
                        args, allEpochs=False
                    )
                    agent_counter = [p['iteration'] for p in genePopulation if 'iteration' in p]
                    if len(agent_counter) > 0:
                        agent_counter = max(agent_counter)
                    else:
                        agent_counter = 0

                    # if len(genePopulation) < self.populationSize:
                    #     tmp, _ = self.generateRandomGenePopulation(self.populationSize - len(genePopulation), self.geneMin)
                    #     for p in tmp:
                    #         p['epoch'] = epoch
                    #         p['fname'] = self.savePath + '/genes/chromosome_' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f") + str(int(np.random.rand() *10000000000000000000) )
                    #         p['iteration'] = agent_counter
                    #         p['fitnessScore'] = None
                    #         if self.geneFormat == 'json':
                    #             p['fname'] = p['fname'] + '.json'
                    #             with open( p['fname'], 'wt') as f:
                    #                 json.dump(p, f)
                    #         elif self.geneFormat == 'pickle':
                    #             p['fname'] = p['fname'] + '.pkl'
                    #             with zstandard.open( p['fname'], 'wb') as f:
                    #                 f.write(pickle.dumps(p))
            elif agent < 0:
                    print('Agent number is invalid')
                    return -1
            else:
                self.agent_idx = agent
                self.geneLength = countTypeAndArray(vars_and_bounds)
                if params_ML is None:
                    self.params_ML = self.bulidin_params()

                self.ML_param_Length = countTypeAndArray(self.params_ML)
                
                self.geneFormat = 'json' if (self.geneLength + self.ML_param_Length) < 50 else 'pickle'

        self.vars_and_bounds = copy.deepcopy(vars_and_bounds)

        # Build per-param slice list and merged co-evolution groups.
        # ES plugins read self.coevo_groups to run independent per-group updates.
        # Untagged params (no 'coevo_group' in YAML) all land in '_default'.
        self.param_slices = buildParamSlices(vars_and_bounds)

        # --- Preliminary group merge (min/max range per group name) ---
        coevo_groups_raw = {}
        for ps in self.param_slices:
            g = ps['coevo_group']
            if g not in coevo_groups_raw:
                coevo_groups_raw[g] = {
                    'start': ps['start'],
                    'end':   ps['end'],
                    'dim':   ps['dim'],
                    'params': [ps['name']],
                }
            else:
                coevo_groups_raw[g]['start'] = min(coevo_groups_raw[g]['start'], ps['start'])
                coevo_groups_raw[g]['end']   = max(coevo_groups_raw[g]['end'],   ps['end'])
                coevo_groups_raw[g]['dim']   = coevo_groups_raw[g]['end'] - coevo_groups_raw[g]['start']
                coevo_groups_raw[g]['params'].append(ps['name'])

        # --- Contiguity check: split any group whose range spans foreign slices ---
        # Catches the case where an untagged param sits between two tagged groups in
        # the YAML, which would make min/max incorrectly include foreign params.
        coevo_groups = {}
        for gname, ginfo in coevo_groups_raw.items():
            g_start, g_end = ginfo['start'], ginfo['end']
            foreign = [
                s for s in self.param_slices
                if g_start <= s['start'] and s['end'] <= g_end and s['coevo_group'] != gname
            ]
            if not foreign:
                coevo_groups[gname] = ginfo
            else:
                print(
                    f"[GA_class WARNING] co-evolution group '{gname}' is non-contiguous "
                    f"(foreign params {[s['name'] for s in foreign]} inside range "
                    f"[{g_start},{g_end}]). Auto-splitting into contiguous sub-groups.",
                    flush=True
                )
                members = sorted(
                    [s for s in self.param_slices if s['coevo_group'] == gname],
                    key=lambda s: s['start']
                )
                seg_idx    = 0
                seg_start  = members[0]['start']
                seg_end    = members[0]['end']
                seg_params = [members[0]['name']]
                for i in range(1, len(members)):
                    curr = members[i]
                    if curr['start'] == seg_end:       # contiguous with previous
                        seg_end = curr['end']
                        seg_params.append(curr['name'])
                    else:                               # gap — flush segment
                        coevo_groups[f'{gname}_{seg_idx}'] = {
                            'start': seg_start, 'end': seg_end,
                            'dim': seg_end - seg_start, 'params': list(seg_params),
                        }
                        seg_idx   += 1
                        seg_start  = curr['start']
                        seg_end    = curr['end']
                        seg_params = [curr['name']]
                coevo_groups[f'{gname}_{seg_idx}'] = {
                    'start': seg_start, 'end': seg_end,
                    'dim': seg_end - seg_start, 'params': list(seg_params),
                }

        self.coevo_groups = coevo_groups
        self.full_params_fn = None   # optional hook: (gene_record) → array; set by GA.py after difEvoInit

    def set_geneMin(self, geneMin):
        self.geneMin = geneMin

    def bulidin_params(self):
        inYaml = dict()
        # inYaml['ML'] = dict()
        # inYaml['ML']['loss'] = 0.0

        # !!!
        # # inYaml['ML']['mutationPoints'] = {
        # #     'min': 0.0,
        # #     'max': 1.0,
        # #     'type': 'float',
        # #     'name': 'mutationPoints',
        # #     # 'default': 1.0,
        # # }

        # # inYaml['ML']['mutationRate'] = {
        # #     'min': 0.0,
        # #     'max': 1.0,
        # #     'type': 'float',
        # #     'name': 'mutationRate',
        # #     # 'default': 1.0,
        # # }

        # inYaml['ML']['ANN_Hidden_size'] = { 
        #     'min': 10,
        #     'max': 500,
        #     'type': 'int',
        #     'name': 'ANN_Hidden_size',
        #     'default': 500,  #######################
        # }
        
        # inYaml['ML']['Diffusion_steps'] = { 
        #     'min': 10,
        #     'max': 5000,
        #     'type': 'int',
        #     'name': 'Diffusion_steps',
        #     'default': 1000,   #######################
        # }

        # inYaml['ML']['Diffusion_ANN_max_epoch'] = { 
        #     'min': 100,
        #     'max': 5000,
        #     'type': 'int',
        #     'name': 'Diffusion_ANN_max_epoch',
        #     'default': 600,   #######################
        # }


        # inYaml['ML']['Roulette_wheel_selection_pressure'] = {
        #     'min': 10.0, # 0.1,
        #     'max': 11.0, #30.0,
        #     'type': 'float',
        #     'name': 'Roulette_wheel_selection_pressure', #higher S mean more gready selection
        #     'default': 10.0,  #######################
        # }

        # # inYaml['ML']['Learning rate'] = {
        # #     'min': 1e-10,
        # #     'max': 1e-2,
        # #     'type': 'float',
        # #     'name': 'Learning rate',
        # #     # 'default': 1.0,
        # # }
        #!!!!
        # inYaml['ML']['HistoryWindow'] = 0.2,

        # inYaml['ML']['ANN_Hidden_size'] =  1500 
        
        # inYaml['ML']['Diffusion_steps'] =  500

        # inYaml['ML']['Diffusion_ANN_max_epoch'] =  300

        # inYaml['ML']['Roulette_wheel_selection_pressure'] = 10.0

        # inYaml['ML']['HistoryWindow'] =  0.5

        # inYaml['ML']['batch_size'] = 32

        # inYaml['ML']['Diffusion_lr'] = 0.003

        # inYaml['ML']['Diffusion_ANN_dropout_rate'] = 0.2

        # inYaml['ML']['learning_rate'] = 0.001

        # inYaml['ML']['output_size'] = 5  # Assuming a single fitness score output

        # print(self.params_ML)

        if self.params_ML is not None:# and isinstance(self.params_ML, dict):
            for k1 in self.params_ML.keys():
                inYaml[k1] = copy.deepcopy(self.params_ML[k1])

        return inYaml

    def generateRandomGene(self):
        gene = np.zeros(self.geneLength, dtype=np.float32)

        if self.geneMin == 0:
            gene_low, gene_high = 0.0, 1.0
        else:
            gene_low, gene_high = float(self.geneMin), float(-self.geneMin)

        for sl in self.param_slices:
            s, e, dim = sl['start'], sl['end'], sl['dim']
            p_min, p_max = sl.get('min'), sl.get('max')

            if p_min is not None and p_max is not None:
                # Normal distribution centred on each param's midpoint,
                # mapped back into gene space.  std chosen so ±3σ spans
                # the full param range (99.7 % inside bounds).
                p_min, p_max = float(p_min), float(p_max)
                if p_min > p_max:
                    p_min, p_max = p_max, p_min
                mid = (p_min + p_max) / 2.0
                std = (p_max - p_min) / 6.0  # 3-sigma rule

                # map param-space normal back to gene-space
                if self.geneMin != 0:
                    # gene ∈ [-1,1] → param = (g+1)*(max-min)/2 + min
                    # so g = 2*(param - min)/(max - min) - 1
                    g_mid = 2.0 * (mid - p_min) / (p_max - p_min) - 1.0
                    g_std = 2.0 * std / (p_max - p_min)
                else:
                    # gene ∈ [0,1] → param = min + (max - min)*g
                    # so g = (param - min) / (max - min)
                    g_mid = (mid - p_min) / (p_max - p_min)
                    g_std = std / (p_max - p_min)

                segment = np.random.normal(g_mid, g_std, size=dim).astype(np.float32)
                segment = np.clip(segment, gene_low, gene_high)
            else:
                # no min/max defined — fall back to uniform
                segment = np.random.uniform(gene_low, gene_high, size=dim).astype(np.float32)

            gene[s:e] = segment

        return gene.tolist()
    
    def generateRandomGenePopulation(self, populationSize, geneMin=None):
        value = list()
        for i in range(populationSize):
            value.append(dict())
            value[-1]['gene'] = self.generateRandomGene()
            value[-1]['fitnessScore'] = None
            value[-1]['iteration'] = 0

        return value, geneMin
        
    def saveFounder(self, args):
        # TODO If beter gene found:
        # 1. delete all the founders that are in the epoch
        # 2. run correlation on the genes and select the one that are not correlated and have high fitness score then 10% of the population

        while True:  
            print('P-1', flush=True)

            try:
                # data = self.loadGeneHistory(args)
                # I think that if loading the popolation and the founders there is no nneed to load the whole history.
                loader = ChunkedGeneHistoryLoader(
                    savePath=args.path,
                    agent_id=args.agent_idx,
                    geneFormat=self.geneFormat,
                    required_keys_and_types=['epoch', 'iteration', 'fitnessScore', 'loss',
                                                'Average pairwise distances', 'fname', 'Average pairwise distances'],
                    shuffle=True
                )
                data = loader.loadAllGeneHistory(
                    args, allEpochs=False
                )

                if len(data) == 0:
                    return []

                cleaned_data = []
                for d in data:
                    if (d['fitnessScore'] is not None) and (d['fitnessScore'] != []) and (d['fitnessScore'] != [None]) and (not isinstance(d['fitnessScore'],str)):
                        if np.isnan(np.array(d['fitnessScore'], dtype=np.float32)).sum() == 0 and np.isinf(np.array(d['fitnessScore'], dtype=np.float32)).sum() == 0:
                            cleaned_data.append(d)
                data = cleaned_data

                if len(data) == 0:
                    return []

                data.sort(key=lambda x: np.mean(x['fitnessScore']))

                bestScore = np.mean(data[0]['fitnessScore']) if args.evolutionTarget == -1 else np.mean(data[-1]['fitnessScore'])
                bestChromosome_fname = data[0]['fname'] if args.evolutionTarget == -1 else data[-1]['fname']
                bestChromosome = None
                if self.geneFormat == 'json':
                    with open(bestChromosome_fname, 'rt') as f:
                        for line in f:
                            candidate = json.loads(line)
                            if np.mean(candidate['fitnessScore']) == bestScore:
                                bestChromosome = candidate
                                break
                    if bestChromosome is None:
                        raise RuntimeError(f"Error in loading the best chromosome from json (fname={bestChromosome_fname}, bestScore={bestScore})")
                elif self.geneFormat == 'pickle':
                    csv_source = bestChromosome_fname
                    if FramedRecordReader.exists_for_csv(csv_source):
                        with FramedRecordReader.from_csv_path(csv_source, codec='pickle') as reader:
                            for idx in range(len(reader)):
                                try:
                                    candidate = reader.read(idx)
                                    if np.mean(candidate['fitnessScore']) == bestScore:
                                        bestChromosome = candidate
                                        break
                                except Exception:
                                    continue
                    else:
                        pkl_source = bestChromosome_fname
                        if pkl_source.endswith('.csv'):
                            pkl_source = pkl_source.replace('.csv', '.pkl')
                        with zstandard.open(pkl_source, 'rb') as f:
                            while True:
                                try:
                                    candidate = pickle.load(f)
                                    if np.mean(candidate['fitnessScore']) == bestScore:
                                        bestChromosome = candidate
                                        break
                                except EOFError:
                                    break
                    if bestChromosome is None:
                        raise RuntimeError(f"Error in loading the best chromosome from pickle (fname={bestChromosome_fname}, bestScore={bestScore})")
                bestChromosome['fitnessScore'] = float(str(bestChromosome['fitnessScore']))
                
                print('P0', flush=True)

                # Load all founders
                reports = glob.glob(self.savePath + '/founders/founder_*.json')
                score = list()
                skip_saving = True
                for r in reports:
                    with open(r, 'rt') as f:
                        dataPoint = json.load(f)
                    if self.geneFormat == 'pickle':
                        with zstandard.open(r.replace('json','pkl'), 'rb') as f:
                            dataPoint.update(pickle_loads_compat(f.read()))
                            
                    if int(dataPoint['epoch']) == int(args.epoch):
                        score.append([np.mean(dataPoint['fitnessScore']), r])
                        if (args.evolutionTarget == -1 and bestScore < np.mean(dataPoint['fitnessScore'])) or (args.evolutionTarget == 1 and bestScore > np.mean(dataPoint['fitnessScore'])):
                            skip_saving = False
                
                print('P1', flush=True)

                # Check if there is no other founder with the same score 
                if (bestChromosome != '') and (len(score)==0 or not skip_saving):            
                    fname = self.savePath + '/founders/founder_' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f") + str(int(np.random.rand() *10000000000000000000) ) 
                    if self.geneFormat == 'json':
                        with open(fname + '.temp', 'wt') as f:
                            json.dump(bestChromosome, f)
                        os.system("mv " + fname + '.temp ' + fname + '.json' )
                        score.append([bestScore, fname + '.json'])
                        if self.full_params_fn is not None:
                            try:
                                _fp = self.full_params_fn(bestChromosome)
                                _fp_tmp = fname + '_full_params.pt.temp.' + str(os.getpid())
                                import torch as _torch
                                _torch.save(_fp, _fp_tmp)
                                os.replace(_fp_tmp, fname + '_full_params.pt')
                                print(f"[Founder] Full params saved: {fname}_full_params.pt", flush=True)
                            except Exception as _e:
                                print(f"[Founder WARNING] full_params_fn failed: {_e}", flush=True)

                    elif self.geneFormat == 'pickle':
                        temp_data = dict(bestChromosome)
                        bestChromosome.pop('param')
                        bestChromosome.pop('gene')
                        
                        with open(fname + '.temp', 'wt') as f:
                            json.dump(bestChromosome, f)
                        os.system("mv " + fname + '.temp ' + fname + '.json' )
                        score.append([bestScore, fname + '.json'])

                        with zstandard.open(fname + '.temp', 'wb') as f:                            
                            f.write(pickle.dumps(temp_data))

                        os.system("mv " + fname + '.temp ' + fname + '.pkl' )

                        if self.full_params_fn is not None:
                            try:
                                _fp = self.full_params_fn(temp_data)
                                _fp_tmp = fname + '_full_params.pt.temp.' + str(os.getpid())
                                import torch as _torch
                                _torch.save(_fp, _fp_tmp)
                                os.replace(_fp_tmp, fname + '_full_params.pt')
                                print(f"[Founder] Full params saved: {fname}_full_params.pt", flush=True)
                            except Exception as _e:
                                print(f"[Founder WARNING] full_params_fn failed: {_e}", flush=True)

                        del temp_data
                    del bestChromosome

                print('P2', flush=True)

                if len(score)>1:
                    # Clean up
                    score.sort()
                    for i in range(1, len(score)):
                        if args.evolutionTarget == 1:
                            if score[i-1][0] >= score[i][0]:
                                os.system("rm " + score[i][1])
                                os.system("rm -f " + score[i][1].replace('.json', '_full_params.pt'))
                                if self.geneFormat == 'pickle':
                                    os.system("rm " + score[i][1].replace('json','pkl'))
                            else:
                                os.system("rm " + score[i-1][1])
                                os.system("rm -f " + score[i-1][1].replace('.json', '_full_params.pt'))
                                if self.geneFormat == 'pickle':
                                    os.system("rm " + score[i-1][1].replace('json','pkl'))
                        elif args.evolutionTarget == -1:
                            if score[i-1][0] >= score[i][0]:
                                os.system("rm " + score[i-1][1])
                                os.system("rm -f " + score[i-1][1].replace('.json', '_full_params.pt'))
                                if self.geneFormat == 'pickle':
                                    os.system("rm " + score[i-1][1].replace('json','pkl'))
                            else:
                                os.system("rm " + score[i][1])
                                os.system("rm -f " + score[i][1].replace('.json', '_full_params.pt'))
                                if self.geneFormat == 'pickle':
                                    os.system("rm " + score[i][1].replace('json','pkl'))
                print('P3', flush=True)
                return data
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"[saveFounder WARNING] Retrying after error: {e}", flush=True)
                time.sleep(np.random.rand())

    def geneTOparam(self, gene, param=None, param_list=None, param_type = 'genral'):
        # return the align the gene acording to the var boundaries  '
        # NOTE the main assumption is that gene already between the geneMin and -geneMin range! or 0/1 range

        if param_list is None:
            return

        if param is None:
            if param_type == 'ML':
                param = copy.deepcopy(self.params_ML)
            elif param_type == 'genral':
                param = copy.deepcopy(self.vars_and_bounds)

            global end
            global start
            end = 0
            start = 0

        # Function to convert from range [-1, 1] to [min_val, max_val]
        def convert_from_minus1_1(x, min_val, max_val):
            return (x + 1) * (max_val - min_val) / 2 + min_val
                      
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
                if ('array' in details) and details['array'] != None and not(details['array'] == 'T'):
                    if (("#dots" in details) and (details["#dots"]>0)):
                        array_size = 3 * details["#dots"]
                    else:
                        array_shape = np.array([p for p in details['array']])
                        for p in array_shape:
                            array_size *= p
                end += array_size

                if 'default' in details and details['default'] is not None:
                    param[k] = details['default']
                    
                    if param[k]=='None' or param[k]=='none' or param[k]=='NONE':
                        param[k] = None
                else:               
                    if details['type'] == 'float':
                        if 'min' in details and 'max' in details:
                            details['min'] = np.array(details['min'], dtype=np.float32)
                            details['max'] = np.array(details['max'], dtype=np.float32)
                            
                            if details['min'] > details['max']:
                                a = details['min']
                                details['min'] = details['max']
                                details['max'] = a
                            
                            if self.geneMin != 0:
                                new_gene = np.array(gene[start:end], dtype=np.float32)
                                # new_gene = np.tanh(new_gene)
                                # new_gene = np.clip(new_gene, -1, 1)
                                param[k] = convert_from_minus1_1(new_gene, details['min'], details['max'])
                            else:
                                new_gene = np.array(gene[start:end], dtype=np.float32)
                                # new_gene = sigmoid(new_gene)
                                # new_gene = np.clip(new_gene, 0, 1)
                                param[k] = (details['min'] + abs(details['min'] - details['max']) * new_gene)
                        else:
                            # if min and max are not present in the json file
                            param[k] = np.array(gene[start:end], dtype=np.float32) 

                        param[k] = np.array(param[k], dtype=np.float32).tolist()
                        param_list[details['name']] = param[k]
                    
                    elif details['type'] == 'int':
                        if 'min' in details and 'max' in details:
                            details['min'] = int(details['min'])
                            details['max'] = int(details['max'])
                            
                            if details['min'] > details['max']:
                                a = details['min']
                                details['min'] = details['max']
                                details['max'] = a
                            
                            if self.geneMin != 0:
                                new_gene = np.array(gene[start:end], dtype=np.float32)
                                # new_gene = np.tanh(newg_ene)
                                # new_gene = np.clip(new_gene, -1, 1)
                                param[k] = convert_from_minus1_1(new_gene, details['min'], details['max'])
                            else:
                                new_gene = np.array(gene[start:end], dtype=np.float32)
                                # new_gene = sigmoid(new_gene, dtype=np.float32))
                                # new_gene = np.clip(new_gene, 0, 1)
                                param[k] = (details['min'] + abs(details['min'] - details['max']) * new_gene)
                        else:
                            # if min and max are not present in the json file
                            param[k] = np.array(gene[start:end], dtype=np.float32) 

                        param[k] = np.array(param[k], dtype=int).tolist()
                        param_list[details['name']] = param[k]
                    
                    elif details['type'] == 'bool':
                        try:
                            param[k] = [float(x)>0.5 if self.geneMin==0 else float(x)>0 for x in gene[start:end]]
                            if len(gene[start:end]) ==1:
                                param[k] = param[k][0]
                        except:
                            param[k] = np.array(gene[start:end], dtype=np.float32) # NEED to be update !!!

                        param_list[details["name"]] = param[k]

                    # code deprecated == ralavent to BETSE
                    # elif details['type'] == 'image':
                    #     base_size = 1; array_size = 1
                    #     array_shape = np.array([p for p in details['array']])
                    #     for p in array_shape:
                    #         array_size *= p
                    #     base_shape = np.array([p for p in details['use_base'].shape])
                    #     for p in base_shape:
                    #         base_size *= p

                    #     if details['#dots'] == 0:
                    #         # if png 
                    #         param[k] = np.array(np.round(255 * np.array(gene[start:end], dtype=np.float32), 0), dtype=np.uint8)
                    #         if base_size == array_size:
                    #             param[k] = param[k].reshape(details['array'])
                    #         else:
                    #             tmp = (details['use_base']==255)==False
                    #             tmp[:,:,1:-1] = False
                    #             grad_img = param[k]
                    #             param[k] = np.copy(details['use_base'])
                    #             param[k][tmp] = grad_img
                    #     else:
                    #         # parse genes to create dots on the base shape in the red RGB
                    #         if base_size == array_size:
                    #             xSize = details['array'][0]
                    #             ySize = details['array'][1]
                    #             param[k] = np.zeros((xSize,ySize,4), dtype=np.uint8)
                    #             param[k][:,:,3] = 255
                    #             canvas = np.zeros((xSize,ySize), dtype=np.uint8)
                    #         else:
                    #             xSize = details['use_base'].shape[0]
                    #             ySize = details['use_base'].shape[1]
                    #             param[k] = np.copy(details['use_base'])
                    #             canvas = np.copy(details['use_base'][:,:,0])
                            
                    #         allPixels = np.array(list(range(xSize*ySize))).reshape(xSize,ySize)
                    #         mask = (details['use_base'][:, :, 0]==255)==False
                    #         availablePixels = allPixels[mask]

                    #         for d in range(0, details['#dots']*3, 3): # numOrAvilPixel,r,vmem
                    #             p,r,vMem = np.array(gene[start+d:start+d+3], dtype=np.float32)
                    #             p = int(round(p*(len(availablePixels)-1),0))
                    #             p = availablePixels[p]
                    #             gx,gy = np.where(allPixels==p)
                    #             r = 1 + int(round(r * (((xSize+ySize)/2)*0.05),0))
                    #             vMem = 1+int(round(vMem * 254,0))

                    #             y,x = np.ogrid[-gx:xSize-gx, -gy:ySize-gy]
                    #             mask = x*x + y*y <= r*r
                    #             canvas[mask] = vMem

                    #         param[k][:,:,0] = canvas

                    #     param[k] = {'image_content':param[k], 'image_path':details['name']}
                        
                    if 'round' in details and details['round'] != None:
                        param[k] = np.round(param[k], int(details['round']))
                        param_list[details['name']] = param[k]
                    
                    if 'array' in details and details['array'] != None:
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
        return param_list,param

    def paramTOgene(self, param, param_type = 'genral'): 
        gene = []
        if param_type == 'ML':
            org_param = copy.deepcopy(self.params_ML)
        elif param_type == 'genral':
            org_param = copy.deepcopy(self.vars_and_bounds)
        
        # Function to convert from range [min_val, max_val] to [-1, 1]
        def convert_to_minus1_1(y, min_val, max_val):
            return 2 * (y - min_val) / (max_val - min_val) - 1
    
        def convert_param(p,k):
            p = np.array(p, dtype=np.float32)

            if 'type' in k and k['type'] == 'float':
                if k["min"]==k["max"]:
                    return np.full(p.shape, self.geneMin, dtype=np.float32) #return minimum value for gene
                if self.geneMin != 0:
                    return np.array(convert_to_minus1_1(p, k['min'], k['max']), dtype=np.float32)
                else:
                    return (p - k['min']) / (k['max'] - k['min'])
                
            elif 'type' in k and k['type'] == 'int':
                if k["min"]==k["max"]:
                    return np.full(p.shape, self.geneMin, dtype=np.float32)
                if self.geneMin != 0:
                    return np.array(convert_to_minus1_1(p, k['min'], k['max']), dtype=np.float32)
                else:
                    return np.array((p - k['min']) / (k['max'] - k['min']),dtype=np.float32)
                
            elif 'type' in k and k['type'] == 'bool':
                if self.geneMin != 0:
                    p[p>0] = 1
                    p[p<=0] = -1
                    return p
                else:
                    p[p>0.5] = 1
                    p[p<=0.5] = 0
                    return p

            else:
                print('Error - paramTOgene = Unknown type!')
        
        def process_param(param, global_param):
            nonlocal gene
            if isinstance(global_param, dict):
                for k in global_param.keys():
                    if k in param and isinstance(param[k], dict):
                        process_param(param[k], global_param[k])                    
                    elif k in param:
                        if isinstance(param[k], str) and param[k].count(',')>0:
                                param[k] = list(param[k].split(','))
                        temp = np.array(param[k], dtype=np.float32).flatten()
                        gene.extend(convert_param(temp, global_param[k]))

        process_param(param, org_param)
        gene = [str(g) for g in gene]
        return gene
    
    def extractBuildInParamFromGene(self, parameters, build_in_params):
        # recursevly seperate between the parameters in the dictionary  that are built in and the parameters that are not
        if parameters is None or build_in_params is None:
            return None, None
        
        org_parameters = copy.deepcopy(parameters)

        if isinstance(org_parameters, dict):
            for k in org_parameters.keys():
                if k in build_in_params and k in parameters and isinstance(parameters[k], dict):
                    build_in_params[k], parameters[k] = self.extractBuildInParamFromGene(parameters[k], build_in_params[k])
                    parameters.pop(k)
                elif k in build_in_params:
                    build_in_params[k] = parameters[k]
                    parameters.pop(k)
        
        return build_in_params, parameters

    def finlized(self,args):
        None

