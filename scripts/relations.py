from itertools import permutations
from collections import defaultdict
import utils
from statistics import median

import os
import numpy as np
import pandas as pd
import json


def load_relation_pairs(combination, order=True):
    
    all_pairs = set()
    
    path_dir = '../data/relations'
    if order == True:
        name = '-'.join(combination)
        paths = [f'{path_dir}/{name}.txt']
    else:
        perms = list(permutations(combination, len(combination)))
        paths = []
        for perm in perms:
            name = '-'.join(perm)
            paths.append(f'{path_dir}/{name}.txt')
    
    for path in paths:
        if os.path.isfile(path):
            with open(path) as infile:
                lines = infile.read().strip().split('\n')
            pairs = [(l.split(',')[0], l.split(',')[1]) for l in lines]
            all_pairs.update(pairs)
    return all_pairs






def load_relation_pairs_hyp(target_rel):
    rels_evidence = {'typical_of_property', 'variability_limited',  'afforded_usual', 'affording_activity'}
    rels_no_evidence = {'typical_of_concept', 'afforded_unusual', 'implied_category', 'variability_open'}
    target_rel = target_rel[0]
    # find combinations in which only one relation is associated with evidence
    # if no relation is associated with evidence, take the top non-evidence relation
    all_pairs = set()
    path_dir = '../data/relations'
    all_files = os.listdir(path_dir)
    for f in all_files:
        rels = f.split('.')[0].split('-')
        #print(rels)
        if target_rel in rels:
            rels_ev = [r for r in rels if r in rels_evidence]
            if len(rels_ev) == 1 and target_rel == rels_ev[0]:
                #load data
                with open(f'{path_dir}/{f}') as infile:
                    lines = infile.read().strip().split('\n')
                    all_pairs.update([tuple(l.split(',')) for l in lines])
            else:
                # only take if isolated or top:
                if rels[0] == target_rel:
                    with open(f'{path_dir}/{f}') as infile:
                        lines = infile.read().strip().split('\n')
                    all_pairs.update([tuple(l.split(',')) for l in lines])
                    
                
        
    return all_pairs


def str_to_tuple(pair_str):
    
    pair_str = pair_str.strip('(').strip(')').replace("'", '')
    pair_list = pair_str.split(', ')
    pair_t = tuple(pair_list)
    return pair_t


def load_scores(analysis_name, model_name, evidence_type):

    path_dir = f'../analysis/{model_name}/pairs/'
    path_file = f'{path_dir}/{analysis_name}.csv'
    df = pd.read_csv(path_file)

    pair_score_dict = dict()
    for i, row in df.iterrows():
        pair = row['pair']
        pair = str_to_tuple(pair)
        score = row[evidence_type] 
        pair_score_dict[pair] = score
    return pair_score_dict


def relation_overview(pair_score_dict, evidence_type, mode = 'strict'):

    relations =  ['pos', 'neg', 'all', 'some', 'few',
                         'evidence', 'no_evidence_pos', 'no_evidence_neg',
                         'implied_category', 
                         'typical_of_concept', 'typical_of_property', 
                         'affording_activity', 'afforded_usual', 'afforded_unusual',
                         'variability_limited', 'variability_open',
                         #'variability_limited_scalar', 'variability_open_scalar',
                         'rare', 'unusual', 'impossible', 'creative']

    table = []
    for rel in relations:
        if mode == 'strict':
            pairs = load_relation_pairs([rel], order=True)
        elif mode == 'hyp':
            pairs = load_relation_pairs_hyp([rel])
        scores = []
        for pair in pairs:
            if pair in pair_score_dict.keys():
                scores.append(pair_score_dict[pair])
        if len(scores) > 0:
            #med = median(scores)
            med = np.nanmedian(scores)
            #mean = np.nanmean(scores)
            #print(rel, med, mean, len(scores))
            d = dict()
            d['relation'] = rel
            d[evidence_type] = med
            d['n_pairs'] = len(scores)
            table.append(d)

    df = pd.DataFrame(table).set_index('relation')
    return df



def relations_to_file(properties):
    path_dir = '../data/relations'
    os.makedirs(path_dir, exist_ok=True)
    relation_pair_dict = defaultdict(set)
    for prop in properties:
        prop_dict = utils.load_prop_data(prop)
        rels_evidence = {'typical_of_property', 'variability_limited',
                         'afforded_usual', 'affording_activity', 
                         #'variability_limited_scalar', 'variability_open_scalar'
                        }
        rels_no_evidence = {'typical_of_concept', 'afforded_unusual', 'implied_category', 'variability_open'}

        for c, d in prop_dict.items():
            # sort by lables
            ml_label = d['ml_label']
            if ml_label in {'all', 'some', 'all-some', 'few-some'}:
                l = 'pos'
            elif ml_label in {'few'}:
                l = 'neg' 
            else:
                l = 'no-label'
            relation_pair_dict[(l,)].add((prop, c))

            # sort by subset of instances
            if ml_label in {'all', 'all-some'}:
                l_qu = 'all'
            elif ml_label in {'some', 'few-some'}:
                l_qu = 'some'
            elif ml_label in {'few'}:
                l_qu = 'few'
            else:
                l_qu = 'none'
            relation_pair_dict[(l_qu,)].add((prop, c))

            # sort relations:
            rel_dict = d['relations']
            p_rel_dict = defaultdict(list)
            for rel, p in rel_dict.items():
                if p > 0.5:
                    p_rel_dict[p].append(rel)
            sorted_p = sorted(list(p_rel_dict.keys()), reverse=True)
            rel_comb = []
            for p in sorted_p:
                rels = sorted(p_rel_dict[p])
                # join if the same score
                rels = '__'.join(rels)
                rel_comb.append(rels)
            rel_comb_t = tuple(rel_comb)
            relation_pair_dict[rel_comb_t].add((prop, c))

            rel_ev = [rel for rel in rel_comb if rel in rels_evidence]
            rel_no_ev  = [rel for rel in rel_comb if rel in rels_no_evidence]

            if len(rel_ev) > 0 and l=='pos':
                hyp = 'evidence'
            elif len(rel_no_ev) > 0 and l=='pos':
                hyp = 'no_evidence_pos'
            elif l == 'neg' and len(rel_ev) == 0 and len(rel_no_ev) ==  0:
                hyp = 'no_evidence_neg'
            else:
                hyp  = 'no-hyp'
            relation_pair_dict[(hyp, )].add((prop, c))
            
    print(len(relation_pair_dict[('evidence',)]) + len(relation_pair_dict[('no_evidence_pos',)]))

    # to file
    for rel, pairs in relation_pair_dict.items():

        name = '-'.join(rel)
        path = f'{path_dir}/{name}.txt'
        with open(path,  'w') as outfile:
            for prop, c in pairs:
                outfile.write(f'{prop},{c}\n')
                    
                    
if __name__ == '__main__':
                    
    properties = utils.get_properties()  
    # exclude female from this analysis
    properties = [prop for prop in properties  if prop != 'female']
    relations_to_file(properties)