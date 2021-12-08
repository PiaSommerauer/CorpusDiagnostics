import pandas as pd
import csv
import os
import json
import numpy as np
from collections import defaultdict

def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties

def load_evidence_type_dict(prop, model_name):
    
    evidence_type_dict = dict()
    dir_corpus = f'../analysis/{model_name}/annotation-tfidf-top_3_3-raw-10000-categories'

    with open(f'{dir_corpus}/{prop}/annotation-updated-done.csv') as infile:
        data = list(csv.DictReader(infile))
    for d in data:
        t = d['evidence_type']
        c = d['context']
        evidence_type_dict[c] = t
    return evidence_type_dict

def raw_to_distance(df, score_names, reference_name = 'median', score =  'dist-percent', sum_scores = ['sum']): 
    
    df_dict = df.to_dict('index')
    df_dict_distance = dict()
    
    median_dict = df_dict[reference_name]
    #score_names = ['prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    
    for i, d in df_dict.items():
        d_distance = dict()
        for k, v in d.items():
            if k in score_names:
                median = median_dict[k]
                if v == 0.0:
                    v = np.nan
                    dist = np.nan
                    dist_p = np.nan
                else:
                    dist =  v -  median
                    # distnace in percent:
                    dist_p = dist/median
                if score == 'dist-raw':
                    d_distance[k] = dist
                elif score == 'dist-percent':
                    d_distance[k] = dist_p 
                elif score == 'raw':
                    d_distance[k] = v
            elif k == 'n_pairs':
                d_distance['n_pairs'] = v
        df_dict_distance[i] = d_distance
    df_dict_distance[reference_name+'-reference'] = median_dict
    #df_dist = pd.DataFrame(df_dict_distance).T
    
    # sum values: 
    summed_dict = dict()
    for p, d  in df_dict_distance.items():
        new_d = dict()
        new_d.update(d)
        
        for sum_score in sum_scores:
            if sum_score == 'sum':
                # only sum non_nan values 
                vals = [v for k, v in d.items() if not np.isnan(v) and k != 'n_pairs']
                if len(vals) > 0:
                    new_d[sum_score] = sum(vals)/len(vals)
                else:
                    new_d[sum_score] = np.nan
            elif sum_score == 'bin':
                #total = len(d)
                above_zero = len([s for k, s in d.items() if s > 0 and not np.isnan(s) and k != 'n_pairs'])
                total = len([s for s in d.values() if not np.isnan(s)])
                if total > 0:
                    new_d[sum_score] = above_zero/total
                else:
                    new_d[sum_score] = 0
        summed_dict[p] = new_d
        
    df_dist_sum = pd.DataFrame(summed_dict).T
    return df_dist_sum


def get_examples(model_name, prop, label):
    
    concepts_pos = set()
    # use 'all' category
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_all_pos = f'{dir_path}/{prop}/all/{label}/'
    
    for f in os.listdir(path_all_pos):
        concept = f.split('.')[0]
        if concept != '':
            concepts_pos.add(concept)
    return concepts_pos


def get_categories_concept(prop, concept, model_name):
    
    categories = set()
    
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_prop = f'{dir_path}/{prop}/'
    labels = ['pos', 'neg']
    
    for cat in os.listdir(path_prop):
        for label in labels:
            full_path = f'{path_prop}/{cat}/{label}/{concept}.csv'
            if os.path.isfile(full_path):
                categories.add(cat)
    return categories


def load_prop_data(prop):
    
    path = f'../data/aggregated_semantic_info_scalar/{prop}.json'
    with open(path) as infile:
        concept_dict = json.load(infile)
    return concept_dict



def get_overview(metrics, level, model_name, evidence_type):
    path_dir = f'../analysis/{model_name}/{level}'
    overview_dict = defaultdict(dict)
    if level == 'properties':
        target = 'property'
    elif level == 'relations' or level  == 'relations-hyp':
        target = 'relation'
    for metric in metrics:
        if target == 'relation':
            metric_f = metric+'_'+evidence_type
        else:
            metric_f = metric
        path = f'{path_dir}/{metric_f}.csv'
        with open(path) as infile:
            data = list(csv.DictReader(infile))
        for d in data:
            if target in d:
                prop = d[target]
            else:
                prop = d['']
            sc = d[evidence_type]
            if sc == '':
                sc = 0
            overview_dict[prop][metric] = float(sc)
            if 'n_pairs' in d:
                overview_dict[prop]['n_pairs'] = float(d['n_pairs'])
                
    df = pd.DataFrame(overview_dict)
    return df.T