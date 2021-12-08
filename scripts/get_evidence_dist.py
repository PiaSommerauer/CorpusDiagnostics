from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import csv
import os

import utils
import relations


def get_categories(prop, model_name):
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}'
    categories = set()
    for d in os.listdir(path_dir):
        if '.' not in d:
            categories.add(d)
    return categories


def get_context_f1_dict_categories(prop, model_name):
    
    context_scores_dict = defaultdict(list)
    context_f1_dict = dict()
    
    categories = get_categories(prop, model_name)
    
    aggregation_name = f'aggregated-tfidf-raw-10000-categories'
    path_dir_agg = f'../analysis/{model_name}/{aggregation_name}/{prop}'
    
    for cat in categories:
        path = f'{path_dir_agg}/{cat}.csv'
        with open(path) as infile:
            data = list(csv.DictReader(infile))
#         n_pos = int(data[0]['total_pos'])
#         n_neg = int(data[0]['total_neg'])
    
        for d in data:
            context = d['context']
            f1 = float(d['f1'])
            context_scores_dict[context].append(f1)
    
    for context, scores in context_scores_dict.items():
        mean = sum(scores)/len(scores)
        context_f1_dict[context] = mean
    return context_f1_dict
                


def get_evidence_dist(prop, model_name, evidence_type_dict, calc):
    
    evidence_dist_dict = dict()
    
    type_evidence_dict = defaultdict(list)
    
    for c, t in evidence_type_dict.items():
        type_evidence_dict[t].append(c)
        t_c = 'all'
        type_evidence_dict[t_c].append(c)
        if t in ['p', 'n', 'l']:
            t_c = 'prop-specific'
            type_evidence_dict[t_c].append(c)
        elif t in ['i', 'r', 'b']:
            t_c = 'non-specific'
            type_evidence_dict[t_c].append(c)
        if t  in ['p', 'n', 'l', 'i', 'r', 'b']:
            t_c = 'all-p'
            type_evidence_dict[t_c].append(c)
     
    context_f1_dict = get_context_f1_dict_categories(prop, model_name)
    
    for t, contexts in type_evidence_dict.items():
        f1_scores = [context_f1_dict[c] for c in contexts if c in context_f1_dict]
        if calc == 'mean':
            score = sum(f1_scores)/len(f1_scores)
        elif calc == 'max':
            score = max(f1_scores)
        evidence_dist_dict[t] = score
        
    return evidence_dist_dict

    
def get_evidence_dist_concept(prop, concept, label, model_name, 
                                evidence_type_dict, context_f1_dict, calc):
  
    ev_dist_concept = dict()
    # categories
    categories = utils.get_categories_concept(prop, concept, model_name)
     
    # collect all relevant context of the concept
    contexts = set()
    for category in categories:
        dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
        full_path = f'{dir_path}/{prop}/{category}/{label}/{concept}.csv'

        with open(full_path) as infile:
            data = list(csv.DictReader(infile))
        for d in data:
            context = d['']
            diff = float(d['diff'])
            if context in evidence_type_dict and diff > 0:
                contexts.add(context)
                
    # collect scores:
    et_scores_dict = defaultdict(list)
    for c in contexts:
        et  = evidence_type_dict[c]
        f1 = context_f1_dict[c]
        if et in ['p', 'n', 'l']:
            t_c = 'prop-specific'
            et_scores_dict[t_c].append(f1) 
        elif et in ['i', 'r', 'b']:
            t_c = 'non-specific'
            et_scores_dict[t_c].append(f1)
        et_scores_dict[et].append(f1)
        
    # calculate score:
    for et, scores in et_scores_dict.items():
        if calc == 'mean':
            res = sum(scores)/len(scores)
        elif calc == 'max':
            res = max(scores)
        ev_dist_concept[et] = res
    return ev_dist_concept  


def get_evidence_dist_properties(model_name, calc):
    
    table = []
    
    properties = utils.get_properties()

    for prop in properties:
        evidence_type_dict = utils.load_evidence_type_dict(prop, model_name)
        evidence_dist = get_evidence_dist(prop, model_name, evidence_type_dict, calc)
        evidence_dist['property'] = prop
        table.append(evidence_dist)
    
    columns = ['all', 'all-p', 'prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    df = pd.DataFrame(table).set_index('property')[columns]
    # set nana to 0 before median
    #df = df.fillna(0.0)
    median = df.median(axis=0)
    df.loc['median'] = median
    
    return df



def get_evidence_dist_concepts(model_name, properties, calc):
    
    table = []
    keys = set()
    
    for prop in properties:
        concept_label_dict = dict()
        concepts_pos = utils.get_examples(model_name, prop, 'pos')
        concepts_neg = utils.get_examples(model_name, prop, 'neg')
        for c in concepts_pos:
            concept_label_dict[c] = 'pos'
        for c in concepts_neg:
            concept_label_dict[c] = 'neg'
        evidence_type_dict = utils.load_evidence_type_dict(prop, model_name)
        context_f1_dict = get_context_f1_dict_categories(prop, model_name)

        for concept, label in concept_label_dict.items():
            ev_dist_concept =  get_evidence_dist_concept(prop, concept, label, model_name, 
                                evidence_type_dict, context_f1_dict, calc)
            ev_dist_concept['label'] = label
            keys.update(ev_dist_concept.keys())
            ev_dist_concept['pair'] = (prop, concept)
            table.append(ev_dist_concept)
        print('finished prop', prop)
        
    columns = ['label', 'all-p', 'prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    columns = [c for c in columns if c in keys]
    df = pd.DataFrame(table).set_index('pair')
    # set nana to 0 before median
    #df = df.fillna(0.0)
    median = df.median(axis=0)
    df.loc['median'] = median
    df = df[columns]

    return df


def main():
    
    
    model_names = ['giga_full_updated', 'wiki_updated']
    analysis_names = ['dist-mean', 'dist-max']
    properties = utils.get_properties() 
    
    for model_name in model_names:
        for analysis_name in analysis_names:
            if analysis_name.endswith('-mean'):
                calc = 'mean'
            elif analysis_name.endswith('-max'):
                calc = 'max'
        
            # properties
            level = 'properties'
            df = get_evidence_dist_properties(model_name, calc)
            # to file
            path_dir = f'../analysis/{model_name}/{level}/'
            os.makedirs(path_dir, exist_ok=True)
            path_file = f'{path_dir}/{analysis_name}.csv'
            df.to_csv(path_file)
            
#             # pairs
#             level = 'pairs'
#             df = get_evidence_dist_concepts(model_name, properties, calc)
#             # to file
#             path_dir = f'../analysis/{model_name}/pairs/'
#             os.makedirs(path_dir, exist_ok=True)
#             path_file = f'{path_dir}/{analysis_name}.csv'
#             df.to_csv(path_file)

#             # relations
#             level = 'relations'
#             pair_score_dict = relations.load_scores(analysis_name, model_name)
#             df = relations.relation_overview(pair_score_dict)
#             # to file:
#             path_dir = f'../analysis/{model_name}/{level}/'
#             os.makedirs(path_dir, exist_ok=True)
#             path_file = f'{path_dir}/{analysis_name}.csv'
#             df.to_csv(path_file)
      
        
      
        
    
if __name__ == '__main__':
    main()