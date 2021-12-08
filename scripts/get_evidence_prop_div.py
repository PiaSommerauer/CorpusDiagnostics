from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import csv
import os

import utils
import relations


def get_evidence_prop_div(evidence_type_dict, cnt = 'prop'):
    
    evidence_prop_dict = dict()
    evidence_div_dict = dict()

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
        
    n_candidates = len(evidence_type_dict.keys())
                 
    for t, contexts in type_evidence_dict.items():
        n_contexts = len(contexts)
        p = n_contexts/n_candidates
        evidence_prop_dict[t] = p
        evidence_div_dict[t] = n_contexts
    if cnt == 'prop':
        result = evidence_prop_dict
    elif cnt == 'div':
        result = evidence_div_dict
    return result
        
        
def get_evidence_prop_div_concept_category(model_name, evidence_type_dict, 
                                           prop, concept, label, category, cnt = 'prop'):
    
    evidence_prop_dict = dict()
    evidence_div_dict = dict()
    
    dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    full_path = f'{dir_path}/{prop}/{category}/{label}/{concept}.csv'
    
    with open(full_path) as infile:
        data = list(csv.DictReader(infile))
    contexts = [d[''] for d  in data if float(d['diff']) > 0]
   
    context_candidates = [c for c in contexts if c in evidence_type_dict.keys()]
    n_candidates = len(context_candidates)
    evidence_context_dict = defaultdict(list)
    for c in context_candidates:
        if c in evidence_type_dict:
            t = evidence_type_dict[c]
            evidence_context_dict[t].append(c)
            if t in ['p', 'n', 'l']:
                t_c = 'prop-specific'
                evidence_context_dict[t_c].append(c)
                #print(concept, c, label)
            elif t in ['i', 'r', 'b']:
                t_c = 'non-specific'
                evidence_context_dict[t_c].append(c)
            if t  in ['p', 'n', 'l', 'i', 'r', 'b']:
                t_c = 'all-p'
                evidence_context_dict[t_c].append(c)
    for t, contexts in evidence_context_dict.items():
        n_contexts = len(contexts)
        evidence_div_dict[t] = n_contexts
        evidence_prop_dict[t] = n_contexts/n_candidates
        
    if cnt == 'prop':
        result = evidence_prop_dict
    elif cnt == 'div':
        result = evidence_div_dict
    return result
    
    
    
def get_evidence_prop_div_concept(prop, concept, label, 
                                  evidence_type_dict, model_name, cnt):
    categories = utils.get_categories_concept(prop, concept, model_name)
    ev_prop_concept = Counter()

    for cat in categories:
        ev_prop_concept_cat = get_evidence_prop_div_concept_category(model_name, evidence_type_dict, 
                                                                     prop, concept, label,
                                                                     cat, cnt = cnt)
       

        for ev, p in ev_prop_concept_cat.items():
            ev_prop_concept[ev] += p

    # calculate means
    for ev, p in ev_prop_concept.items():
        p_mean = p/len(categories)
        ev_prop_concept[ev] = p_mean
    return ev_prop_concept  


def get_evidence_prop_div_properties(model_name, cnt):
    
    table = []
    
    properties = utils.get_properties()

    for prop in properties:
        evidence_type_dict = utils.load_evidence_type_dict(prop, model_name)
        evidence_prop = get_evidence_prop_div(evidence_type_dict, cnt = cnt)
        evidence_prop['property'] = prop
        table.append(evidence_prop)
    
    columns = ['all', 'all-p', 'prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    df = pd.DataFrame(table).set_index('property')[columns]
    # set nana to 0 before median
    df = df.fillna(0.0)
    median = df.median(axis=0)
    df.loc['median'] = median
    
    return df



def get_evidence_prop_div_concepts(model_name, properties, cnt):
    
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

        for concept, label in concept_label_dict.items():
            ev_prop_concept =  get_evidence_prop_div_concept(prop, concept, label,
                                                             evidence_type_dict, 
                                                             model_name, cnt=cnt)
#             if label == 'neg':
#                 print(ev_prop_concept)
            ev_prop_concept['label'] = label
            keys.update(ev_prop_concept.keys())
            ev_prop_concept['pair'] = (prop, concept)
            table.append(ev_prop_concept)
        print('finished prop', prop)
        
    columns = ['label', 'prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    columns = [c for c in columns if c in keys]
    df = pd.DataFrame(table).set_index('pair')
    # set nana to 0 before median
    df = df.fillna(0.0)
    median = df.median(axis=0)
    df.loc['median'] = median
    df = df[columns]

    return df


def main():
    
    
    model_names = ['giga_full_updated', 'wiki_updated']
    analysis_names = ['proportion', 'diversity']
    properties = utils.get_properties()
    evidence_types = ['prop-specific', 'non-specific', 'l', 'p']
    
    for model_name in model_names:
        
        for analysis_name in analysis_names:
            if analysis_name == 'proportion':
                cnt = 'prop'
            else:
                cnt = 'div'
        
            #properties
#             level = 'properties'
#             df = get_evidence_prop_div_properties(model_name, cnt)
#             # to file:
#             path_dir = f'../analysis/{model_name}/{level}/'
#             os.makedirs(path_dir, exist_ok=True)
#             path_file = f'{path_dir}/{analysis_name}.csv'
#             df.to_csv(path_file)
          

#             # pairs
#             level = 'pairs'
#             df = get_evidence_prop_div_concepts(model_name, properties, cnt)
#             # to file
#             path_dir = f'../analysis/{model_name}/pairs/'
#             os.makedirs(path_dir, exist_ok=True)
#             path_file = f'{path_dir}/{analysis_name}.csv'
#             df.to_csv(path_file)
#             print('finished', analysis_name, level)
        
            # relations
            for evidence_type in evidence_types:
                level = 'relations'
                pair_score_dict = relations.load_scores(analysis_name, model_name, evidence_type)
                df = relations.relation_overview(pair_score_dict, evidence_type)
                # to file:
                path_dir = f'../analysis/{model_name}/{level}/'
                os.makedirs(path_dir, exist_ok=True)
                path_file = f'{path_dir}/{analysis_name}_{evidence_type}.csv'
                df.to_csv(path_file)

                # relations - hyp
                level = 'relations-hyp'
                df = relations.relation_overview(pair_score_dict, evidence_type, mode = 'hyp')
                # to file:
                path_dir = f'../analysis/{model_name}/{level}/'
                os.makedirs(path_dir, exist_ok=True)
                path_file = f'{path_dir}/{analysis_name}_{evidence_type}.csv'
                df.to_csv(path_file)
          
            

    
if __name__ == '__main__':
    main()