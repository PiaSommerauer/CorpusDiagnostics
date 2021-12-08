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


def get_context_strengths_concept(prop, concept, label, model_name, 
                                evidence_type_dict):
  
    context_strength = dict()
    # categories
    categories = utils.get_categories_concept(prop, concept, model_name)

    # collect all relevant context of the concept
    
    context_strengths = defaultdict(list)
    
    #print(evidence_type_dict.keys())
  
    for category in categories:

        # collect strengths:
        dir_path = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
        full_path = f'{dir_path}/{prop}/{category}/{label}/{concept}.csv'

        with open(full_path) as infile:
            data = list(csv.DictReader(infile))
        for d in data:
            context = d['']
            strength = float(d['target'])
            if context in evidence_type_dict:
                context_strengths[context].append(strength)
    for context, strengths in context_strengths.items():
        context_strength[context] = sum(strengths)/len(strengths)   
    return context_strength 

                


def get_evidence_strength(prop, model_name, evidence_type_dict, calc):
    
    evidence_strength_dict = dict()
    
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
            

    concept_label_dict = dict()
    concepts_pos = utils.get_examples(model_name, prop, 'pos')
    concepts_neg = utils.get_examples(model_name, prop, 'neg')
    for c in concepts_pos:
        concept_label_dict[c] = 'pos'
    for c in concepts_neg:
        concept_label_dict[c] = 'neg'
    
    all_contexts_strengths = defaultdict(list)
    context_strength_dict = dict()
    for concept, label in concept_label_dict.items():
        context_strengths_concept =  get_context_strengths_concept(prop, concept, label, model_name, 
                            evidence_type_dict)
        for c, strength in context_strengths_concept.items():
            all_contexts_strengths[c].append(strength)
    for c, strengths in all_contexts_strengths.items():
        mean = sum(strengths)/len(strengths)
        context_strength_dict[c] = mean
    
    
    for t, contexts in type_evidence_dict.items():
        strengths = [context_strength_dict[c] for c in contexts if c in context_strength_dict]
        if calc == 'mean':
            score = sum(strengths)/len(strengths)
        elif calc == 'max':
            score = max(strengths)
        evidence_strength_dict[t] = score
        
    return evidence_strength_dict


def get_evidence_strength_concept(prop, concept,  label, model_name, evidence_type_dict, calc):
    
    evidence_strength_dict = dict()
    
    type_evidence_dict = defaultdict(list)
    
    for c, t in evidence_type_dict.items():
        type_evidence_dict[t].append(c)
        if t in ['p', 'n', 'l']:
            t_c = 'prop-specific'
            type_evidence_dict[t_c].append(c)
        elif t in ['i', 'r', 'b']:
            t_c = 'non-specific'
            type_evidence_dict[t_c].append(c)
    context_strength_dict =  get_context_strengths_concept(prop, concept, 
                                                 label, model_name, evidence_type_dict)
    
    for t, contexts in type_evidence_dict.items():
        strengths = [context_strength_dict[c] for c in contexts if c in context_strength_dict]
        if len(strengths) > 0:
            if calc == 'mean':
                score = sum(strengths)/len(strengths)
            elif calc == 'max':
                score = max(strengths)
        else:
            score = 0
        evidence_strength_dict[t] = score
        
    return evidence_strength_dict

     


def get_evidence_strength_properties(model_name, calc):
    
    table = []
    
    properties = utils.get_properties()

    for prop in properties:
        evidence_type_dict = utils.load_evidence_type_dict(prop, model_name)
        evidence_strength = get_evidence_strength(prop, model_name, evidence_type_dict, calc)
        evidence_strength['property'] = prop
        table.append(evidence_strength)
        print('finished prop', prop)
    
    columns = ['all', 'all-p', 'prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
    df = pd.DataFrame(table).set_index('property')[columns]
    # set nana to 0 before median
    #df = df.fillna(0.0)
    median = df.median(axis=0)
    df.loc['median'] = median
    
    return df



def get_evidence_strength_concepts(model_name, properties, calc):
    
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
            ev_strength_concept =  get_evidence_strength_concept(prop, concept,  
                                                             label, model_name, 
                                                             evidence_type_dict, calc)
            ev_strength_concept['label'] = label
            keys.update(ev_strength_concept.keys())
            ev_strength_concept['pair'] = (prop, concept)
            table.append(ev_strength_concept)
        print('finished prop', prop)
        
    columns = ['label', 'prop-specific', 'non-specific', 'p', 'l', 'n', 'i', 'r', 'b', 'u']
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
    evidence_types = ['prop-specific', 'non-specific', 'l', 'p']
    #analysis_names = ['dist-mean', 'dist-max']
    analysis_names = ['str-mean', 'str-max']
    properties = utils.get_properties() 
    for model_name in model_names:
        for analysis_name in analysis_names:
            if analysis_name.endswith('-mean'):
                calc = 'mean'
            elif analysis_name.endswith('-max'):
                calc = 'max'
        
#             # properties
#             level = 'properties'
#             print()
#             print(level)
#             df = get_evidence_strength_properties(model_name, calc)
#             # to file
#             path_dir = f'../analysis/{model_name}/{level}/'
#             os.makedirs(path_dir, exist_ok=True)
#             path_file = f'{path_dir}/{analysis_name}.csv'
#             df.to_csv(path_file)
# # # 
#             # pairs
#             level = 'pairs'
#             print()
#             print(level)
#             df = get_evidence_strength_concepts(model_name, properties, calc)
#             # to file
#             path_dir = f'../analysis/{model_name}/pairs/'
#             os.makedirs(path_dir, exist_ok=True)
#             path_file = f'{path_dir}/{analysis_name}.csv'
#             df.to_csv(path_file)

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