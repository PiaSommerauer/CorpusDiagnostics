import os
from collections import Counter, defaultdict
import csv
import pandas as pd


def get_categories(prop, model_name):
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}'
    categories = set()
    for d in os.listdir(path_dir):
        if '.' not in d:
            categories.add(d)
    return categories


def get_top_scores_cat(data):
    # sort data by f1
    f1_dict = defaultdict(list)
    top_word_dicts = dict()
    for d in data:
        f1 = float(d['f1'])
        f1_dict[f1].append(d)
    top_score = max(list(f1_dict.keys()))
    top_dicts = f1_dict[top_score]
    for d in top_dicts:
        top_word_dicts[d['context']] = d
    return top_score, top_word_dicts
   
    

def get_top_distinctive_contexts(properties, model_name, top_cutoff=3, concept_cutoff=3):
    aggregation_name = 'aggregated-tfidf-raw-10000-categories'
    ann_name = f'annotation-tfidf-top_{top_cutoff}_{concept_cutoff}-raw-10000-categories'
    path_results = f'../results/{model_name}/tfidf-raw-10000/each_target_vs_corpus_per_category'
    table = []
    for prop in properties:
        categories = get_categories(prop, model_name)
        path_dir_agg = f'../analysis/{model_name}/{aggregation_name}/{prop}'
        #top_f1_scores = defaultdict(list)
        #word_dicts = defaultdict(list)
        top_scores = []
        top_words = set()
        for cat in categories:
            path = f'{path_dir_agg}/{cat}.csv'
            # load file containing all contexts
            with open(path) as infile:
                data = list(csv.DictReader(infile))
            # only iclude if cat has more than x positive examples:
            n_pos = int(data[0]['total_pos'])
            n_neg = int(data[0]['total_neg'])
            if n_pos > 9 and n_neg > 9:
                top_score, top_dicts = get_top_scores_cat(data)
                top_words.update(top_dicts.keys())
                top_scores.append(top_score)
              
                #for w, d in top_dicts.items():
                 #   top_f1_scores[w].append(float(top_score))
                  #  word_dicts[w].append(d)
        # get mean over categories:
#         mean_word_dict = defaultdict(list)
#         for w, scores in top_f1_scores.items():
#             mean = sum(scores)/len(scores)
#             mean_word_dict[mean].append(w)
        #overall top score and words:
        #top_mean_score = max(list(mean_word_dict.keys()))
        mean_top_score = sum(top_scores)/len(top_scores)
        top_contexts  = ' '.join(top_words)
        #top_words_mean = mean_word_dict[top_mean_score]
        # get p and r from dicts:
#         ps = []
#         rs = []
#         for w in top_words_mean:
#             ds = word_dicts[w]
#             for d in ds:
#                 rs.append(float(d['r']))
#                 ps.append(float(d['p']))
#         mean_p = sum(ps)/len(ps)
#         mean_r = sum(rs)/len(rs)
            
        # get n extracted candidates
        f_ann =  f'../analysis/{model_name}/{ann_name}/{prop}/annotation-updated.csv'
        with open(f_ann) as infile:
            data = list(csv.DictReader(infile))
        n_contexts = len(data)
     
        d_prop = dict()
        d_prop['property'] = prop
        d_prop['n_contexts'] = n_contexts
        d_prop['f1-mean'] = mean_top_score
        #d_prop['p-mean'] = mean_p
        #d_prop['r-mean'] = mean_r
        d_prop['contexts'] = top_contexts
        table.append(d_prop)
    return table