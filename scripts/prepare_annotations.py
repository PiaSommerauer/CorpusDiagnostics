import os
from collections import Counter, defaultdict
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def get_categories(prop, model_name):
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}'
    categories = set()
    for d in os.listdir(path_dir):
        categories.add(d)
    return categories

def get_context_cnts(prop, cat, label, model_name):
    
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}'
    path_label = f'{path_dir}/{cat}/{label}'
    
    context_cnt = Counter()
    for f in os.listdir(path_label):
        full_path = f'{path_label}/{f}'
        if full_path.endswith('.csv'):
            with open(full_path) as infile:
                data = list(csv.DictReader(infile))
            for d in data:
                context = d['']
                diff = float(d['diff'])
                if diff > 0:
                    context_cnt[context] += 1
    return context_cnt
    
def get_n_concepts_total(prop, cat, model_name):
    
    analysis_type = 'tfidf-raw-10000/each_target_vs_corpus_per_category'
    path_dir = f'../results/{model_name}/{analysis_type}'
    path_dir = f'{path_dir}/{prop}'
    label = 'pos'
    path_pos = f'{path_dir}/{cat}/{label}'
    label = 'neg'
    path_neg = f'{path_dir}/{cat}/{label}'
    
    files_pos = [f for f in os.listdir(path_pos) if f.endswith('.csv')]
    files_neg = [f for f in os.listdir(path_neg) if f.endswith('.csv')]
    
    return len(files_pos), len(files_neg)

def get_f1_distinctiveness(n_pos, n_neg, total_pos, total_neg):
    
   
    total_instances = total_pos + total_neg
    labels = []
    [labels.append('pos') for i in range(total_pos)]
    [labels.append('neg') for i in range(total_neg)]
    pred_labels_pos = []
    for i in range(total_pos):
        if i < n_pos:
            pred_labels_pos.append('pos')
        else:
            pred_labels_pos.append('neg')
#     print(n_pos, total_pos)
#     print(pred_labels_pos.count('pos'), pred_labels_pos.count('neg'))
    
    pred_labels_neg = []
    for i in range(total_neg):
        if i < n_neg:
            pred_labels_neg.append('pos')
        else:
            pred_labels_neg.append('neg')
#     print(n_neg, total_neg)
#     print(pred_labels_neg.count('pos'), pred_labels_neg.count('neg'))
    
    predictions = pred_labels_pos + pred_labels_neg
    
    
    #print(len(labels), len(predictions))
    #print(pos_predictions, neg_predictions)
    
    p, r, f1, supp = precision_recall_fscore_support(labels, predictions, average = 'weighted', 
                                                     zero_division=0)
    #average='weighted'
    
    return p, r, f1


    
def aggregate_contexts(prop, cutoff, model_name):
    aggregation_name = 'aggregated-tfidf-raw-10000-categories'
    path_dir_agg = f'../analysis/{model_name}/{aggregation_name}/{prop}'
    os.makedirs(path_dir_agg, exist_ok = True)
    
    context_cnts_all = Counter()
    context_cat_dict = defaultdict(set)

    cats = get_categories(prop, model_name)

    for cat in cats:
        context_cnts_pos = get_context_cnts(prop, cat, 'pos', model_name)
        context_cnts_neg = get_context_cnts(prop, cat, 'neg', model_name)
        total_pos, total_neg = get_n_concepts_total(prop, cat, model_name)
        
        context_f1_dict = Counter()
        context_score_dict = defaultdict(dict)
        
        # get distinctiveness
        for c, cnt_pos in context_cnts_pos.most_common():
            cnt_neg = context_cnts_neg[c]
            p, r, f1 = get_f1_distinctiveness(cnt_pos, cnt_neg, total_pos, total_neg)
            context_f1_dict[c] = f1
            context_score_dict[c] = {'p': p,'r':r, 'f1': f1}
        
        table = []
        for c, f1 in context_f1_dict.most_common():
            scores = context_score_dict[c]
            d = dict()
            d['context'] = c
            d.update(scores)
            d['n_pos'] = context_cnts_pos[c]
            d['total_pos'] = total_pos
            d['n_neg'] = context_cnts_neg[c]
            d['total_neg'] = total_neg
            table.append(d)
        
        # collect and write to file
        f = f'{path_dir_agg}/{cat}.csv'
        
        header = table[0].keys()
        with open(f, 'w') as outfile:
            writer = csv.DictWriter(outfile, fieldnames = header)
            writer.writeheader()
            for d in table:
                writer.writerow(d)
        
                
def prepare_annotation(prop, model_name, cutoff=3, cutoff_concepts = 5):
    
    annotation_name = f'annotation-tfidf-top_{cutoff}_{cutoff_concepts}-raw-10000-categories'
    path_dir_annotation = f'../analysis/{model_name}/{annotation_name}/{prop}'
    os.makedirs(path_dir_annotation, exist_ok = True)
    f_annotation = f'../analysis/{model_name}/{annotation_name}/{prop}/annotation-updated.csv'
    
    # paths aggregated files:
    aggregation_name = 'aggregated-tfidf-raw-10000-categories'
    path_dir_agg = f'../analysis/{model_name}/{aggregation_name}/{prop}'

    
    # get categories
    cats = get_categories(prop, model_name)
    
    # collect all contexts and categories 
    context_cats_dict = defaultdict(set)
    
    # load top per category
    for cat in cats:
        path = f'{path_dir_agg}/{cat}.csv'
        with open(path) as infile:
            data = list(csv.DictReader(infile))
        # sort by f1
        f1_dict  = defaultdict(list)
        for d in data:
            f1 = d['f1']
            f1_dict[f1].append(d)
        scores = sorted(list(f1_dict.keys()), reverse=True)
        top_scores = scores[:cutoff]
        top_context_dicts = []
        for ts in top_scores:
            dicts = f1_dict[ts]
            for d in dicts:
                n_pos = int(d['n_pos'])
                if n_pos > cutoff_concepts:
                    top_context_dicts.append(d)
    
        contexts = [d['context'] for d in top_context_dicts]
        # record categories
        for c in contexts:
            context_cats_dict[c].add(cat)
    
    with open(f_annotation, 'w') as outfile:
        outfile.write('context,evidence_type,categories\n')
        for c, cats in context_cats_dict.items():
            outfile.write(f'{c}, ,{" ".join(cats)}\n')




