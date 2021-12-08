import json
import pandas as pd
from statistics import stdev
import os
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from collections import defaultdict

def get_properties():
    properties = []
    for path in os.listdir('../data/aggregated_semantic_info/'):
        prop = path.split('.')[0]
        if 'female-' not in prop and prop != '':
            properties.append(prop)
    return properties
            
        

def load_data(prop):
    path = f'../data/aggregated_semantic_info/{prop}.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict

# def combine_contexts(prop, label, target_words, model):
    
#     c = ''
#     for target in target_words:
#         path = f'../contexts/{model}/vocab/{target}.txt'
#         if os.path.isfile(path):
#             #print('found file', target)
#             with open(path) as infile:
#                 c += f' {infile.read()}'
#         else:
#             print('not found', target)
#             print(path)
#     path_dir = f'../contexts/{model}/{prop}/{label}'
#     os.makedirs(path_dir, exist_ok=True)
#     path_all = f'{path_dir}/ALL.txt'
#     with open(path_all, 'w') as outfile:
#         outfile.write(c)
        
def get_tfidf(path_target, paths_corpus, class_label, cnt_type='raw', max_features = 1000):
         
    # first path is pos, rest is neg:
    paths = [path_target]
    paths.extend(paths_corpus)
    if cnt_type == 'raw':
        vectorizer = tfidf(input = 'filename', max_features=max_features)
    elif cnt_type == 'binary':
        vectorizer = tfidf(input = 'filename', binary=True, max_features=max_features)
    x = vectorizer.fit_transform(paths)
    x = x.toarray()

    vocab = vectorizer.get_feature_names()


    # number_of_paths
    n_paths = len(paths)
    vec_target = x[0]

    vec_dict = dict()
    vec_dict['target'] = vec_target
    vec_dict['mean_corpus'] =  []
    for vec in x.T:
        vec_corpus = vec[1:]
        mean_corpus = sum(vec_corpus)/len(vec_corpus)
        vec_dict['mean_corpus'].append(mean_corpus)
    for n, path in enumerate(paths):
        if n > 0:
            vec_dict[path] = x[n]
    df = pd.DataFrame(vec_dict, index = vocab).sort_values('target', ascending=False)
    diff = []
    for t, corp in zip(df['target'], df['mean_corpus']):
        diff.append(t-corp)
    df['diff'] = diff
    return df

def get_category_dict(prop_dict):
    category_concept_dict = defaultdict(set)
    for concept,  d in prop_dict.items():
        for cat in d['categories'].keys():
            category_concept_dict[cat].add(concept)
    return category_concept_dict


def tfidf_target_vs_corpus(prop, model, cnt_type, target_label, max_features):
    condition = 'each_target_vs_corpus_per_category'


    prop_dict = load_data(prop)
    category_concept_dict = get_category_dict(prop_dict)
    labels_pos =  ['all', 'all-some', 'few-some', 'some']
    labels_neg  = ['few']

    
    cat_concept_dict_merged = defaultdict(set)
    
    all_concepts = set()
    for cat, concepts in category_concept_dict.items():
        all_concepts.update(concepts)
    
    # collect corpus paths
    concept_path_dict = dict()
        
    for concept in all_concepts:
        path = f'../contexts/{model}/vocab/{concept}.txt'
        if os.path.isfile(path):
            #print('found path corpus', path)
            concept_path_dict[concept]=path
    
    # get rid of categories that are too small and catch concepts
    for cat, concepts in category_concept_dict.items():
        
        target_pos = [c for c in concepts if prop_dict[c]['ml_label'] in  labels_pos]
        target_neg = [c for c in concepts if prop_dict[c]['ml_label'] in  labels_neg]
        paths_pos = [concept_path_dict[c] for c in target_pos  if c in concept_path_dict]
        paths_neg = [concept_path_dict[c] for c in target_neg if c in concept_path_dict]
    
        
        if len(paths_pos) == 0 or len(paths_neg) == 0:
            cat = 'no-cat'
        cat_concept_dict_merged[cat].update(concepts)
        cat_concept_dict_merged['all'].update(concepts)
        
        
    for cat, concepts in cat_concept_dict_merged.items():
        
        target_pos = [c for c in concepts if prop_dict[c]['ml_label'] in  labels_pos]
        target_neg = [c for c in concepts if prop_dict[c]['ml_label'] in  labels_neg]

        if target_label == 'pos':
            target = target_pos
            corpus = target_neg
        elif target_label == 'neg':
            target = target_neg
            corpus = target_pos
            
        paths_corpus = [concept_path_dict[c] for c in corpus  if c in concept_path_dict]
        paths_target = [concept_path_dict[c] for c in target if c in concept_path_dict]
        
        
        if len(paths_corpus) > 0 and len(paths_target) > 0:
            path_results_dir = f'../results/{model}/tfidf-{cnt_type}-{max_features}/{condition}/{prop}/{cat}/{target_label}'
        
            os.makedirs(path_results_dir, exist_ok=True)
            for concept in target:
                if concept in concept_path_dict:
                    path_target = concept_path_dict[concept]
      
                    df = get_tfidf(path_target, paths_corpus, target_label, cnt_type, max_features)
                    df = df[['target', 'mean_corpus', 'diff']]
                    df.to_csv(f'{path_results_dir}/{concept}.csv') 
                else:
                    print('no contexts', concept)
        else:
            print('not enough data', cat)
            print('positive examples', len(path_target))
            print('negative examples', len(paths_corpus))
    

def main():
 
    model = 'wiki_updated'
    cnt_type = 'raw'
    max_features = 10000
    test = True
    properties_test = []
    labels = ['pos', 'neg']
    
    #properties = get_properties()
    properties = ['swim' ]
    #properties = [p for p in properties if p not in properties_test]
    
    for prop in properties:
        print(prop)
        for target_label in labels:
            print(target_label)
            tfidf_target_vs_corpus(prop, model, cnt_type, target_label, max_features)

            
            
if __name__ == '__main__':
    main()
    
    
       