import json

from tqdm import tqdm
import os
import sys
from collections import defaultdict
import multiprocessing
from collections import defaultdict
import time
import itertools

def load_data(prop):
    path = '../data/aggregated/'+prop+'.json'
    with open(path) as infile:
        prop_dict = json.load(infile)
    return prop_dict

# get vocab to check what is in vocab

def load_vocab(path):
    vocab_counts = dict()
    with open (path) as infile:
        for line in infile:
            word, cnt = line.strip().split(' ')
            vocab_counts[word] = cnt
    return vocab_counts

# def load_pairs(path):
#     pairs = []
#     with open(path) as infile:
#         for line in infile:
#             pair = line.strip().split(' ')
#             pairs.append(pair)
#     return pairs



def extract_contexts(targets_pair_line):
    targets, pair_line = targets_pair_line
    pair = pair_line.strip().split(' ')
    w, c = pair
    target_contexts=defaultdict(list)
    for target in targets:
        if target == w:
            target_contexts[target].append(c)
    return target_contexts



def get_batches(n_lines, n_batches):
    batch_size = round(n_lines / n_batches)
    batches = []
    start = 0
    for n in range(n_batches):
        end = start + batch_size
        batches.append((start, end))
        start = end
    # add final batch
    last_b = batches[-1]
    end = last_b[1]
    if end < n_lines:
        batches.append((end, n_lines + 1))
    return batches
                        
def main():


    #prop = sys.argv[1]
    path_dir = '../../data/volume_1/models/giga_full/counts'
    model_name = 'giga_full'
    if model_name == 'test':
        pair_path = 'pairs-test'
    else:
        pair_path =  f'{path_dir}/pairs'


    # for progress bar  - use wc -l on terminal to get line number of pairs file
    n_lines = 1886693907
    n_batches = 1000

    # if model_name == 'test':
    #     n_lines = 100
    #     n_batches = 10

    # test:
    #n_batches = 1
    # n_lines = 6

    batches = get_batches(n_lines, n_batches)
    print(f'created {len(batches)} batches.')
    path_vocab ='../data/vocab.txt'
    if model_name == 'test':
        path_vocab = '../data/vocab-test.txt'
    with open(path_vocab) as infile:
        targets = set(infile.read().strip().split('\n'))
    print('loaded vocab of size: ', len(targets))

    # # get already collected:
    # already_collected = set()
    # collected_dir = f'../contexts/{model_name}/vocab/'
    # if os.path.isdir(collected_dir):
    #     for f in os.listdir(collected_dir):
    #         w = f.split('.')[0]
    #         already_collected.add(w)
    # take 100
    # print('already collected contexts for:')
    # print(already_collected)
    # targets_remaining = list(targets.difference(already_collected))
    # print('Targets remaining:', len(targets_remaining))
    # vocab_size = 400
    # if len(targets_remaining) > vocab_size:
    #     targets_to_collect = targets_remaining[:vocab_size]
    # else:
    #     targets_to_collect = targets_remaining
    batch_cnt = 0
    outputs = []
    output_batches = 0
    paths = set()
    total_dur = 0

    for start, end in batches:
        with open(pair_path) as infile:
            batch_cnt += 1
            start_time = time.time()
            print(f'processing batch {batch_cnt} of {len(batches)}')
            print(f'batch size: {end-start}')
            #output_dicts_batch = []

            my_lines = itertools.islice(infile, start, end)

            # use 5 cpus
            po = multiprocessing.Pool(10)
            #extract_contexts(targets_pair_line)
            out = po.map(extract_contexts, [(targets, pair_line) for pair_line in my_lines])


            po.close()
            po.join()
            #outputs.extend(out)
            print(f'processed {len(out)} lines')

            print('writing to file')
            print(f'processing {len(outputs)} outputs')
            n_contexts = 0
            context_dict_all = defaultdict(list)
            for context_dict in out:
                for target, contexts in context_dict.items():
                    n_contexts += len(contexts)
                    context_dict_all[target].extend(contexts)
            print('number of targets:', n_contexts)
            output_batches = 0
            #outputs = []
            # write to datablock instead of repo
            #path_dir = f'../contexts/{model_name}/vocab'
            path_dir = f'../../data/volume_1/contexts/{model_name}/vocab'
            os.makedirs(path_dir, exist_ok=True)
            for target, contexts in context_dict_all.items():
                path_target = f'{path_dir}/{target}.txt'
                if path_target in paths:
                    mode = 'a'
                else:
                    mode = 'w'
                with open(path_target, mode) as outfile:
                    outfile.write('\n'.join(contexts)+'\n')
                paths.add(path_target)

        end_time = time.time()
        dur = (end_time-start_time)/60 # min
        total_dur += dur
        mean_dur = total_dur/batch_cnt
        remaining_batches = len(batches) - batch_cnt
        est_dur = (remaining_batches * mean_dur )/ 60 #hours
        print(f'Finished batch {batch_cnt}')
        print(f'{remaining_batches} batches remaining')
        print(f'This batch took {dur} minutes (mean duration: {mean_dur} minutes)')
        print(f'Projected duration remaining: {est_dur} hours')



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
