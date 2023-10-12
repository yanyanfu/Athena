# the codebase for Athena is based on https://github.com/microsoft/CodeBERT/
import os
import torch
import copy
import argparse
import json

import data
import extractor
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from git import Git, Repo
from collections import defaultdict
from scipy.spatial.distance import cdist
from parser import remove_comments_and_docstrings
from typing import List, Dict, Any, Set, Optional
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)

from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec
from gensim import similarities
import gensim.downloader as api
from gensim.models import KeyedVectors


def clone_repo(dataset_csv, project_path):
    for _, row in dataset_csv.iterrows():
        repo_path = Path(project_path) / row['repo']
        if not repo_path.exists():
            try:
                repo_path.mkdir(parents = True)
                Repo.clone_from (row['github_link'].split('/commit/')[0], str(repo_path))
            except Exception as e:
                print (e)


def search_query(file_path, method_name, method_df, lang):
    overload_idxes = []
    idxes = method_df.index[
        method_df.path == str(file_path)
    ].tolist()  
    for idx in idxes:
        mtd = remove_comments_and_docstrings(method_df.method[idx],lang).split('\n')
        for i in range(len(mtd)):
            if mtd[i].lstrip().startswith('@') or not mtd[i]:
                continue
            if '(' not in mtd[i]:
                mtd[i] += mtd[i+1]
            if method_name == mtd[i].split('(')[0].split()[-1].split('*/')[-1]:
                overload_idxes.append (idx)
            break 
    return overload_idxes, idxes


def convert_examples_to_features_code(item, parser, lang):
    code = remove_comments_and_docstrings(item,lang)
    tree = parser.parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    tokens_index = tree_to_token_index(root_node)     
    code = code.split('\n')
    code_tokens_tmp = [index_to_code_token(x,code) for x in tokens_index]
    code_tokens_tmp = [x for x in code_tokens_tmp if x]
    return code_tokens_tmp


def calculate_metric(all_distances, distances):
    query_size = all_distances.shape[0]   
    ranks = np.zeros(distances.shape)
    for i in range(query_size):
        rank = []
        grd_truth = np.expand_dims (distances[:, i], axis = -1)
        rank = np.sum (all_distances <= grd_truth, axis = -1)
        ranks[:, i] = rank

    ranks [ranks == 0] = 1
    sort_ranks = np.sort(ranks, kind = 'mergesort')
    grd_truth_size = np.sum (sort_ranks < all_distances.shape[1], axis = -1)

    rank = sort_ranks[:,0]
    rr = 1.0 / rank
    avep, hit_10 = np.zeros(query_size), np.zeros(query_size)
    for i in range(query_size):
        if rank[i] <= 10:
            hit_10[i] = 1
        p_list = []
        for j in range(query_size):
            if sort_ranks[i][j] == all_distances.shape[1]:
                break
            p_list.append((j+1)/sort_ranks[i][j])
            avep[i] = np.mean(p_list) if p_list else 0.0
    return rank, rr, avep, hit_10, sort_ranks, grd_truth_size


def main():

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--project_path", default='./athena_reproduction_package/projects', type=str,
                        help="the path of the downloaded projects by using GitHub URL from the dataset")    
    parser.add_argument("--pretrained_model_name", default='microsoft/graphcodebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--finetuned_model_path", default='./athena_reproduction_package/finetuned_models/doc2vec.model', type=str,
                        help="The model checkpoint after finetuned on the code search task.")
    parser.add_argument("--lang", default='java', type=str,
                        help="The programming language for parsing")
    parser.add_argument('--output_dir', default='./athena_reproduction_package/results/doc2vec', help='Path where to save results.')
    parser.add_argument("--weight", default=0.5, type=float,
                        help="The weight used to balance the method and its neighbor method information")
    parser.add_argument("--MAX", default=10000, type=int,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--nebr_num", default=30, type=int,
                        help="# of third-order nebrs is taken into considerating")
    parser.add_argument("--version", default='baseline', type=str,
                        help="The version used to obtain the results: athena or baseline")
    args = parser.parse_args()

    # read the dataset and clone the repositories
    dataset_csv = pd.read_csv('./dataset/alexandria.csv')
    clone_repo(dataset_csv, args.project_path)
    dataset = defaultdict(lambda: defaultdict(list))
    for _, row in dataset_csv.iterrows():  
        dataset[row['repo']][row['parent_commit']].append(
            row['file_path'] + '<sep>' + row['method_name']
        )

    LANGUAGE = Language('parser/my-languages.so', args.lang)        
    parser_java = Parser()
    parser_java.set_language(LANGUAGE) 

    results = [[] for i in range(3)]
    ## obtain the results for each query in the co-changed methods
    for repo in tqdm(dataset):
        for parent_commit in tqdm(dataset[repo]): 
            # build the call graph                                        
            repo_path = Path(args.project_path) / repo
            repo_cg = data.SoftwareRepo(repo_path, parent_commit)
            method_df = repo_cg.method_df
            
            # store indexes of query methods. Two-dimensional list to handle overloaded methods
            query_overload_idxes, file_idxes, method_path = [], [], []
            for path_line in dataset[repo][parent_commit]:
                path = path_line.split('<sep>')
                overload_idxes, idxes = search_query (repo_path / path[0], path[1], method_df, args.lang)
                if overload_idxes:
                    query_overload_idxes.append(overload_idxes)
                    file_idxes.append(idxes)
                    method_path.append(os.path.join(path[0], path[1]))

            corpus_vecs = []         
            model = Doc2Vec.load(args.finetuned_model_path) 
            for mtd in method_df.method.values:
                corpus_input = convert_examples_to_features_code(mtd, parser_java, args.lang)
                corpus_vecs.append(model.infer_vector(corpus_input))
            query_size, corpus_size = len(query_overload_idxes), len(corpus_vecs)
            query_vecs = np.zeros ([query_size, len(corpus_vecs[0])])              
            for i in range(query_size):
                query_vecs[i] = corpus_vecs[query_overload_idxes[i][0]] 

            all_distances_origin = cdist (query_vecs, corpus_vecs, metric = 'cosine')                
            distances = np.zeros ([query_size, query_size])

            for level in range(3):
                all_distances = copy.deepcopy (all_distances_origin) 
                distances = np.zeros ([query_size, query_size])
                for i in range(query_size):                          
                    # set the distance between the query and itself to MAX value
                    for idx in query_overload_idxes[i]:
                        all_distances[i][idx] = args.MAX
                    if level % 3 == 1:
                        for idx in range(corpus_size):
                            if idx not in file_idxes[i]:
                                all_distances[i][idx] = args.MAX
                    elif level % 3 == 2:
                        for idx in file_idxes[i]:
                            all_distances[i][idx] = args.MAX
                    for j in range(query_size):
                        if (len(query_overload_idxes[j]) == 1):
                            distances[i][j] = all_distances[i][query_overload_idxes[j][0]]
                        else:
                            dist = []
                            for k in query_overload_idxes[j]:
                                dist.append (all_distances[i][k])
                            distances[i][j] = min (dist)
            
                # compute rank matrix (query * grd_truth)
                rank, rr, avep, hit_10, sort_ranks, grd_truth_size = calculate_metric(all_distances, distances)
                for i in range(query_size):
                    if rank[i] != corpus_size:
                        results[level].append({
                            "repo": repo,
                            "parent commit": parent_commit,
                            "method path": method_path[i],
                            "rank": rank[i],
                            "RR": rr[i],
                            "AP": avep[i],
                            "hit@10": hit_10[i],
                            "sort_ranks": sort_ranks[i].tolist(),
                            "ground truth size": grd_truth_size[i],
                            "inner corpus size": len(file_idxes[i]) - len(query_overload_idxes[i]),
                            "outer corpus size": all_distances.shape[1] - len(file_idxes[i]),
                            "repo size": all_distances.shape[1] 
                        })                 
                results_csv = pd.DataFrame(results[level])
                write_path = os.path.join(args.output_dir, 'results_' + str((level)+1) + '.csv')
                results_csv.to_csv(write_path, index=False)                             


if __name__ == "__main__":
    main()