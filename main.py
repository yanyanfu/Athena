
# the codebase for Athena is based on https://github.com/microsoft/CodeBERT/
import os
import torch
import copy
import argparse

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
       
def clone_repo(dataset_csv, project_path):
    for _, row in dataset_csv.iterrows():
        repo_path = Path(project_path) / row['repo']
        if not repo_path.exists():
            try:
                repo_path.mkdir(parents = True)
                Repo.clone_from (row['github_link'].split('/commit/')[0], str(repo_path))
            except Exception as e:
                print (e)


def search_query(file_path, method_name, method_df):
    overload_idxes = []
    idxes = method_df.index[
        method_df.path == str(file_path)
    ].tolist()  
    for idx in idxes:
        mtd = remove_comments_and_docstrings(method_df.method[idx],'java').split('\n')
        for i in range(len(mtd)):
            if mtd[i].lstrip().startswith('@') or not mtd[i]:
                continue
            if '(' not in mtd[i]:
                mtd[i] += mtd[i+1]
            if method_name == mtd[i].split('(')[0].split()[-1].split('*/')[-1]:
                print(method_name, end = ' ')
                overload_idxes.append (idx)
            break 
    return overload_idxes, idxes


def get_noralimized_adjacency(src_nodes, trgt_nodes, corpus_size, weight):
    adj, adj_sec = np.zeros ([corpus_size, corpus_size]), np.zeros ([corpus_size, corpus_size])
    for i in range(corpus_size):
        direct_nebr = set()
        for j in range(len(src_nodes)):
            if i == src_nodes[j]:
                adj[i][trgt_nodes[j]] = 1
                direct_nebr.add(trgt_nodes[j])
            if i == trgt_nodes[j]:
                adj[i][src_nodes[j]] = 1  
                direct_nebr.add(src_nodes[j])
        for j in range(len(src_nodes)):
            for k in direct_nebr:
                if k == src_nodes[j] and not adj[i][trgt_nodes[j]]:
                    adj_sec[i][trgt_nodes[j]] = 1
                if k == trgt_nodes[j] and not adj[i][src_nodes[j]]:
                    adj_sec[i][src_nodes[j]] = 1
        adj[i][i] = 0
        adj_sec[i][i] = 0

    degree = np.sum (adj, axis = 1)
    for i in range(len(degree)):
        if degree[i]:
            degree[i] = 1 / np.sqrt(degree[i])  
    degree_sec = np.sum (adj_sec, axis = 1)
    for i in range(len(degree_sec)):
        if degree_sec[i]:
            degree_sec[i] = 1 / np.sqrt(degree_sec[i]) 
    adj = np.matmul(np.matmul(np.diag(degree), adj * weight), np.diag(degree)) + np.identity(corpus_size)
    adj += np.matmul(np.matmul(np.diag(degree_sec), adj_sec * weight), np.diag(degree_sec))
    
    return adj  


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
    return rank, rr, avep, hit_10, grd_truth_size


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--project_path", default='../athena_reproduction_package/projects', type=str,
                        help="the path of the downloaded projects by using GitHub URL from the dataset")    
    parser.add_argument("--pretrained_model_name", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--finetuned_model_path", default='../athena_reproduction_package/finetuned_models/codebert.bin', type=str,
                        help="The model checkpoint after finetuned on the code search task.")
    parser.add_argument("--weight", default=0.5, type=float,
                        help="thw weight used to balance the method and its neighbot method information")
    parser.add_argument("--MAX", default=10000, type=int,
                        help="The model checkpoint for weights initialization.")
    args = parser.parse_args()

    if args.pretrained_model_name == 'microsoft/codebert-base':
        embed = extractor.embed_codebert(args.pretrained_model_name, args.finetuned_model_path)
    elif args.pretrained_model_name == 'microsoft/graphcodebert-base':
        embed = extractor.embed_graphcodebert(args.pretrained_model_name, args.finetuned_model_path)
    else:
        embed = extractor.embed_unixcoder(args.pretrained_model_name, args.finetuned_model_path)    
    embed.load_finetuned_model()       

    dataset_csv = pd.read_csv('./dataset/alexandria.csv')
    clone_repo(dataset_csv, args.project_path)
    dataset = defaultdict(lambda: defaultdict(list))
    for _, row in dataset_csv.iterrows():  
        dataset[row['repo']][row['parent_commit']].append(
            row['file_path'] + '<SPLIT>' + row['method_name']
        )
    
    results = [[] for i in range(3)]
    for repo in tqdm(dataset):
        for parent_commit in tqdm(dataset[repo]):                                         
            repo_path = Path(args.project_path) / repo
            repo_cg = data.SoftwareRepo(repo_path, parent_commit)
            method_df = repo_cg.method_df
            src_nodes = repo_cg.call_edge_df.from_id.values
            trgt_nodes = repo_cg.call_edge_df.to_id.values
            
            corpus_vecs = embed.extract_corpus_vecs(method_df.method.values)

            # store indexes of query methods. Two-dimensional list to handle overload methods
            query_overload_idxes, file_idxes = [], []
            method_names, file_paths = [], []
            for path_line in dataset[repo][parent_commit]:
                path = path_line.split('<SPLIT>')
                file_path, method_name = path[0], path[1]
                overload_idxes, idxes = search_query (repo_path / file_path, method_name, method_df)
                if overload_idxes:
                    query_overload_idxes.append(overload_idxes)
                    file_idxes.append(idxes)

            query_size, corpus_size, edge_size = len(query_overload_idxes), len(corpus_vecs), len(src_nodes)
            adj = get_noralimized_adjacency(src_nodes, trgt_nodes, corpus_size, args.weight)                                            
            corpus_vecs = np.matmul(adj, corpus_vecs)                
            query_vecs = np.zeros ([query_size, corpus_vecs.shape[1]])              
            for i in range(query_size):
                query_vecs[i] = corpus_vecs[query_overload_idxes[i][0]]      
                
            # calculate query-grd_truth distances and query-corpus distances
            all_distances_origin = cdist (query_vecs, corpus_vecs, metric = 'cosine')                             
            for level in range(3):
                all_distances = copy.deepcopy (all_distances_origin)                
                distances = np.zeros ([query_size, query_size])
                for i in range(query_size):                          
                    # set the distance between the query and itself to MAX value
                    for idx in query_overload_idxes[i]:
                        all_distances[i][idx] = args.MAX
                    if level == 1:
                        for idx in range(corpus_size):
                            if idx not in file_idxes[i]:
                                all_distances[i][idx] = args.MAX
                    elif level == 2:
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
                rank, rr, avep, hit_10, grd_truth_size = calculate_metric(all_distances, distances)
                for i in range(query_size):
                    if rank[i] != corpus_size:
                        results[level].append({
                            "repo": repo,
                            "parent commit": parent_commit,
                            "rank": rank[i],
                            "RR": rr[i],
                            "AP": avep[i],
                            "hit@10": hit_10[i],
                            "ground truth size": grd_truth_size[i] - 1,
                            "inner corpus size": len(file_idxes[i]) - 1,
                            "outer corpus size": all_distances.shape[1] - len(file_idxes[i]),
                            "repo size": all_distances.shape[1] 
                        })                 
                results_csv = pd.DataFrame(results[level])
                write_path = './results_' + str(level+1) + '.csv'
                results_csv.to_csv(write_path, index=False)                             


if __name__ == "__main__":
    main()