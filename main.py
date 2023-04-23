
# the codebase for Athena is based on https://github.com/microsoft/CodeBERT/
import os
import sys
import torch
import copy
import argparse
import multiprocessing

import data
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from git import Git, Repo
from collections import defaultdict
from scipy.spatial.distance import cdist
from tree_sitter import Language, Parser
from typing import List, Dict, Any, Set, Optional
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset

from parser import DFG_java
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from model import CodeBERTModel, GraphCodeBERTModel, UniXcoderModel


#load parsers
LANGUAGE = Language('parser/my-languages.so', 'java')        
parser = Parser()
parser.set_language(LANGUAGE) 
parser_gcb = [parser,DFG_java] 

def convert_examples_to_features_cb(item, model):
    code = remove_comments_and_docstrings(item,lang)
    tree = parser.parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    tokens_index = tree_to_token_index(root_node)     
    code = code.split('\n')
    code_tokens = [index_to_code_token(x,code) for x in tokens_index]
    code_tokens = [x for x in code_tokens if x]
    code_tokens = tokenizer.tokenize(' '.join(code_tokens))

    if model == 'codeBERT':
        code_tokens =[tokenizer.cls_token]+code_tokens[:code_length-2]+[tokenizer.sep_token]
    else:
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens[:code_length-4]+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length

    return code_ids


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        
        
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg
    

def convert_examples_to_features_gcb(item):
    #extract data flow
    code_tokens,dfg=extract_dataflow(item)
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    code_tokens=code_tokens[:code_length+data_flow_length-2-min(len(dfg),data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:code_length+data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=code_length+data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]          
    
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg)


class TextDataset(Dataset):
    def __init__(self, methods, pool = None):
        self.examples = []
        for method in methods:
            self.examples.append(convert_examples_to_features(method))      
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((code_length+data_flow_length, code_length+data_flow_length),dtype=np.bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx))


def search_query(method_name, file_path, method_df):
    overload_idxes = []
    idxes = method_df.index[
        method_df.path == str(repo_path / file_path)
    ].tolist()  
    for idx in idxes:
        mtd = remove_comments_and_docstrings(method_df.method[idx],lang).split('\n')
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


# calculate query-grd_truth distances
def calculate_metric(all_distances, distances, query_size):
    ranks = np.zeros(distances.shape)
    for i in range(query_size):
        rank = []
        grd_truth = np.expand_dims (distances[:, i], axis = -1)
        rank = np.sum (all_distances <= grd_truth, axis = -1)
        ranks[:, i] = rank

    ranks [ranks == 0] = 1
    sort_ranks = np.sort(ranks, kind = 'mergesort')
    grd_truth_size = np.sum (sort_ranks < corpus_len, axis = -1)

    rank = sort_ranks[:,0]
    rr = 1.0 / rank
    avep, hit_10 = np.zeros(query_size), np.zeros(query_size)
    for i in range(query_size):
        if rank[i] <= 10:
            hit_10[i] = 1
        p_list = []
        for j in range(query_size):
            if sort_ranks[i][j] == corpus_len:
                break
            p_list.append((j+1)/sort_ranks[i][j])
            avep[i] = np.mean(p_list) if p_list else 0.0
    return rr, avep, hit_10, grd_truth_size


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--project_path", default='./projects', type=str,
                        help="the path of the downloaded projects by using GitHub URL from the dataset")    
    parser.add_argument("--model_name_or_path", default='microsoft/codebert-base', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--finetuned_model_path", default='', type=str,
                        help="The model checkpoint for weights initialization.")

    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)     
    model = CodeBERTModel(model)
    model.load_state_dict(torch.load(args.finetuned_model_path),strict=False)
    model.to(device)


    dataset_csv = pd.read_csv('./dataset/alexandria.csv')
    dataset = defaultdict(lambda: defaultdict(list))
    for _, row in dataset_csv.iterrows():  
        dataset[row['repo']][row['parent_commit']].append(
            row['file_path'] + '<SPLIT>' + row['method_name']
        )

    results = [[] for i in range(3)]
    for repo in tqdm(dataset):
        for parent_commit in tqdm(dataset[repo]):                                         
            repo_path = Path(PROJECTS_PATH) / repo
            repo_cg = data.SoftwareRepo(repo_path, parent_commit)
            method_df = repo_cg.method_df
            src_nodes = repo_cg.call_edge_df.from_id.values
            trgt_nodes = repo_cg.call_edge_df.to_id.values

            # store indexes of query methods. Two-dimensional list to handle overload methods
            query_overload_idxes, file_idxes = []
            method_names, file_paths = [], []
            for path_line in dataset[repo][parent_commit]:
                path = path_line.split('<SPLIT>')
                overload_idxes, idxes = search_query (method_name, file_path, method_df)
                if overload_idxes:
                    query_overload_idxes.append(overload_idxes)
                    file_idxes.append(idxes)

            # two dimenstional list to store all the query ids and corpus ids
            corpus_inputs = []
            for mtd in method_df.method.values:
                corpus_inputs.append(convert_examples_to_features_code(mtd))
            query_size, corpus_size, edge_size = len(query_overload_idxes), len(corpus_inputs), len(src_nodes)

            corpus_dataset = TensorDataset(torch.tensor(corpus_inputs))
            corpus_sampler = SequentialSampler(corpus_dataset)
            corpus_dataloader = DataLoader(corpus_dataset, sampler=corpus_sampler, batch_size=128,num_workers=4)
            model.eval()
            query_vecs=[] 
            corpus_vecs=[]
            for batch in corpus_dataloader:
                corpus_inputs = batch[0].to(device)   
                with torch.no_grad():
                    corpus_vec= model(corpus_inputs)
                    corpus_vecs.append(corpus_vec.cpu().numpy()) 
            model.train()  
            corpus_vecs=np.concatenate(corpus_vecs,0)

            adj, adj_sec = np.zeros ([corpus_size, corpus_size]), np.zeros ([corpus_size, corpus_size])
            for i in range(corpus_size):
                direct_nebr = set()
                for j in range(edge_size):
                    if i == src_nodes[j]:
                        adj[i][trgt_nodes[j]] = 1
                        direct_nebr.add(trgt_nodes[j])
                    if i == trgt_nodes[j]:
                        adj[i][src_nodes[j]] = 1  
                        direct_nebr.add(src_nodes[j])
                for j in range(edge_size):
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

            # calculate query-grd_truth distances and query-corpus distances
            adj = np.matmul(np.matmul(np.diag(degree), adj * 0.5), np.diag(degree)) + np.identity(corpus_size)
            adj += np.matmul(np.matmul(np.diag(degree_sec), adj_sec * 0.5), np.diag(degree_sec))                                                  
            corpus_vecs = np.matmul(adj, corpus_vecs)                
            query_vecs = np.zeros ([query_size, corpus_vecs.shape[1]])              
            for i in range(query_size):
                query_vecs[i] = corpus_vecs_w[query_overload_idxes[i][0]]      
                
            # calculate query-grd_truth distances and query-corpus distances
            all_distances_origin = cdist (query_vecs, corpus_vecs_w, metric = 'cosine')                             
            for level in range(3):
                all_distances = copy.deepcopy (all_distances_origin)                
                distances = np.zeros ([query_size, query_size])
                for i in range(query_size):                          
                    # set the distance between the query and itself to MAX value
                    for idx in query_overload_idxes[i]:
                        all_distances[i][idx] = MAX
                    if level == 1:
                        for idx in range(corpus_size):
                            if idx not in file_idxes[i]:
                                all_distances[i][idx] = MAX
                    elif level == 2:
                        for idx in file_idxes[i]:
                            all_distances[i][idx] = MAX
                    for j in range(query_size):
                        if (len(query_overload_idxes[j]) == 1):
                            distances[i][j] = all_distances[i][query_overload_idxes[j][0]]
                        else:
                            dist = []
                            for k in query_overload_idxes[j]:
                                dist.append (all_distances[i][k])
                            distances[i][j] = min (dist)
               
                # compute rank matrix (query * grd_truth)
                rr, avep, hit_10, grd_truth_size = calculate_metric(all_distances, distances, query_size)
                for i in range(query_size):
                    if rank[i] != corpus_size:
                        results[level].append({
                            "repo": repo,
                            "parent commit": parent_commit,
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
