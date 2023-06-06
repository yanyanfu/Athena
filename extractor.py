import torch
import numpy as np

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset
from parser import DFG_java
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
from model import CodeBERTModel, GraphCodeBERTModel, UniXcoderModel


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

    
class TextDataset(Dataset):
    def __init__(self, examples, code_length, data_flow_length):
        self.examples = examples 
        self.code_length = code_length
        self.data_flow_length = data_flow_length    
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.code_length+self.data_flow_length, self.code_length+self.data_flow_length),dtype=np.bool)
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


class Embed(ABC):
    def __init__(self, model_name, finetuned_model_path):
        super().__init__()
        LANGUAGE = Language('parser/my-languages.so', 'java')     
        parser = Parser()
        parser.set_language(LANGUAGE) 
        self.parser = [parser,DFG_java]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = RobertaConfig.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)       

        self.code_length = 256
        self.data_flow_length = 64
        self.batch_size = 128
        self.finetuned_model_path = finetuned_model_path

    @abstractmethod
    def load_pretrained_model(self):
        pass

    @abstractmethod
    def load_finetuned_model(self):
        pass

    @abstractmethod
    def preprocess(self, code):
        pass

    @abstractmethod
    def convert_examples_to_features(self, code):
        pass

    @abstractmethod
    def extract_corpus_vecs(self):
        pass


class EmbedCodebert(Embed):

    def load_pretrained_model(self):
        self.model = CodeBERTModel(self.model)
        self.model.to(self.device)

    def load_finetuned_model(self):
        self.load_pretrained_model()
        self.model.load_state_dict(torch.load(self.finetuned_model_path),strict=False)

    def preprocess(self, code):
        code=remove_comments_and_docstrings(code,'java')  
        tree = self.parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        code_tokens = [x for x in code_tokens if x]
        return code_tokens
    
    def convert_examples_to_features(self, code):
        code_tokens = self.preprocess(code)
        code_tokens = self.tokenizer.tokenize(' '.join(code_tokens))
        code_tokens =[self.tokenizer.cls_token]+code_tokens[:self.code_length-2]+[self.tokenizer.sep_token]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = self.code_length - len(code_ids)
        code_ids+=[self.tokenizer.pad_token_id]*padding_length
        return code_ids

    def extract_corpus_vecs(self, methods):
        corpus_inputs = []
        for method in methods:
            corpus_inputs.append(self.convert_examples_to_features(method))
        corpus_dataset = TensorDataset(torch.tensor(corpus_inputs))
        corpus_sampler = SequentialSampler(corpus_dataset)
        corpus_dataloader = DataLoader(corpus_dataset, sampler=corpus_sampler, batch_size=self.batch_size,num_workers=4)
        self.model.eval()
        query_vecs=[] 
        corpus_vecs=[]
        for batch in corpus_dataloader:
            corpus_inputs = batch[0].to(self.device)   
            with torch.no_grad():
                corpus_vec= self.model(corpus_inputs)
                corpus_vecs.append(corpus_vec.cpu().numpy()) 
        self.model.train()  
        corpus_vecs=np.concatenate(corpus_vecs,0)
        return corpus_vecs


class EmbedGraphcodebert(Embed):

    def load_pretrained_model(self):
        self.model = GraphCodeBERTModel(self.model)
        self.model.to(self.device)

    def load_finetuned_model(self):
        self.load_pretrained_model()
        self.model.load_state_dict(torch.load(self.finetuned_model_path),strict=False)
 
    def preprocess(self, code):
        try:
            code=remove_comments_and_docstrings(code,'java')   
            tree = self.parser[0].parse(bytes(code,'utf8'))    
            root_node = tree.root_node  
            tokens_index=tree_to_token_index(root_node)     
            code=code.split('\n')
            code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
            index_to_code={}
            for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
                index_to_code[index]=(idx,code)  
            try:
                DFG,_=self.parser[1](root_node,index_to_code,{}) 
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

    def convert_examples_to_features(self, code):
        #extract data flow 
        code_tokens,dfg=self.preprocess(code)       
        code_tokens=[self.tokenizer.tokenize('@ '+x)[1:] if idx!=0 else self.tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        #truncating
        code_tokens=code_tokens[:self.code_length+self.data_flow_length-2-min(len(dfg),self.data_flow_length)]
        code_tokens =[self.tokenizer.cls_token]+code_tokens+[self.tokenizer.sep_token]
        code_ids =  self.tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+self.tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg=dfg[:self.code_length+self.data_flow_length-len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        code_ids+=[self.tokenizer.unk_token_id for x in dfg]
        padding_length=self.code_length+self.data_flow_length-len(code_ids)
        position_idx+=[self.tokenizer.pad_token_id]*padding_length
        code_ids+=[self.tokenizer.pad_token_id]*padding_length    
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([self.tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]          
        
        return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg)
    
    def extract_corpus_vecs(self, methods):
        examples = []
        for method in methods:
            examples.append(self.convert_examples_to_features(method)) 
        corpus_dataset=TextDataset(examples, self.code_length, self.data_flow_length)
        corpus_sampler = SequentialSampler(corpus_dataset)
        corpus_dataloader = DataLoader(corpus_dataset, sampler=corpus_sampler, batch_size=self.batch_size,num_workers=4)

        self.model.eval() 
        corpus_vecs=[]
        for batch in corpus_dataloader:
            corpus_inputs = batch[0].to(self.device)
            attn_mask = batch[1].to(self.device)
            position_idx =batch[2].to(self.device)   
            with torch.no_grad():
                corpus_vec= self.model(code_inputs=corpus_inputs, attn_mask=attn_mask,position_idx=position_idx)
                corpus_vecs.append(corpus_vec.cpu().numpy()) 
        self.model.train() 
        corpus_vecs=np.concatenate(corpus_vecs,0)
        return corpus_vecs


class EmbedUnixcoder(Embed):

    def load_pretrained_model(self):
        self.model = UniXcoderModel(self.model)
        self.model.to(self.device)

    def load_finetuned_model(self):
        self.load_pretrained_model()
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model 
        model_to_load.load_state_dict(torch.load(self.finetuned_model_path))

    def preprocess(self, code):
        code=remove_comments_and_docstrings(code,'java')  
        tree = self.parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        code_tokens = [x for x in code_tokens if x]
        return code_tokens
    
    def convert_examples_to_features(self, code):
        code_tokens = self.preprocess(code)
        code_tokens = self.tokenizer.tokenize(' '.join(code_tokens))
        code_tokens =[self.tokenizer.cls_token,"<encoder-only>",self.tokenizer.sep_token]+code_tokens[:self.code_length-4]+[self.tokenizer.sep_token]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = self.code_length - len(code_ids)
        code_ids+=[self.tokenizer.pad_token_id]*padding_length
        return code_ids
    

    def extract_corpus_vecs(self, methods):
        corpus_inputs = []
        for method in methods:
            corpus_inputs.append(self.convert_examples_to_features(method))
        corpus_dataset = TensorDataset(torch.tensor(corpus_inputs))
        corpus_sampler = SequentialSampler(corpus_dataset)
        corpus_dataloader = DataLoader(corpus_dataset, sampler=corpus_sampler, batch_size=self.batch_size,num_workers=4)
        self.model.eval()
        query_vecs=[] 
        corpus_vecs=[]
        for batch in corpus_dataloader:
            corpus_inputs = batch[0].to(self.device)   
            with torch.no_grad():
                corpus_vec= self.model(corpus_inputs)
                corpus_vecs.append(corpus_vec.cpu().numpy()) 
        self.model.train()  
        corpus_vecs=np.concatenate(corpus_vecs,0)

        return corpus_vecs
        


