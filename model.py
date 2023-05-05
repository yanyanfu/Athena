# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# representations are extracted based on https://github.com/microsoft/CodeBERT

import torch.nn as nn
import torch
    
class CodeBERTModel(nn.Module):   
    def __init__(self, encoder):
        super(CodeBERTModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            return self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0][:,0]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0][:,0]


class GraphCodeBERTModel(nn.Module):   
    def __init__(self, encoder):
        super(GraphCodeBERTModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0][:,0]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]


class UniXcoderModel(nn.Module):   
    def __init__(self, encoder):
        super(UniXcoderModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        
 
