import numpy as np
import torch

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
            
    return results

def MAS(model, tokenizer, pronoun, candidate_a, candidate_b, sentence_a, sentence_b=None, layer=None, head=None):
    
    """
    Computes the Maximum Attention Score (MAS) given a sentence, a pronoun and candidates for substitution.
    Parameters
    ----------
    model : transformers.BertModel
        BERT model from BERT visualization that provides access to attention
    tokenizer:  transformers.tokenization.BertTokenizer
        BERT tolenizer
    pronoun: string
        pronoun to be replaced by a candidate
    candidate_a: string
        First pronoun replacement candidate
    candidate_b: string
        Second pronoun replacement candidate
    sentence_a: string
       First, or only sentence
    sentence_b: string (optional)
        Optional, second sentence
    layer: None, int
        If none, MAS will be computed over all layers, otherwise a specific layer
    head: None, int
        If none, MAS will be compputer over all attention heads, otherwise only at specific head
    Returns
    -------
    
    activity : list
        List of scores [score for candidate_a, score for candidate_b]
    """
    
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    
    candidate_a_ids = tokenizer.encode(candidate_a)[1:-1]
    candidate_b_ids = tokenizer.encode(candidate_b)[1:-1]
    pronoun_ids = tokenizer.encode(pronoun)[1:-1]
    
    if next(model.parameters()).is_cuda:
        attention = model(input_ids.cuda(), token_type_ids=token_type_ids.cuda())[-1]
    else:
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        
    attn = format_attention(attention)
    
    if next(model.parameters()).is_cuda:
        A = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
        B = torch.zeros((attn.shape[0], attn.shape[1])).cuda()
    else:
        A = torch.zeros((attn.shape[0], attn.shape[1]))
        B = torch.zeros((attn.shape[0], attn.shape[1]))
    
    if not layer is None:
        assert layer<attn.shape[0], "Maximum layer number "+str(attn.shape[0])+" exceeded"
        layer_slice = slice(layer,layer+1,1)
    else:
        layer_slice = slice(None,None,None)
        
    if not head is None:
        assert head<attn.shape[1], "Maximum head number "+str(attn.shape[1])+" exceeded"
        head_slice = slice(head,head+1,1)
    else:
        head_slice = slice(None,None,None)
    
    assert len(find_sub_list(pronoun_ids, input_ids[0].tolist())) > 0, "pronoun not found in sentence"
    assert len(find_sub_list(candidate_a_ids, input_ids[0].tolist())) > 0, "candidate_a not found in sentence"
    assert len(find_sub_list(candidate_b_ids, input_ids[0].tolist())) > 0, "candidate_b not found in sentence"
    
    for _,src in enumerate(find_sub_list(pronoun_ids, input_ids[0].tolist())):
    
    
        for _, tar_a in enumerate(find_sub_list(candidate_a_ids, input_ids[0].tolist())):
            A=A+attn[layer_slice,head_slice, slice(tar_a[0],tar_a[1]+1,1), slice(src[0],src[0]+1,1)].mean(axis=2).mean(axis=2)

        for _, tar_b in enumerate(find_sub_list(candidate_b_ids, input_ids[0].tolist())):
            B=B+attn[layer_slice,head_slice, slice(tar_b[0],tar_b[1]+1,1),slice(src[0],src[0]+1,1)].mean(axis=2).mean(axis=2)
    score = sum((A >= B).flatten()).item()/(A.shape[0]*A.shape[1])
    return [score, 1.0-score]