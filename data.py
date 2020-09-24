from transformers import BertTokenizer
import torch 
import json
import re
import collections

def prepare_data_matres_tcr(dataset_list, tokenizer, label_dict,tokenizer_max_len):
    input_ids = []
    attention_masks = []
    triggers = {}
    tokens = {}
    labels = []
    idxs = []
    for i in range(len(dataset_list)):
        item = dataset_list[i]
        # inputs = tokenizer(item['input_sequence'],return_tensors = 'pt', max_length = tokenizer_max_len, padding= 'max_length')
        inputs = tokenizer(item['input_tokens'],is_pretokenized = True, return_tensors = 'pt', max_length = tokenizer_max_len, padding= 'max_length')
        # token = tokenizer.convert_ids_to_tokens(inputs['input_ids'].tolist()[0])

        trigger = item['trigger']
        label = label_dict[item['label']]
        
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        triggers[i] = trigger
        tokens[i] = item['input_tokens']
        labels.append(label)
        idxs.append(i)
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    idxs = torch.tensor(idxs)
    print('=============================================')
    print('inputs:')
    print(input_ids[0])
    print(attention_masks[0])
    print(triggers[0])
    print(tokens[0])
    print(labels[0])
    print(idxs[0])
    print('=============================================')

    return input_ids,attention_masks,triggers,tokens,labels,idxs


def prepare_data_UDS_T(dataset_list, tokenizer, label_dict,tokenizer_max_len):
    input_ids = []
    attention_masks = []
    triggers = {}
    tokens = {}
    labels = []
    idxs = []
    for i in range(len(dataset_list)):
        item = dataset_list[i]
        # inputs = tokenizer(item['input_sequence'],return_tensors = 'pt', max_length = tokenizer_max_len, padding= 'max_length')
        input_tokens = item['input_tokens'][0] + [tokenizer.sep_token] +  item['input_tokens'][1] 
        # print(input_tokens)
        inputs = tokenizer(input_tokens,is_pretokenized = True, return_tensors = 'pt', max_length = tokenizer_max_len, padding= 'max_length')
        # token = tokenizer.convert_ids_to_tokens(inputs['input_ids'].tolist()[0])
        # print(token)
        # quit()
        
        trigger = item['trigger'].copy()
        
        # print(trigger[1], item['input_tokens'][1][trigger[1][0]])
        trigger[1][0] += (len(item['input_tokens'][0])+1)
        # print(trigger[1], input_tokens[trigger[1][0]])
        label = label_dict[item['label']]        
        
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        triggers[i] = trigger
        tokens[i] = input_tokens
        labels.append(label)
        idxs.append(i)
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    idxs = torch.tensor(idxs)
    print('=============================================')
    print('inputs:')
    print(input_ids[0])
    print(attention_masks[0])
    print(triggers[0])
    print(tokens[0])
    print(labels[0])
    print(idxs[0])
    print('=============================================')

    return input_ids,attention_masks,triggers,tokens,labels,idxs