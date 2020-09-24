import json
from transformers import BertTokenizer, BertConfig, BertTokenizerFast

"""pretrained model of choice"""
# pretrained_model_name = 'SpanBERT/spanbert-base-cased'
pretrained_model_name = 'bert-base-cased'
# pretrained_model_name = 'bert-large-cased'
"""set up tokenizer"""
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
# tokenizer_max_len = 512


def bert_token_idx_remap(original_tokens,query_idx,tokenizer):
    bert_token_list = []
    for ot in original_tokens:
        bert_token_list.append(tokenizer(ot,return_tensors="pt")['input_ids'].tolist()[0][1:-1])
    print(bert_token_list)
    
    mapping = []
    bert_token_count = 1 # start with [CLS]
    for i in range(len(original_tokens)):
        mapped_length = len(bert_token_list[i])
        mapped_list = [j for j in range(bert_token_count,bert_token_count+mapped_length)]
        mapping.append(mapped_list)
        bert_token_count += mapped_length
    print(mapping)
    return mapping[query_idx]

# original_tokens =  [
#                 "Again",
#                 "there",
#                 "is",
#                 "no",
#                 "official",
#                 "written",
#                 "statement",
#                 "from",
#                 "Sistani",
#                 "'s",
#                 "office",
#                 "confirming",
#                 "this",
#                 "allegation",
#                 ",",
#                 "which",
#                 "I",
#                 "thinkkk",
#                 "is",
#                 "intentional",
#                 "."
#             ]
# bert_tokens = tokenizer(original_tokens,is_pretokenized = True, return_tensors = 'pt')
# token = tokenizer.convert_ids_to_tokens(bert_tokens['input_ids'].tolist()[0])
# print('bret tokens ids: ', bert_tokens['input_ids'])
# print('bret tokens: ', token)
# print('bret tokens: ', len(token))
# ret = bert_token_idx_remap(original_tokens,17,tokenizer)
# print(ret)
# for r in ret:
#     print(token[r])
