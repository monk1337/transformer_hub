import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import logging, sys
logging.disable(sys.maxsize)

import paddle.fluid.dygraph as D
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel
from transformers import TransfoXLTokenizer, TransfoXLModel
from transformers import XLMTokenizer, XLMModel
from transformers import ElectraTokenizer, ElectraModel
from transformers import BertModel, BertConfig, BertTokenizer


from transformers import BartModel,BartTokenizer
from transformers import T5Tokenizer, T5Model
from transformers import AlbertTokenizer, AlbertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import GPT2Tokenizer, GPT2Model
from collections import namedtuple


import torch

# There is some clarification about the use of the last hidden states in the BERT Paper.
# According to the paper, the last hidden state for [CLS] is mainly used for classification tasks 
# and the last hidden states for all tokens are used for token level tasks such as sequence tagging or 
# question answering.

# From the paper:
# At the output, the token representations are fed into an output layer for token level tasks, 
# such as sequence tagging or question answering, and the [CLS] representation is fed into an output 
# layer for classification, such as entailment or sentiment analysis.

# Reference:
# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (https://arxiv.org/pdf/1810.04805.pdf)



# last hidden dim shape [1, 1024]


# todo 
# taking last vectors mean
# taking last 4 hidden dim and then mean
# https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb

# make it batch
# save it as dict



trained_models = {'bert'      : 'bert-large-uncased', 
                  'xlnet'     : 'xlnet-large-cased', 
                  'gpt'       : 'gpt2-medium', 
                  'roberta'   : 'roberta-large-openai-detector', 
                  'albert'    : 'albert-large-v2', 
                  't5'        : 't5-large', 
                  'bart'      : 'facebook/bart-large', 
                  'electra'   : 'google/electra-large-discriminator', 
                  'xlm'       : 'xlm-mlm-en-2048', 
                  'xl'        : 'transfo-xl-wt103', 
                  'ernie'     : '..'}




model_config = namedtuple('model_config', ['tok', 'model'])

bert_tuple         = (BertTokenizer.from_pretrained, BertModel.from_pretrained)
xlnet_tuple        = (XLNetTokenizer.from_pretrained, XLNetModel.from_pretrained)
gpt_2_tuple        = (GPT2Tokenizer.from_pretrained, GPT2Model.from_pretrained)
robert_tuple       = (RobertaTokenizer.from_pretrained, RobertaModel.from_pretrained)
albert_tuple       = (AlbertTokenizer.from_pretrained, AlbertModel.from_pretrained)
t5_tuple           = (T5Tokenizer.from_pretrained, T5Model.from_pretrained)
bart_tuple         = (BartTokenizer.from_pretrained, BartModel.from_pretrained)
electra_tuple      = (ElectraTokenizer.from_pretrained, ElectraModel.from_pretrained)
xlm_tuple          = (XLMTokenizer.from_pretrained, XLMModel.from_pretrained)
xl_tuple           = (TransfoXLTokenizer.from_pretrained,TransfoXLModel.from_pretrained)
erine_tuple        = (ErnieTokenizer.from_pretrained, ErnieModel.from_pretrained)


model_resources = {'bert' : model_config._make(bert_tuple), 'xlnet' : model_config._make(xlnet_tuple), 
                   'gpt': model_config._make(gpt_2_tuple), 'roberta': model_config._make(robert_tuple), 
                   'albert': model_config._make(albert_tuple), 't5' : model_config._make(t5_tuple), 
                   'bart': model_config._make(bart_tuple), 'electra' : model_config._make(electra_tuple), 
                   'xlm': model_config._make(xlm_tuple), 'xl' : model_config._make(xl_tuple), 
                   'ernie': model_config._make(erine_tuple)}





# https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over/62723657#62723657



def load_models(list_of_models, config):
    
    if list_of_models == 'all':
        
        model_list = ['bert', 'gpt', 'xlnet', 'roberta', 
                      'albert', 't5', 'bart', 'electra', 'xlm', 'xl', 'ernie']
    else:
        model_list = list_of_models
    
    load_ = {}

    for model in model_list:
        if model in model_resources:
            load_[model] = model_resources[model]
    
    return load_
                   

# model_list = ['bert', 'gpt', 'xlnet', 'robert', 'albert', 't5', 'bart', 'electra', 'xlm', 'xl', 'erine']


def get_embedding_vector(tokenizer, model, config, sentence):
    
    
    
    inputs = tokenizer(sentence, return_tensors="pt")
    #     input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    #     outputs   = model(input_ids)
    outputs = model(**inputs)
    if config['output_vector'] == 'cls':
        features  = outputs[0][:,0,:].detach().numpy().squeeze()
    else:  # mean
        features  = torch.mean(outputs[0][-1], dim=0).squeeze()
    return features


def get_ernie_vector(tokenizer, model, config, sentence):
    
    
    model.eval()
    ids, _ = tokenizer.encode(sentence)
    ids = D.to_variable(np.expand_dims(ids, 0))  # insert extra `batch` dimension
    pooled, encoded = model(ids)                 # eager execution
    
    return pooled.numpy()


def get_t5_vector(tokenizer, model, config, sentence):
    
    
    input_ids   = tokenizer(sentence, return_tensors="pt")["input_ids"]
    outputs     = model(input_ids=input_ids, decoder_input_ids=input_ids)
    
    if config['output_vector'] == 'cls':
        features  = outputs[0][:,0,:].detach().numpy().squeeze()
    else:  # mean
        features  = torch.mean(outputs[0][-1], dim=0).squeeze()
    return features


def get_embeddings(sentence_input, model_list, config):
    
    models_list = load_models(model_list, config)
    
    all_embeddings = {}
    
    for model_name, model_data in tqdm(models_list.items()):
        
        if model_name in model_resources:
            
            print(f"Loading {model_name} model...")
            tokenizer = model_data.tok(trained_models[model_name])
            D.guard().__enter__()
            model     = model_data.model(trained_models[model_name])
            
            # load model session once and encode all sentences to save time instead of loading each time

            
            print(f"{model_name} Loaded..\n Encoding sentences from {model_name} model...")

            
            if model_name == 'ernie':
                
                if isinstance(sentence_input, list):
                    print("encoding in batch")
                    all_sentences = []
                    for sentence in sentence_input:
                        all_sentences.append(get_ernie_vector(tokenizer = tokenizer, 
                                                              model     = model, 
                                                              config    = config,
                                                              sentence  = sentence))
                    all_embeddings[model_name] = all_sentences
                else:
                    all_embeddings[model_name] = get_ernie_vector(tokenizer = tokenizer, 
                                                                  model     = model, 
                                                                  config    = config,
                                                                  sentence  = sentence_input)
                
                
            elif model_name == 't5':
                
                if isinstance(sentence_input, list):
                    print("encoding in batch")
                    all_sentences = []
                    for sentence in sentence_input:
                        all_sentences.append(get_t5_vector(tokenizer = tokenizer, 
                                                              model     = model, 
                                                              config    = config,
                                                              sentence  = sentence))
                        
                        
                
                
                    
                    all_embeddings[model_name] = all_sentences
                else:
                    all_embeddings[model_name] = get_t5_vector(tokenizer = tokenizer, 
                                                                  model     = model, 
                                                                  config    = config,
                                                                  sentence  = sentence_input)
                    
            else:
                
                if isinstance(sentence_input, list):
                    print("encoding in batch")
                    all_sentences = []
                    for sentence in sentence_input:
                        all_sentences.append(get_embedding_vector(tokenizer = tokenizer, 
                                                                  model     = model, 
                                                                  config    = config,
                                                                  sentence  = sentence))
                    all_embeddings[model_name] = all_sentences


                else:
                    all_embeddings[model_name] = get_embedding_vector(tokenizer = tokenizer, 
                                                                  model     = model, 
                                                                  config    = config,
                                                                  sentence  = sentence_input)
                
                
    return all_embeddings


