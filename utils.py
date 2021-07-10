import transformers
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.autograd import Function
from transformers import AutoTokenizer

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None



class MainModule_ner(nn.Module):
    def __init__(self, input_encoder_model, input_encoder_config, ner_tags, pos_tags):
        super(MainModule_ner, self).__init__()
        self.input_encoder_config = input_encoder_config
        self.input_encoder_model = input_encoder_model

        self.dropout = nn.Dropout(0.25)
        self.ner_lin = nn.Linear(self.input_encoder_config.hidden_size, ner_tags).to(device)
        self.pos_lin = nn.Linear(self.input_encoder_config.hidden_size, pos_tags).to(device)

        self.cls_disc = nn.Linear(self.input_encoder_config.hidden_size, 2) # num clusters
        self.act = nn.ReLU()
        self.active_grl = True
        self.grl_alpha = 0.9


    def forward(self ,input_ids=None ,attention_mask=None ,token_type_ids=None):

        input_encoded = self.input_encoder_model(input_ids ,attention_mask ,token_type_ids)
        ###################################
        input_embedding = input_encoded[1]
        tokens_embedding = input_encoded[0]
        ###################################
        # mask_to_mul = attention_mask
        # mask_len = torch.sum(mask_to_mul,1).view(-1,1)
        # mask_to_mul_size = list(mask_to_mul.size()) + [1]
        # mask_to_mul = mask_to_mul.view(mask_to_mul_size)
        # mask_to_mul = mask_to_mul.repeat((1,1,input_encoded[0].size(2)))
        # input_embedding = input_encoded[0]*mask_to_mul
        # input_embedding = torch.sum(input_embedding,(1)) / mask_len
        ###################################


        ########## Disc:
        if self.active_grl:
            input_embedding_reverse = ReverseLayerF.apply(input_embedding, self.grl_alpha)
        else:
            input_embedding_reverse = input_embedding

        logits_disc = self.cls_disc(input_embedding_reverse)


        ########## NER:

        tokens_embedding_droped = self.dropout(tokens_embedding)
        logits_NER = self.ner_lin(tokens_embedding_droped)

        ########## POS:

        tokens_embedding_droped = self.dropout(tokens_embedding)
        logits_POS = self.pos_lin(tokens_embedding_droped)


        return_dict = {"logits_ner" :logits_NER ,
                       "logits_disc" :logits_disc ,
                       "logits_pos" :logits_POS ,
                       "input_encoded" :input_encoded ,
                       }
        return return_dict


class MainModule(nn.Module):
    def __init__(self, input_encoder_model, input_encoder_config, n_tags):
        super(MainModule, self).__init__()
        self.input_encoder_config = input_encoder_config
        self.input_encoder_model = input_encoder_model

        self.dropout = nn.Dropout(0.25)
        self.lin1 = nn.Linear(self.input_encoder_config.hidden_size, 2).to(device)
        self.lin2 = nn.Linear(self.input_encoder_config.hidden_size, n_tags).to(device)

        self.cls_disc = nn.Linear(self.input_encoder_config.hidden_size, 2) # num clusters
        self.act = nn.ReLU()
        self.active_grl = True
        self.grl_alpha = 0.9


    def forward(self ,input_ids=None ,attention_mask=None ,token_type_ids=None):

        input_encoded = self.input_encoder_model(input_ids ,attention_mask ,token_type_ids)
        ###################################
        input_embedding = input_encoded[1]
        tokens_embedding = input_encoded[0]
        ###################################
        # mask_to_mul = attention_mask
        # mask_len = torch.sum(mask_to_mul,1).view(-1,1)
        # mask_to_mul_size = list(mask_to_mul.size()) + [1]
        # mask_to_mul = mask_to_mul.view(mask_to_mul_size)
        # mask_to_mul = mask_to_mul.repeat((1,1,input_encoded[0].size(2)))
        # input_embedding = input_encoded[0]*mask_to_mul
        # input_embedding = torch.sum(input_embedding,(1)) / mask_len
        ###################################


        ########## Disc:
        if self.active_grl:
            input_embedding_reverse = ReverseLayerF.apply(input_embedding, self.grl_alpha)
        else:
            input_embedding_reverse = input_embedding

        logits_disc = self.cls_disc(input_embedding_reverse)


        ########## Sentiment:

        input_embedding_droped = self.dropout(input_embedding)
        logits_similarity = self.lin1(input_embedding_droped)


        ########## POS:

        tokens_embedding_droped = self.dropout(tokens_embedding)
        logits_POS = self.lin2(tokens_embedding_droped)



        return_dict = {"logits_similarity" :logits_similarity ,
                       "logits_disc" :logits_disc ,
                       "logits_pos" :logits_POS ,
                       "input_encoded" :input_encoded ,
                       }
        return return_dict


