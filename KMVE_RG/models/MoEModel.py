import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('../')
from modules.visual_extractor import VisualExtractor
from modules.moe_decoder import MoEDecoderOnly

import torch.nn.functional as F

class MoEModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(MoEModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        
        self.encoder_decoder = MoEDecoderOnly(args, tokenizer)
        
        print('vocabulary size:', self.tokenizer.get_vocab_size())
        self.classfication_layers = classfication()
        self.routes = None
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0, _, dense_vec1 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, _, dense_vec2 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        dense_vec = torch.cat((dense_vec1, dense_vec2), dim=1)
        classified = self.classfication_layers(dense_vec)
        self.routes = F.softmax(classified, dim=1)
        self.encoder_decoder.routes = self.routes
        if mode == 'train':
            # print(f"train mode, input shape: {fc_feats.shape}, {att_feats.shape}, {targets.shape}")
            # print(f"fc_feats dtype: {fc_feats.dtype}, att_feats dtype: {att_feats.dtype}, targets dtype: {targets.dtype}")
            output, _ = self.encoder_decoder(fc_feats, att_feats, targets, routes=self.routes, mode='forward')
            
            return output, classified
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            # print(f"sample mode, input shape: {fc_feats.shape}, {att_feats.shape}")
            return output, classified
        elif mode == 'evaluate':
            output, first_sentence, first_attmap, first_sentence_probs = \
                self.encoder_decoder(fc_feats, att_feats, mode='evaluate')
            return output, first_sentence, first_attmap, first_sentence_probs
        else:
            raise ValueError
        


class classfication(nn.Module):
    def __init__(self, organ_num=3, avg_dim=1024):
        super(classfication, self).__init__()
        self.logit = nn.Linear(avg_dim, organ_num)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, avg):
        avg_visual = self.dropout(avg)
        x = self.logit(avg_visual)
        # outputs = self.sigm(x) # 每个样本可以属于多个类别
        # outputs = nn.Softmax(dim=1)(x)
        outputs = x 
        return outputs
