import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('../')
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from torch.autograd import Variable
from modules.new_model_utils import SemanticEmbedding, classfication


class AllOrgan(nn.Module):
    def __init__(self, args, tokenizer):
        super(AllOrgan, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        print('vocabulary size:', self.tokenizer.get_vocab_size())

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0, _, _ = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, _, _ = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        
        if mode == 'train':
            output, _ = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'evaluate':
            output, first_sentence, first_attmap, first_sentence_probs = \
                self.encoder_decoder(fc_feats, att_feats, mode='evaluate')
            return output, first_sentence, first_attmap, first_sentence_probs
        else:
            raise ValueError
        return output