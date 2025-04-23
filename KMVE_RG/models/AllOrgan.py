import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('../')
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder, DecoderOnly
from modules.encoder_decoder_rmsn import DecoderOnly as DecoderOnly_rmsn



class AllOrgan(nn.Module):
    def __init__(self, args, tokenizer):
        super(AllOrgan, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        if self.args.decoderonly=="True" and self.args.norm == 'rmsnorm':
            print("using decoderonly rmsn model.")
            self.encoder_decoder = DecoderOnly_rmsn(args, tokenizer)
        elif self.args.decoderonly=="True":
            print("using decoderonly model.")
            self.encoder_decoder = DecoderOnly(args, tokenizer)
        else:
            self.encoder_decoder = EncoderDecoder(args, tokenizer)
        
        print('vocabulary size:', self.tokenizer.get_vocab_size())
        self.classfication_layers = classfication()

        self.saved_grads = []

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def dense_hook(self, grad):
        # print("Hook triggered, grad norm:", grad.norm())
        self.saved_grads.append(grad.detach().cpu())

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0, _, dense_vec1 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, _, dense_vec2 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        dense_vec = torch.cat((dense_vec1, dense_vec2), dim=1)
        if att_feats.requires_grad:
            att_feats.register_hook(self.dense_hook)
        if mode == 'train':
            # print(f"train mode, input shape: {fc_feats.shape}, {att_feats.shape}, {targets.shape}")
            # print(f"fc_feats dtype: {fc_feats.dtype}, att_feats dtype: {att_feats.dtype}, targets dtype: {targets.dtype}")
            output, _ = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            classified = self.classfication_layers(dense_vec)
            return output, classified
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            # print(f"sample mode, input shape: {fc_feats.shape}, {att_feats.shape}")
            classified = self.classfication_layers(dense_vec)
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
        outputs = self.sigm(x)
        return outputs
