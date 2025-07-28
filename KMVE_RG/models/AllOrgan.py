import torch
import torch.nn as nn
import numpy as np

import sys

sys.path.append('../')
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder, DecoderOnly
from modules.encoder_decoder_rmsn import DecoderOnly as DecoderOnly_rmsn

from transformers import AutoModel, AutoTokenizer


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
        use_clip_loss = self.args.use_clip_loss if hasattr(self.args, 'use_clip_loss') else False
        if use_clip_loss:
            print("Using clip loss.")
        
        # 文本编码器（新增）
        self.text_encoder = TextEncoder(model_name='bert-base-chinese', output_dim=1024)

        print('vocabulary size:', self.tokenizer.get_vocab_size())
        self.classification_layers = classification()

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def clip_loss(self, image_embed, text_embed, temperature=0.07):
        # normalize
        image_embed = nn.functional.normalize(image_embed, dim=-1)
        text_embed = nn.functional.normalize(text_embed, dim=-1)

        # similarity matrix
        logits_per_image = image_embed @ text_embed.T  # (B, B)
        logits_per_text = text_embed @ image_embed.T  # (B, B)

        labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
        loss_i2t = nn.functional.cross_entropy(logits_per_image / temperature, labels)
        loss_t2i = nn.functional.cross_entropy(logits_per_text / temperature, labels)

        return (loss_i2t + loss_t2i) / 2


    def forward(self, images, targets=None, mode='train', text_ids=None, text_mask=None):
        att_feats_0, fc_feats_0, _, dense_vec1 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, _, dense_vec2 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        dense_vec = torch.cat((dense_vec1, dense_vec2), dim=1)
        if mode == 'train':
            output, _ = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')

            # 图像嵌入
            image_embed = dense_vec  # shape: (B, D)

            # 文本嵌入（新增）
            
            
            use_clip_loss = self.args.use_clip_loss if hasattr(self.args, 'use_clip_loss') else False
            if use_clip_loss:
            
                text_embed = self.text_encoder(text_ids, text_mask)  # shape: (B, D)
                loss_clip = self.clip_loss(image_embed, text_embed)
            else:
                loss_clip = 0
            # 分类任务
            classified = self.classification_layers(dense_vec)

            return output, classified, loss_clip
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            classified = self.classification_layers(dense_vec)
            return output, classified
        elif mode == 'evaluate':
            output, first_sentence, first_attmap, first_sentence_probs = \
                self.encoder_decoder(fc_feats, att_feats, mode='evaluate')
            return output, first_sentence, first_attmap, first_sentence_probs
        else:
            raise ValueError
        
class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-chinese', output_dim=4096):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(self.encoder.config.hidden_size, output_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.to(self.device)
    def forward(self, input_ids, attention_mask):
        device = self.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():  # 防止计算图累积
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_token = outputs.last_hidden_state[:, 0]  # 取 [CLS] 向量

        return self.proj(cls_token)  # proj 层可以训练

class classification(nn.Module):
    def __init__(self, organ_num=3, avg_dim=1024):
        super(classification, self).__init__()
        self.logit = nn.Linear(avg_dim, organ_num)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, avg):
        avg_visual = self.dropout(avg)
        x = self.logit(avg_visual)
        outputs = self.sigm(x)
        return outputs
