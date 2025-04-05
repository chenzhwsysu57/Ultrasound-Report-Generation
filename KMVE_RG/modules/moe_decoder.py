from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Generator import pack_wrapper, GenModel

def hyperbolic_distance(x, y, alpha=1.0):
    """计算双曲空间中的 arcosh 距离"""
    euclidean_dist = torch.norm(x - y, dim=-1, p=2) ** 2  # ||x - y||^2
    return torch.acosh(1 + alpha * euclidean_dist + 1e-6)  # 避免数值问题

from functools import wraps
import time
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()  # 记录结束时间
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask,routes=None):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask,routes)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask,routes=None):
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, routes)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



class RMSNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))  # 可学习的缩放参数
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, routes):
        total_loss = 0
        for layer in self.layers:
            x, loss = layer(x, hidden_states, src_mask, tgt_mask, routes)
            total_loss += loss
        return self.norm(x), total_loss


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model  # 512
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.norm3 = LayerNorm(self.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, routes):
        m = hidden_states

        x = x + self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x)

        x = x + self.src_attn(x, m, m, src_mask)
        x = self.norm2(x)

        ffn_out, loss = self.feed_forward(x, routes)  
        x = x + ffn_out  
        x = self.norm3(x)

        return x, loss  


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class IdentityEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(IdentityEncoderLayer, self).__init__()

    def forward(self, x, mask):
        return x
    
class IdentityEncoder(nn.Module):
    def __init__(self, layer, N):
        super(IdentityEncoder, self).__init__()

    def forward(self, x, mask):
        return x
    
class MixtureOfExpertsFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, num_experts=6, num_shared_experts=2):
        super(MixtureOfExpertsFFN, self).__init__()
        # 是否 d_ff = d_ff / num_experts ?
        # self.shared_expert = nn.Linear(d_model, d_ff) 
        self.shared_experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff) for _ in range(num_shared_experts)
        ])
        # 三个独立的专家
        self.experts = nn.ModuleList([
            PositionwiseFeedForward(d_model, d_ff) for _ in range(num_experts)
        ])
        
    # @timing_decorator
    def forward(self, x, routes):
        # routing_weights = routes
        max_indices = routes.argmax(dim=1)
        binary_routes = torch.zeros_like(routes)
        binary_routes[torch.arange(routes.size(0)), max_indices] = 1
        binary_routes
        """
        x: 输入特征，形状 [batch_size, seq_len, d_model]
        routing_weights: 路由权重，形状 [batch_size, 3]
        按照 weights 给batch内的每个样本为三个路由配置权重。例如第 i 个样本的 weights 是 tensor([0.9, 0.1, 0.1])，那么有0.9的输出来自第一个专家，0.1的输出来自第二个专家，0.1的输出来自第三个专家
        """
        # TODO
        # batch_size, seq_len, _ = x.shape  
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [batch, seq_len, d_ff, num_experts]

        num_experts = len(self.experts)
        repeat_factor = num_experts // 3
        
        routing_weights = binary_routes.unsqueeze(1).unsqueeze(2).expand(-1, 1, 1, num_experts)

        # 按照 routing_weights 计算专家输出的加权和
        routed_out = torch.sum(expert_outputs * routing_weights, dim=-1)  # [batch, seq_len, d_ff]

        # 共享专家计算
        shared_out = sum(expert(x) for expert in self.shared_experts)
        # 组合输出（共享专家 + MoE 输出）
        output = shared_out + routed_out  # [batch, seq_len, d_model]

        # TODO 重写 loss。 loss 包含三部分组成

        ###################
        ### Shared loss ###
        ###################
        mean_tensor = torch.mean(shared_out, dim=1)  # [12, 512]
        # 计算 L2 距离
        differences = mean_tensor.unsqueeze(1) - mean_tensor.unsqueeze(0)
        euclidean_distances = torch.norm(differences, dim=-1)

        alpha = 0.5  # 示例值
        transformed_distances = torch.acosh(1 + alpha * euclidean_distances + 1e-6)

        # 假设 transformed_distances 是损失的一部分
        loss1_argmin  = transformed_distances.sum()  # 示例损失计算

        ##############################
        ### cross share-route loss ###
        ##############################
        loss2_argmax = 0.0
        groups = [torch.nonzero(binary_routes[:, i]).squeeze() for i in range(3)]
        groups = [group if group.dim() > 0 else group.unsqueeze(0) for group in groups]

        
        for group in groups:
            # 提取每组的数据
            shared_group = shared_out[group]
            routed_group = routed_out[group]
            
            # 计算均值
            shared_mean = torch.mean(shared_group, dim=1)  # [4, 512]
            routed_mean = torch.mean(routed_group, dim=1)  # [4, 512]
            
            # 计算 L2 距离
            differences = shared_mean.unsqueeze(1) - routed_mean.unsqueeze(0)
            euclidean_distances = torch.norm(differences, dim=-1)  # [4, 4]
            
            # 应用 torch.acosh
            transformed_distances = torch.acosh(1 + alpha * euclidean_distances + 1e-6)
            
            # 计算组损失
            group_loss = transformed_distances.sum()
            loss2_argmax += group_loss

        ##############################
        ### cross share loss ###
        ##############################
        # TODO 针对这三组routed_group，每个group按照sample取均值得到shape为115,1024的向量三个；然后求他们的距离作为 loss3_argmax
        loss3_argmax = 0.0
        routed_means = [torch.mean(routed_out[group], dim=0) for group in groups]
        routed_means_tensor = torch.stack(routed_means)
        differences = routed_means_tensor.unsqueeze(1) - routed_means_tensor.unsqueeze(0)
        euclidean_distances = torch.norm(differences, dim=-1)  # [3, 3]
        transformed_distances = torch.acosh(1 + alpha * euclidean_distances + 1e-6)
        loss3_argmax = transformed_distances.sum()
        expert_loss = (loss1_argmin - loss2_argmax - loss3_argmax )/1500 + 10
        # print("loss1_argmin: ", loss1_argmin)
        # print("loss2_argmax: ", loss2_argmax)
        # print("loss3_argmax: ", loss3_argmax)
        # print("total expert_loss: ", expert_loss)
        
        return output, expert_loss

        
class MoEDecoderOnly(GenModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        # ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        ff = MixtureOfExpertsFFN(self.d_model, self.d_ff, self.dropout, num_experts=self.num_experts, num_shared_experts=self.num_shared_experts)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            IdentityEncoder(IdentityEncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)))
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(MoEDecoderOnly, self).__init__(args, tokenizer)
        self.args = args
        self.num_experts = int(args.num_experts)
        self.num_shared_experts = int(args.num_shared_experts)
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = int(args.d_ff)
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)
        self.routes = None
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, routes=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out, loss = self.model(att_feats, seq, att_masks, seq_mask, routes=routes)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs, out, loss  

    def core(self, it, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, _ = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), routes=self.routes)
        return out[:, -1], [ys.unsqueeze(0)]
