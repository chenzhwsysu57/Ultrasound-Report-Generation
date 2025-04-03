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

# def hyperbolic_distance(x, y, alpha=1.0):
#     """优化后的双曲距离计算，避免使用 torch.acosh"""
#     euclidean_dist = torch.norm(x - y, dim=-1, p=2)  # ||x - y||
#     return torch.sqrt(2 * alpha * euclidean_dist + 1e-6)  # 近似替代 arcosh
# def arcosh(x, eps=1e-6):
#     return torch.log(x + torch.sqrt(x**2 - 1 + eps))

# def compute_expert_loss(shared_out, expert_outputs, routes, alpha=1.0):
#     """
#     shared_out: 共享专家的输出, 形状 [batch, seq_len, d_ff]
#     expert_outputs: 所有专家的输出, 形状 [batch, seq_len, d_ff, num_experts]
#     routes: 路由选择权重, 形状 [batch, num_experts]
#     alpha: 超参数，控制 arcosh 距离的尺度
#     """

#     # 确保 routes 形状匹配 expert_outputs
#     routes_expanded = routes.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_experts]

#     # 按照路由权重计算专家输出的加权和
#     expert_selected_outputs = torch.sum(expert_outputs * routes_expanded, dim=-1)  # [batch, seq_len, d_ff]

#     ## 1. 通用特征约束（希望共享专家的特征在不同类别之间接近） min L_shared
#     shared_loss = arcosh(1 + alpha * torch.norm(shared_out[:, None, :, :] - shared_out[:, :, None, :], dim=-1)).mean()

#     ## 2. 专家与共享专家不同（希望专家与共享专家学到的特征有差异） max L_expert_shared
#     expert_shared_loss = -arcosh(1 + alpha * torch.norm(expert_selected_outputs - shared_out, dim=-1)).mean()

#     ## 3. 专家之间不同（希望不同专家的特征不相似） max L_expert_diversity
#     expert_diversity_loss = -arcosh(
#         1 + alpha * torch.norm(expert_selected_outputs[:, None, :, :] - expert_selected_outputs[:, :, None, :], dim=-1)
#     ).mean()

#     return shared_loss, expert_shared_loss, expert_diversity_loss
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

@timing_decorator
def compute_expert_loss(shared_outputs, expert_outputs, routes, alpha=1.0):
    """
    计算三种损失
    shared_outputs: 共享专家的输出 [batch, seq_len, d_ff]
    expert_outputs: 所有专家的输出 [batch, seq_len, d_ff, num_experts]
    routes: 样本的路由分配 [batch, num_experts]
    alpha: 距离度量的缩放因子
    """

    batch_size, seq_len, d_ff, num_experts = expert_outputs.shape
    device = shared_outputs.device

    # 1. 通用特征约束：不同类别的共享特征应尽可能相似
    shared_features = shared_outputs.mean(dim=1)  # [batch, d_ff]
    # shared_dist = []
    # for i in range(batch_size):
    #     for j in range(batch_size):
    #         if i != j:
    #             shared_dist.append(hyperbolic_distance(shared_features[i], shared_features[j], alpha))
    # shared_loss = torch.stack(shared_dist).mean()
    shared_dist_matrix = hyperbolic_distance(
        shared_features.unsqueeze(1),  # [batch, 1, d_ff]
        shared_features.unsqueeze(0),  # [1, batch, d_ff]
        alpha
    )  # 得到 [batch, batch] 的距离矩阵
    shared_loss = shared_dist_matrix.sum() / (batch_size * (batch_size - 1))  # 排除自身

    # 2. 各路由专家与通用专家不同
    routes_expanded = routes.unsqueeze(1).unsqueeze(2)
    expert_selected_outputs = torch.sum(expert_outputs * routes_expanded, dim=-1)  # [batch, seq_len, d_ff]
    expert_mean = expert_selected_outputs.mean(dim=1)  # [batch, d_ff]
    expert_shared_dist = hyperbolic_distance(shared_features, expert_mean, alpha)
    expert_shared_loss = -expert_shared_dist.mean()  # 负号表示最大化

    # 3. 专家之间不同（不同类别专家应有区别）
    # expert_dist = []
    # for i in range(num_experts):
    #     for j in range(num_experts):
    #         if i != j:
    #             expert_i = expert_outputs[:, :, :, i].mean(dim=1)  # [batch, d_ff]
    #             expert_j = expert_outputs[:, :, :, j].mean(dim=1)  # [batch, d_ff]
    #             expert_dist.append(hyperbolic_distance(expert_i, expert_j, alpha))
    # expert_diversity_loss = -torch.stack(expert_dist).mean()  # 负号表示最大化
    expert_i = expert_outputs.mean(dim=1).unsqueeze(2)  # [batch, d_ff, 1, num_experts]
    expert_j = expert_outputs.mean(dim=1).unsqueeze(3)  # [batch, d_ff, num_experts, 1]
    expert_dist_matrix = hyperbolic_distance(expert_i, expert_j, alpha)  # [batch, d_ff, num_experts, num_experts]
    expert_diversity_loss = -expert_dist_matrix.sum() / (num_experts * (num_experts - 1))

    return shared_loss, expert_shared_loss, expert_diversity_loss

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
        
    @timing_decorator
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
        # 计算所有专家的输出 [batch_size, seq_len, d_ff, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [batch, seq_len, d_ff, num_experts]

        # 调整 routing_weights 形状以匹配 expert_outputs
        # routing_weights = routing_weights.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_experts]
        batch_size = binary_routes.size(0)
        num_experts = len(self.experts)
        repeat_factor = num_experts // 3
        # routing_weights = binary_routes.repeat_interleave(repeat_factor, dim=1) 
        # routing_weights = routing_weights.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_experts]
        routing_weights = binary_routes.unsqueeze(1).unsqueeze(2).expand(-1, 1, 1, num_experts)

        # 按照 routing_weights 计算专家输出的加权和
        routed_out = torch.sum(expert_outputs * routing_weights, dim=-1)  # [batch, seq_len, d_ff]

        # 共享专家计算
        shared_out = sum(expert(x) for expert in self.shared_experts)
        # 组合输出（共享专家 + MoE 输出）
        output = shared_out + routed_out  # [batch, seq_len, d_model]

        # TODO 重写 loss。 loss 包含三部分组成
        shared_loss, expert_shared_loss, expert_diversity_loss = compute_expert_loss(
            shared_out, 
            expert_outputs.view(*expert_outputs.shape[:-1], -1, repeat_factor).sum(dim=-1), 
            binary_routes)
        expert_loss = shared_loss + expert_shared_loss + expert_diversity_loss + 12
        print(shared_loss, expert_shared_loss, expert_diversity_loss)

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
