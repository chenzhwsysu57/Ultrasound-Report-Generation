# 与 result csv fix.py不同，这个文件主要是读取格式已经正确的csv然后批量计算指标并返回。
# 针对统一器官训练的模型结果
# 要先从id得到对应的label是什么器官，然后再进行计算。
import argparse
import sys
import os
import torch
from tqdm import tqdm 
import importlib
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.tokenizers import Tokenizer
from modules.metrics import compute_scores
from config_nassir_urg import Config

def get_label_from_id(id):
    pass

def compute_single_metric(ID, gt, pred):
    single_metrics = compute_scores({0: [gt]}, {0: [pred]})
    return {**single_metrics}

