import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import argparse
import sys
import os
import torch
from tqdm import tqdm 
import importlib
from config_urg import Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # only work in py
import os, sys

# 获取当前 notebook 文件所在的目录
current_dir = os.getcwd()

# 加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))


from modules.dataloaders import MyDataLoader
from modules.tokenizers import Tokenizer
from modules.metrics import compute_scores

def denormalize(tensor, mean, std):
    """
    反标准化一个 shape 为 [B, C, H, W] 或 [C, H, W] 的 Tensor。
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return tensor * std + mean

class UIDCaption:

    '''一个 uid 对应的生成情况'''
    def __init__(self, batch_idx):

        attns = torch.load(f'/home/chenzhw/ultrasound_report_gen/US-Report-Gen/tracker/batch_{batch_idx}/attn.pt')
        past_values = torch.load(f'/home/chenzhw/ultrasound_report_gen/US-Report-Gen/tracker/batch_{batch_idx}/past_values.pt')
        source = torch.load(f'/home/chenzhw/ultrasound_report_gen/US-Report-Gen/tracker/batch_{batch_idx}/source.pt')
        self.source = source
        self.attns = attns
        self.past_values = past_values
        
        
        self.ckpt = self.source['cmd_args']['ckpt']
        self.model = self.load_model(self.source)
    
    def get_images_tensor(self, image_index_in_batch):
        return self.source['images'][image_index_in_batch]
    
    def get_images_denrom(self, images_index_in_batch):
        images = self.source['images'][images_index_in_batch]  

        images_denorm = denormalize(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return images_denorm

    def load_model(self,source):
        config_args = Config(**source['cmd_args'])
        def seed_everything(seed: int):
            if isinstance(seed, str):
                seed = int(seed)
            import random, os
            import numpy as np
            import torch

            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(config_args.seed)
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        model_module = importlib.import_module(f'KMVE_RG.models.{source["cmd_args"]["model"]}', )
        model_class = getattr(model_module, source["cmd_args"]["model"])

        tokenizer = Tokenizer(config_args)
        model = model_class(config_args, tokenizer)
        model = model.to(device)
        model.eval()
        base_dir = os.path.dirname(os.getcwd())
        ckpt = f'{base_dir}/{source["cmd_args"]["ckpt"]}'
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def show_images(self, index):
        images_denorm = self.get_images_denrom(index)
        img1 = images_denorm[0].clamp(0, 1)
        img2 = images_denorm[1].clamp(0, 1)

        # 水平拼接：确保尺寸匹配（都是 [3, 224, 224]）
        concatenated = torch.cat([img1, img2], dim=2)  # dim=2 是宽度方向

        # 显示
        plt.imshow(F.to_pil_image(concatenated))

        plt.axis('off')
        uid = self.source['images_id'][index]
        plt.title(f'UID {uid}')
        plt.show()


    def show_step_attention(self, index: int, step: int):
        
        pass 
    # todo, 展示第 index 个图片在第 step 步的时候，图像热力图以及根据这个热力图得到的结果。展示为热力图+title为预测的词

    def step(self, history, attn):
        pass
    # todo, 传入新的attention矩阵，计算往前推进一步的结果，并更新history

    def alter_attn(self, attn):
        pass
    # todo 人为修改attn


uidcaption = UIDCaption(0)
uidcaption.show_images(7)