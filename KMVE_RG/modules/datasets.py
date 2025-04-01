import os
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
import random

import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()  # 记录结束时间
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper
class BaseDataset(Dataset):
    @timing_decorator
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r', encoding="utf_8_sig").read())

        self.examples = self.ann[self.split]
        random.shuffle(self.examples) 
        self.examples = self.examples[0:args.debug] # [0:50] # used in debug mode
        # for i in range(len(self.examples)):
        #     self.examples[i]['ids'] = tokenizer(self.examples[i]['finding'])[:self.max_seq_length]
        #     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
        # Preload images
        self.preloaded_images = []
        for i in range(len(self.examples)):
            example = self.examples[i]
            image_path = example['image_path']
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
            if self.transform is not None:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)
            self.preloaded_images.append(image)

            # Tokenize and mask
            self.examples[i]['ids'] = tokenizer(self.examples[i]['finding'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])


    def __len__(self):
        return len(self.examples)


class MyDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['uid']
        # image_path = example['image_path']
        # image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        # if self.transform is not None:
        #     image_1 = self.transform(image_1)
        #     image_2 = self.transform(image_2)
        # image = torch.stack((image_1, image_2), 0)
        image = self.preloaded_images[idx]
        report_ids = example['ids']
        report_masks = example['mask']
        mesh_label = example['labels']
        # print(f"mesh_label = {mesh_label}") # 是字符串 Mammary Liver Thyroid
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length, mesh_label)
        return sample



