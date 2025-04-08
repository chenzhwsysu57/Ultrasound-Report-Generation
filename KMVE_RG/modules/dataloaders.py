import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import MyDataset

from torch.utils.data import Sampler
import random

# class BalancedSampler(Sampler):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.organ_indices = self._organize_indices_by_label()
#         self.organ_order = ["Liver", "Mammary", "Thyroid"]  # 预定义采样顺序
#         self.organ_iters = {k: iter(v) for k, v in self.organ_indices.items()}  # 迭代器字典

#     def _organize_indices_by_label(self):
#         """
#         预处理：按照器官类别将数据索引分组，并打乱顺序。
#         """
#         organ_indices = {"Liver": [], "Mammary": [], "Thyroid": []}
#         for idx, example in enumerate(self.dataset.examples):
#             organ = example["labels"]
#             organ_indices[organ].append(idx)

#         for organ in organ_indices:
#             random.shuffle(organ_indices[organ])  # 打乱每个类别的样本顺序

#         return organ_indices

#     def __iter__(self):
#         """
#         轮流采样 Liver → Mammary → Thyroid，直到所有数据采样完毕。
#         """
#         while any(len(v) > 0 for v in self.organ_indices.values()):  # 确保还有数据可采样
#             for organ in self.organ_order:  # 轮流采样 Liver → Mammary → Thyroid
#                 if len(self.organ_indices[organ]) > 0:  # 该类别仍有数据
#                     yield self.organ_indices[organ].pop(0)  # 取出索引

#     def __len__(self):
#         return sum(len(indices) for indices in self.organ_indices.values())
class BalancedSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.organ_indices = self._organize_indices_by_label()
        self.original_organ_indices = {k: v[:] for k, v in self.organ_indices.items()}  # 复制一份原始索引
        self.organ_indices = {k: v[:] for k, v in self.organ_indices.items()}  # 复制一份用于迭代
        self.num_samples = min(len(v) for v in self.organ_indices.values()) * len(self.organ_indices)

    def _organize_indices_by_label(self):
        """
        预处理：按照器官类别将数据索引分组，并打乱顺序。
        """
        organ_indices = {"Liver": [], "Mammary": [], "Thyroid": []}
        for idx, example in enumerate(self.dataset.examples):
            organ = example["labels"]
            organ_indices[organ].append(idx)

        for organ in organ_indices:
            random.shuffle(organ_indices[organ])  # 打乱每个类别的样本顺序

        return organ_indices
    def __iter__(self):
        selected_indices = []
        while len(selected_indices) < self.num_samples:
            for organ, indices in self.organ_indices.items():
                if not indices:  # 如果某个类别已经用完，重置为原始索引
                    self.organ_indices[organ] = self.original_organ_indices[organ][:]
                    random.shuffle(self.organ_indices[organ])  # 重新打乱

                selected_indices.append(self.organ_indices[organ].pop(0))  # 取出一个样本
            
        return iter(selected_indices)

    def __len__(self):
        return self.num_samples

class MyDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, evaluate=False):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        self.dataset = MyDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        
        
        if evaluate == True:
            self.batch_size = args.evaluate_batch

        if self.args.custom_sampler != 'false':
            print("Using custom sampler")
            self.sampler = BalancedSampler(self.dataset)
            self.shuffle = None
        else:
            self.sampler = None
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': int(self.num_workers),
            'sampler': self.sampler,  # 指定自定义的sampler
            'pin_memory': True,
        }
        super().__init__(**self.init_kwargs)

    # @staticmethod
    def collate_fn(self, data):
        images_id, images, reports_ids, reports_masks, seq_lengths, mesh_label = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks
        # print(mesh_label)
        
        if self.args.dataset_name == "all":
            # this code for all organ 
            mesh_label =  tuple([self.args.get_label_from_organ(label) for label in mesh_label])

        # print(mesh_label)
        
        mesh_label = torch.tensor(mesh_label)
        return images_id, images, seq_lengths, torch.LongTensor(targets), torch.FloatTensor(targets_masks), mesh_label
