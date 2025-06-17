# inference_api.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
from modules.tokenizers import Tokenizer
from config_urg import Config
from KMVE_RG.models.AllOrgan import AllOrgan
import json
import torch
from PIL import Image
from torchvision import transforms
from modules.tokenizers import Tokenizer
from config_urg import Config
from KMVE_RG.models.AllOrgan import AllOrgan
import json
import numpy as np
from torch.nn.functional import normalize
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, img_path

class UltrasoundReportModel:
    def __init__(self, organ='all'):
        # ======== 自动设备选择 ========
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[设备选择] 使用设备: {self.device}")

        # 初始化模型等
        self.organ = organ
        self.config = Config(dataset_name=organ, result='none', decoderonly='False')
        self.tokenizer = Tokenizer(self.config)
        self.model = AllOrgan(self.config, self.tokenizer).to(self.device)

        ckpt_path = f'/home/chenzhw/ultrasound_report_gen/US-Report-Gen/Result/TF_organ_balence_sampler-4090/Models/all_best.pth'
        print(f"[加载模型] {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        embedding_file = "/home/chenzhw/ultrasound_report_gen/USData/embedding_index.json"
        if os.path.exists(embedding_file):
            with open(embedding_file, 'r') as f:
                db = json.load(f)
                self.db = db
        else:
            self.db = None
    def preprocess(self, image_list):
        assert len(image_list) == 2, "需要传入两张图像"
        images = [self.transform(Image.open(img).convert('RGB')) for img in image_list]
        input_tensor = torch.stack(images, dim=0).unsqueeze(0)  # shape: (1, 2, C, H, W)
        return input_tensor.to(self.device)

    def infer(self, image_list):
        with torch.no_grad():
            input_tensor = self.preprocess(image_list)
            output_ids, _ = self.model(input_tensor, mode='sample')
            report = ' '.join(self.tokenizer.decode_batch(output_ids.cpu().numpy()))
            return report

    def find_topk_similar(self, image_path, embedding_file, topk=5):
        if not self.db:
            with open(embedding_file, 'r') as f:
                db = json.load(f)
                self.db = db
        else:
            db = self.db
        target_img = self.transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, dense_vec = self.model.visual_extractor(target_img)
            query_vec = normalize(dense_vec, dim=1).squeeze(0).cpu().numpy()

        sims = []
        for entry in db:
            db_vec = np.array(entry['embedding'])
            sim = np.dot(query_vec, db_vec)
            if not np.isclose(sim, 1.0, atol=1e-6):
                sims.append((entry['path'], sim))
            

        sims.sort(key=lambda x: -x[1])
        return sims[:topk]

    def build_embedding_index(self, image_folder, save_path, batch_size=128, num_workers=2):
        if os.path.exists(save_path):
            print(f"[跳过] 已存在embedding文件: {save_path}")
            return

        image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpeg')) +
                             glob.glob(os.path.join(image_folder, '*.png')))
        dataset = ImageFolderDataset(image_paths, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        embeddings = []
        paths = []

        with torch.no_grad():
            for batch_imgs, batch_paths in tqdm(dataloader, desc='批量构建图像Embedding'):
                batch_imgs = batch_imgs.to(self.device)
                _, _, _, dense_vecs = self.model.visual_extractor(batch_imgs)
                dense_vecs = normalize(dense_vecs, dim=1).cpu().numpy()
                embeddings.extend(dense_vecs.tolist())
                paths.extend(batch_paths)

        data = [{'path': p, 'embedding': e} for p, e in zip(paths, embeddings)]
        with open(save_path, 'w') as f:
            json.dump(data, f)
        print(f"[完成] 保存embedding索引到 {save_path}")


# class UltrasoundReportModel:
#     def __init__(self, organ='all'):
#         # 默认 organ，可通过参数修改
#         self.organ = organ
#         self.config = Config(dataset_name=organ, result='none',decoderonly='False')
#         self.tokenizer = Tokenizer(self.config)
#         self.model = AllOrgan(self.config, self.tokenizer)

#         ckpt_path = f'/home/chenzhw/ultrasound_report_gen/US-Report-Gen/Result/TF_organ_balence_sampler-4090/Models/all_best.pth'
#         print(f"[加载模型] {ckpt_path}")
#         checkpoint = torch.load(ckpt_path) #, map_location='cpu')
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.model.eval()

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406),
#                                  (0.229, 0.224, 0.225))
#         ])

#     def preprocess(self, image_list):
#         assert len(image_list) == 2, "需要传入两张图像"
#         images = [self.transform(Image.open(img).convert('RGB')) for img in image_list]
#         return torch.stack(images, dim=0).unsqueeze(0)  # shape: (1, 2, C, H, W)

#     def infer(self, image_list):
#         with torch.no_grad():
#             input_tensor = self.preprocess(image_list)
#             output_ids, _ = self.model(input_tensor, mode='sample')
#             report = ' '.join(self.tokenizer.decode_batch(output_ids.cpu().numpy()))
#             return report

#     # def build_embedding_index(self, image_folder, save_path):
#     #     if os.path.exists(save_path):
#     #         print(f"[跳过] 已存在embedding文件: {save_path}")
#     #         return

#     #     image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpeg')) +
#     #                          glob.glob(os.path.join(image_folder, '*.png')))
#     #     embeddings = []
#     #     paths = []

#     #     with torch.no_grad():
#     #         for path in tqdm(image_paths, desc='构建图像Embedding'):
#     #             img = self.transform(Image.open(path).convert('RGB')).unsqueeze(0)  # (1, 3, H, W)
#     #             _, _, _, dense_vec = self.model.visual_extractor(img)  # shape: (1, D)
#     #             vec = normalize(dense_vec, dim=1).squeeze(0).cpu().numpy()  # normalize to unit length
#     #             embeddings.append(vec)
#     #             paths.append(path)

#     #     data = [{'path': p, 'embedding': e.tolist()} for p, e in zip(paths, embeddings)]
#     #     with open(save_path, 'w') as f:
#     #         json.dump(data, f)
#     #     print(f"[完成] 保存embedding索引到 {save_path}")

#     # ======== 新增功能 2: 查找最相似图片 ========
#     def find_topk_similar(self, image_path, embedding_file, topk=5):
#         with open(embedding_file, 'r') as f:
#             db = json.load(f)

#         target_img = self.transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
#         with torch.no_grad():
#             _, _, _, dense_vec = self.model.visual_extractor(target_img)
#             query_vec = normalize(dense_vec, dim=1).squeeze(0).cpu().numpy()

#         # 计算 cosine 相似度
#         sims = []
#         for entry in db:
#             db_vec = np.array(entry['embedding'])
#             sim = np.dot(query_vec, db_vec)
#             sims.append((entry['path'], sim))

#         sims.sort(key=lambda x: -x[1])
#         return sims[:topk]
    
#     from torch.utils.data import Dataset, DataLoader



#     def build_embedding_index(self, image_folder, save_path, batch_size=128, num_workers=2):
#         if os.path.exists(save_path):
#             print(f"[跳过] 已存在embedding文件: {save_path}")
#             return

#         image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpeg')) +
#                             glob.glob(os.path.join(image_folder, '*.png')))
#         dataset = ImageFolderDataset(image_paths, self.transform)
#         dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

#         embeddings = []
#         paths = []

#         with torch.no_grad():
#             for batch_imgs, batch_paths in tqdm(dataloader, desc='批量构建图像Embedding'):
#                 # batch_imgs shape: (B, 3, H, W)
#                 _, _, _, dense_vecs = self.model.visual_extractor(batch_imgs)
#                 dense_vecs = normalize(dense_vecs, dim=1).cpu().numpy()  # shape: (B, D)

#                 embeddings.extend(dense_vecs.tolist())
#                 paths.extend(batch_paths)

#         data = [{'path': p, 'embedding': e} for p, e in zip(paths, embeddings)]
#         with open(save_path, 'w') as f:
#             json.dump(data, f)
#         print(f"[完成] 保存embedding索引到 {save_path}")

if __name__=="__main__":
    model = UltrasoundReportModel()
    model.build_embedding_index('/home/chenzhw/ultrasound_report_gen/USData/all_report', '/home/chenzhw/ultrasound_report_gen/USData/embedding_index.json')
    image_path = '/home/chenzhw/ultrasound_report_gen/USData/all_report/107368_1.jpeg'
    embedding_file = '/home/chenzhw/ultrasound_report_gen/USData/embedding_index.json'
    sims = model.find_topk_similar( image_path, embedding_file, topk=5)
    for idx, (path, sim) in enumerate(sims):

        print(f"{sim} @ {path}")