# 推理一个例子。
import argparse
import sys
import os
import torch
from tqdm import tqdm 
import importlib
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from modules.dataloaders import MyDataLoader
from modules.tokenizers import Tokenizer
from modules.metrics import compute_scores
from config_nassir_urg import Config
from modules.tokenizers import Tokenizer
from KMVE_RG.models.SGF import SGF
from KMVE_RG.models.AllOrgan import AllOrgan

def compute_single_metric(ID, gt, pred):
    single_metrics = compute_scores({0: [gt]}, {0: [pred]})
    return {**single_metrics}

def get_image(uid):
    # 从 uid 获取图片路径
    path1 = f'/home/chenzhw/ultrasound_report_gen/USData/all_report/{uid}_1.jpeg'
    path2 = f'/home/chenzhw/ultrasound_report_gen/USData/all_report/{uid}_2.jpeg'
    # TODO 修改这段代码
    image_1 = Image.open(path1).convert('RGB')
    image_2 = Image.open(path2).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    if transform is not None:
        image_1 = transform(image_1)
        image_2 = transform(image_2)
    image = torch.stack((image_1, image_2), dim=0)
    return image

import json
def get_tokens(uid):
    data = []
    with open('/home/chenzhw/ultrasound_report_gen/USData/new_all2.json', 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    for split in ["train", "val", "test"]:
        for entry in data[split]:
            
            if entry["uid"] == uid:
                return {
                        "finding": entry["finding"],
                        "labels": entry["labels"],
                        "split": split
                    }
            
    return None
def main():
    import argparse
    parser = argparse.ArgumentParser(description='inference one.')
    parser.add_argument('--uid', type=int, default=215216, help="uid of case." )
    parser.add_argument('--checkpoint', type=str, default='none', help="checkpoint to load model.")
    parser.add_argument('--model', type=str, default='SGF',choices=['SGF','AllOrgan'], help="model to initiate")
    parser.add_argument('--organ',type=str,default='Liver',help='no need if you do not use SGF model.')
    parser.add_argument('--candidate', type=str,default=None, help="if specify, won't use model to generate candidate.")
    parser.add_argument('--reference',type=str, default=None,help="answers to reference.")

    args = parser.parse_args()
    # load model
    if args.model == "SGF":
        uid_info = get_tokens(args.uid)
        args.organ = uid_info['labels']
        # load SGF
        config = Config(dataset_name = args.organ, result = 'none')
        tokenizer = Tokenizer(config)
        # print(tokenizer.tokens)
        model = SGF(config, tokenizer)
        if args.checkpoint == 'none':
            # 自动选择最好的checkpoint
            uid_info = get_tokens(args.uid)
            args.organ = uid_info['labels']
            args.checkpoint = f'/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/Nassir/Models/{args.organ}_best.pth'
            # load checkpoint
            print(f"ckpt {args.checkpoint}")
            ckpt = torch.load(args.checkpoint)
            model.load_state_dict(ckpt['state_dict'])

            model.eval()
            with torch.no_grad():
                image = get_image(args.uid)
                image = torch.stack([image], 0)
                uid_info = get_tokens(args.uid)
                report_ids = tokenizer(uid_info['finding'])
                
                output_ids, _ = model(image,mode='sample')
                predict_reports = ' '.join(model.tokenizer.decode_batch(output_ids.cpu().numpy()))
                # os.system('clear')
                if args.candidate: predict_reports = args.candidate
                print(f'pd \033[1;35m{predict_reports}\033[0m' )
                ground_reports = ' '.join(tokenizer.decode_batch([report_ids[1:-2]]))
                print(f'gt \033[1;36m{ground_reports}\033[0m' )
                split, organ = uid_info['split'],uid_info['labels']
                print(f'split: \033[1;37m{split}\033[0m')
                print(f'organ: \033[1;37m{organ}\033[0m')
                print(f'uid: \033[1;37m{args.uid}\033[0m')
                # 输出各项计算指标
                from modules.metrics import compute_scores
                metrics = compute_single_metric(args.uid, ground_reports, predict_reports)
                print(metrics)
                print(f'/home/chenzhw/ultrasound_report_gen/USData/all_report/{args.uid}_1.jpeg')
                print(f'/home/chenzhw/ultrasound_report_gen/USData/all_report/{args.uid}_2.jpeg')
if __name__=="__main__":
    main()