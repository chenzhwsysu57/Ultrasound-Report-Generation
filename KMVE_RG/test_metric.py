import argparse
import sys
import os
import torch
from tqdm import tqdm 
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dataloaders import MyDataLoader
from modules.tokenizers import Tokenizer
from modules.metrics import compute_scores
from config_urg import Config


def main(cmd_args, config_args):

    # 对指定器官、指定模型的测试集给出结果。输出：
    # Dataset,Method,B1,B2,B3,B4,Meteor,Rougel,CEacc,CEpre,CErecall,CEF1
    # 不包含器官分类

    # 仅仅适用all.json数据集
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
    print(f'Using {device} device')
    model_module = importlib.import_module(f'KMVE_RG.models.{cmd_args["model"]}', )
    model_class = getattr(model_module, cmd_args["model"])

    tokenizer = Tokenizer(config_args)
    test_dataloader = MyDataLoader(config_args, tokenizer, split='test', shuffle=True)
    model = model_class(config_args, tokenizer)
    model = model.to(device)
    model.eval()

    # load weight
    checkpoint = torch.load(cmd_args["ckpt"])
    model.load_state_dict(checkpoint['state_dict'])
    print(f"checkpoint loaded from {cmd_args['ckpt']}")
    metric_ftns = compute_scores
    
    
    with torch.no_grad():
        test_gts, test_res, test_organ = [], [], []

        for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in \
                tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            if cmd_args.get('save_past_values', 'false') == 'true':
                save_dir = f'US-Report-Gen/tracker/batch_{batch_idx}'
                os.makedirs(save_dir, exist_ok=True)
                batch_save_path = os.path.join(save_dir, f"source.pt")
                torch.save({
                    'batch_idx': batch_idx,
                    'images_id': images_id,
                    'images': images,
                    'cap_lens': cap_lens,
                    'reports_ids': reports_ids,
                    'reports_masks': reports_masks,
                    'mesh_label': mesh_label,
                    'cmd_args': cmd_args
                }, batch_save_path)
            
            images, reports_ids, reports_masks, mesh_label = images.to(device), reports_ids.to(
                    device), reports_masks.to(device), mesh_label.to(device)
            output,_  = model(images, mode='sample') # output, _ 增加的这个是为了兼容 organ 分类
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
            test_organ.extend([config_args.get_organs_from_label(one_label) for one_label in mesh_label])

        organ_metrics = {}
        for organ in set(test_organ):
            organ_gts = {i: [gt] for i, (gt, org) in enumerate(zip(test_gts, test_organ)) if org == organ}
            organ_res = {i: [re] for i, (re, org) in enumerate(zip(test_res, test_organ)) if org == organ}
            organ_metrics[organ] = metric_ftns(organ_gts, organ_res)
            
        print(f"\033[1;35mtest result on {cmd_args['comment']}\033[0m")
        for organ in set(test_organ):
            # 输出结果，只保留三位小数
            result = f"{organ},{cmd_args['method']},{organ_metrics[organ]['BLEU_1']:.3f},{organ_metrics[organ]['BLEU_2']:.3f},{organ_metrics[organ]['BLEU_3']:.3f},{organ_metrics[organ]['BLEU_4']:.3f},{organ_metrics[organ]['METEOR']:.3f},{organ_metrics[organ]['ROUGE_L']:.3f},0,0,0,0"
            
            print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--model', type=str, default='AllOrgan', help='Path to the model. you can also use MoEModel, SGF.')
    parser.add_argument('--dataset_name', type=str, default='all', help='Path to the dataset')
    parser.add_argument('--save_path', type=str,  help='Path to save the results')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint')
    parser.add_argument('--method', type=str, help='Method')
    parser.add_argument('--comment', type=str, help='Comment')
    parser.add_argument('--decoderonly', type=str, default='False', help='use decoder only model.')
    parser.add_argument('--norm', type=str, default='layernorm', help='can also use rmsnorm')
    parser.add_argument('--save_past_values', type=str, default='false', help='should only be used during inference')
    known_args, unknown_args = parser.parse_known_args()
    extra_args = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:  # 处理 --key=value 形式
                k, v = key.split("=", 1)
                extra_args[k] = v
            else:  # 处理 --key value 形式
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    extra_args[key] = unknown_args[i + 1]  # 取下一个值
                    i += 1  # 跳过 value
                else:
                    extra_args[key] = True  # 只有 key，没有 value
        i += 1
    cmd_line_args = {**vars(known_args), **extra_args}
    # cmd_args = parser.parse_args()


    config_args = Config(**cmd_line_args)
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
    main(cmd_line_args,config_args)