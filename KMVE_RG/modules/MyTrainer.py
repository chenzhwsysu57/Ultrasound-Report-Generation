import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



import os
from abc import abstractmethod
import time
import torch
import pandas as pd
from numpy import inf
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer, util
from torch.nn import functional as F
from tqdm import tqdm

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

class Timer:
    def __init__(self, comment="Block"):
        self.comment = comment

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print(f"{self.comment} took {self.interval:.4f} seconds")
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.device = 0
        self.model = model.to(self.device)
        self.criterion = criterion
        self.criterionBCE = torch.nn.CrossEntropyLoss()

        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},

                              'test': {self.mnt_metric_test: self.mnt_best}}
        self.sentence_bert = SentenceTransformer('distiluse-base-multilingual-cased',local_files_only=True)

        self.lambada1 = torch.nn.Parameter(torch.tensor(0.7), requires_grad=True)
        self.lambada2 = torch.nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.lambada3 = torch.nn.Parameter(torch.tensor(0.3), requires_grad=True)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        df = None
        path = f'{self.args.Result_prefix}/{self.args.dataset_name}_log.csv'
        file_exists = os.path.isfile(path)
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            df = pd.DataFrame([log])  
            df.to_csv(path, mode='a', index=False, header=not file_exists)
            file_exists = True


            # if df is None:
            #     df = pd.DataFrame.from_dict(log, orient='index').T
            # else:
            #     df = pd.concat([df, pd.Series(log)], ignore_index=True)
            # df.to_csv(path, index=False)
            self._record_best(log)

            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            self._save_checkpoint(epoch, save_last=True)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.Series(self.best_recorder['val'])], ignore_index=True)
        record_table = pd.concat([record_table, pd.Series(self.best_recorder['test'])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    
    @timing_decorator
    def _save_checkpoint(self, epoch, save_best=False,save_last=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, f'{self.args.dataset_name}_epoch_{epoch}_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, f'{self.args.dataset_name}_best.pth')
            torch.save(state, best_path)
            print(f"Saving current best: {self.args.dataset_name}_best.pth ...")
        if save_last:
            last_path = os.path.join(self.checkpoint_dir, f'{self.args.dataset_name}_last.pth')
            torch.save(state, last_path)
            print(f"Saving last checkpoint: {self.args.dataset_name}_last.pth ...")
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        last_path = os.path.join(self.checkpoint_dir, f'{self.args.dataset_name}_last.pth')
        print("Loading checkpoint: {} ...".format(last_path))
        checkpoint = torch.load(last_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.start_epoch > self.args.epochs:
            print("Trained done. Return.")
            return
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        # improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
        #     self.mnt_metric_test]) or \
        #                 (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
        #                     self.mnt_metric_test])
        # if improved_test:
        #     self.best_recorder['test'].update(log)
        for organ in set(['Liver', 'Mammary', 'Thyroid']):
            organ_metric_key = f'{organ}_{self.mnt_metric_test}'
            if organ_metric_key in log:
                improved_organ = (self.mnt_mode == 'min' and log[organ_metric_key] <= self.best_recorder['test'].get(organ_metric_key, float('inf'))) or \
                                (self.mnt_mode == 'max' and log[organ_metric_key] >= self.best_recorder['test'].get(organ_metric_key, float('-inf')))
                if improved_organ:
                    self.best_recorder['test'][organ_metric_key] = log[organ_metric_key]

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def logloss(self, y_true, y_pred, eps=1e-15):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert (len(y_true) and len(y_true) == len(y_pred))

        p = np.clip(y_pred, eps, 1 - eps)
        loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1 - p))

        return loss / len(y_true)

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate \
                    (self.train_dataloader), total=len(self.train_dataloader)):
            images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(self.device), \
                                                             reports_masks.to(self.device), mesh_label.to(self.device)


            indices = torch.randperm(images.shape[0])[:5]
            images_select = images[indices]
            reports_select = reports_ids[indices]

            self.model.eval()
            
            pred_output, _ = self.model(images_select, mode='sample')
            predcit_reports = '.'.join(self.model.tokenizer.decode_batch(pred_output.cpu().numpy()))
            ground_truths = '.'.join(self.model.tokenizer.decode_batch(reports_select[:, 1:].cpu().numpy()))

            self.model.train()
            pred_embeddings = self.sentence_bert.encode(predcit_reports, convert_to_tensor=True)
            gt_embeddings = self.sentence_bert.encode(ground_truths, convert_to_tensor=True)
            pred_embeddings = pred_embeddings.unsqueeze(0)
            gt_embeddings = gt_embeddings.unsqueeze(1)
            similarity_scores = F.cosine_similarity(pred_embeddings, gt_embeddings)
            mean_similarity_score = torch.mean(similarity_scores)
            similarity_loss = 1 - mean_similarity_score
            CS_L = torch.tensor(similarity_loss, requires_grad=True).to(self.device)

            output, kmve_output = self.model(images, reports_ids, mode='train')
            print(f"in Trainer: kvme_output = {kmve_output}, mesh_label = {mesh_label}")
            KMVE_l = self.criterionBCE(kmve_output, mesh_label)
            RG_L = self.criterion(output, reports_ids, reports_masks)
            total_loss = self.lambada1 * RG_L + self.lambada2 * KMVE_l + self.lambada3*CS_L
            train_loss = train_loss + self.lambada1.item() * RG_L.item() + \
                         self.lambada2.item() * KMVE_l.item() + self.lambada3.item() * CS_L.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print(log['train_loss'])

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate(
                    self.val_dataloader), total=len(self.val_dataloader)):
                images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), mesh_label.to(self.device)
                output, kmve_output = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})

            log.update(**{'val_' + k: v for k, v in val_met.items()})

        df = pd.DataFrame(columns=('key', 'gt', 'pred'))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate(
                    self.test_dataloader), total=len(self.test_dataloader)):
                images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), mesh_label.to(self.device)
                output, kmve_output = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                df = pd.concat([df, pd.Series({'key': images_id, 'gt': ground_truths, 'pred': reports, 'TestAcurracy': 0})],
                               ignore_index=True)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            df = pd.concat([df, pd.Series(test_met)], ignore_index=True)
            file_name = f'{self.args.Result_prefix}/{self.args.dataset_name}_test_restult_{epoch}.csv'
            df.to_csv(file_name, index=False, encoding='utf-8-sig')
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log


class TFTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(TFTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.lambda1 = 0.7 # tf 交叉熵的loss
        self.lambda2 = 0.3 # 器官分类的 loss
    

    def logloss(self, y_true, y_pred, eps=1e-15):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert (len(y_true) and len(y_true) == len(y_pred))

        p = np.clip(y_pred, eps, 1 - eps)
        loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1 - p))

        return loss / len(y_true)
    
    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        accumulation_steps = self.args.accumulation_steps  # 从配置中获取累积步数
        for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate \
                    (self.train_dataloader), total=len(self.train_dataloader),desc=f'Training with batch size {self.args.batch_size}'):
            images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(self.device), \
                                                             reports_masks.to(self.device), mesh_label.to(self.device)


            output,pred_classified  = self.model(images, reports_ids, mode='train')
            ORGAN_L = self.criterionBCE(pred_classified, mesh_label)
            
            RG_L = self.criterion(output, reports_ids, reports_masks)
            batch_loss = self.lambda1 * RG_L
            batch_loss /= accumulation_steps
            train_loss = train_loss + self.lambda1 * RG_L.item() 

            batch_loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()  # 更新参数
                self.optimizer.zero_grad()  # 清空梯度

            
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print(f"""train loss {log['train_loss']}""")

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate(
                    self.val_dataloader), total=len(self.val_dataloader),desc=f'Validation with batch size {self.args.batch_size}'):
                images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), mesh_label.to(self.device)
                output,_  = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})

            log.update(**{'val_' + k: v for k, v in val_met.items()})

        results = []
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            test_organ = []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in \
                tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader),desc=f'Test with batch size {self.args.batch_size}'):
                
                images, reports_ids, reports_masks, mesh_label = \
                    images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), mesh_label.to(self.device)
                
                output,_ = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                
                # 不计算 metrics，改为patch函数计算
                results.extend(
                    [{'ID': ID, 'gt': gt, 'pred': pred, } for ID, gt, pred in zip(images_id, ground_truths, reports)]
                )

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_organ.extend([self.args.get_organs_from_label(one_label) for one_label in mesh_label])
            organ_metrics = {}
            for organ in set(test_organ):
                organ_gts = {i: [gt] for i, (gt, org) in enumerate(zip(test_gts, test_organ)) if org == organ}
                organ_res = {i: [re] for i, (re, org) in enumerate(zip(test_res, test_organ)) if org == organ}
                organ_metrics[organ] = self.metric_ftns(organ_gts, organ_res)
            for organ in set(test_organ):
                organ_metric = organ_metrics[organ]
                log.update({
                    f'test_{organ}_BLEU_1': organ_metric['BLEU_1'],
                    f'test_{organ}_BLEU_2': organ_metric['BLEU_2'],
                    f'test_{organ}_BLEU_3': organ_metric['BLEU_3'],
                    f'test_{organ}_BLEU_4': organ_metric['BLEU_4'],
                    f'test_{organ}_METEOR': organ_metric['METEOR'],
                    f'test_{organ}_ROUGE_L': organ_metric['ROUGE_L']
                })

            
            directory = f'{self.args.Result_prefix}/generated'
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = f'{self.args.Result_prefix}/generated/{self.args.dataset_name}_test_result_{epoch}.csv'
            df = pd.DataFrame(results)
            df.to_csv(file_name, index=False, encoding='utf-8-sig')
            
        self.lr_scheduler.step()

        return log

class MoETrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(MoETrainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.lambda1 = 0.6 # tf 交叉熵的loss
        self.lambda2 = 0.4 # 图像 的 loss
        self.lambda3 = 0.01 # MoE loss
    

    def logloss(self, y_true, y_pred, eps=1e-15):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert (len(y_true) and len(y_true) == len(y_pred))

        p = np.clip(y_pred, eps, 1 - eps)
        loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1 - p))

        return loss / len(y_true)
    @timing_decorator
    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        accumulation_steps = self.args.accumulation_steps  # 从配置中获取累积步数
        for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(
                enumerate(self.train_dataloader), 
                total=len(self.train_dataloader),
                desc=f'Training with batch size {self.args.batch_size}'
        ):
            images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), mesh_label.to(self.device)


            output, pred_classified, expert_loss = self.model(images, reports_ids, mode='train')
            ORGAN_L = self.criterionBCE(pred_classified, mesh_label)
            RG_L = self.criterion(output, reports_ids, reports_masks)
            batch_loss = self.lambda1 * RG_L + self.lambda2 * ORGAN_L + expert_loss
            batch_loss /= accumulation_steps
            train_loss = train_loss + self.lambda1 * RG_L.item() + self.lambda2 * ORGAN_L.item() + self.lambda3 * expert_loss.item()
            
            batch_loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()  # 更新参数
                self.optimizer.zero_grad()  # 清空梯度


        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print(f"""train loss {log['train_loss']}""")

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in tqdm(enumerate(
                    self.val_dataloader), total=len(self.val_dataloader),desc=f'Validation with batch size {self.args.batch_size}'):
                images, reports_ids, reports_masks, mesh_label = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device), mesh_label.to(self.device)
                output,_  = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)}, {i: [re] for i, re in enumerate(val_res)})

            log.update(**{'val_' + k: v for k, v in val_met.items()})

        results = []
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            test_organ = []
            for batch_idx, (images_id, images, cap_lens, reports_ids, reports_masks, mesh_label) in \
                tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader),desc=f'Test with batch size {self.args.batch_size}'):
                
                images, reports_ids, reports_masks, mesh_label = \
                    images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), mesh_label.to(self.device)
                
                output,_ = self.model(images, mode='sample')

                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                
                # 不计算 metrics，改为patch函数计算
                results.extend(
                    [{'ID': ID, 'gt': gt, 'pred': pred, } for ID, gt, pred in zip(images_id, ground_truths, reports)]
                )

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_organ.extend([self.args.get_organs_from_label(one_label) for one_label in mesh_label])
            organ_metrics = {}
            for organ in set(test_organ):
                organ_gts = {i: [gt] for i, (gt, org) in enumerate(zip(test_gts, test_organ)) if org == organ}
                organ_res = {i: [re] for i, (re, org) in enumerate(zip(test_res, test_organ)) if org == organ}
                organ_metrics[organ] = self.metric_ftns(organ_gts, organ_res)
            for organ in set(test_organ):
                organ_metric = organ_metrics[organ]
                log.update({
                    f'test_{organ}_BLEU_1': organ_metric['BLEU_1'],
                    f'test_{organ}_BLEU_2': organ_metric['BLEU_2'],
                    f'test_{organ}_BLEU_3': organ_metric['BLEU_3'],
                    f'test_{organ}_BLEU_4': organ_metric['BLEU_4'],
                    f'test_{organ}_METEOR': organ_metric['METEOR'],
                    f'test_{organ}_ROUGE_L': organ_metric['ROUGE_L']
                })

            
            directory = f'{self.args.Result_prefix}/generated'
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = f'{self.args.Result_prefix}/generated/{self.args.dataset_name}_test_result_{epoch}.csv'
            df = pd.DataFrame(results)
            df.to_csv(file_name, index=False, encoding='utf-8-sig')

        self.lr_scheduler.step()

        return log
