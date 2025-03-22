# 输入csv表格
# 首先将csv批量修改为格式争取的csv
# 其次再次读取csv根据csv表格计算metrics并写回csv表格
import ast
import argparse
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.metrics import compute_scores

def fix_csv1(csv_path):
    new_csv_path = csv_path.replace('restult', 'result')
    if os.path.exists(new_csv_path):
        return
    df = pd.read_csv(csv_path)
    assert(df.columns.equals(pd.Index(['key', 'gt', 'pred', '0'], dtype='object')))
    new_data = [] # ID, gt, pred only
    raw_datas = df['0']
    for i in range(0, len(raw_datas), 4):
        # 提取数字、gt 和 predict
        numbers = raw_datas[i]
        gt = raw_datas[i+1]
        pred = raw_datas[i+2]
        
        # 将数据添加到新列表中
        new_data.append([numbers, gt, pred])
    
    # 创建新的 DataFrame
    new_df = pd.DataFrame(new_data, columns=['ID', 'gt', 'predict'])
    
    # 保存修复后的 CSV 文件
    new_csv_path = csv_path.replace('restult', 'result')
    new_df.to_csv(new_csv_path, index=False)
    
    # print(f"修复后的 CSV 文件已保存为: {new_csv_path}")
        
def fix_csv2(csv_path):
    df = pd.read_csv(csv_path)
    # 防止覆盖已经有 metrics 的 csv
    assert(df.columns.equals(pd.Index(['ID', 'gt', 'predict'], dtype='object')))
    new_data = [] # ID, gt, pred only
    # TODO
    df = df[0: -2]
    # for i in tqdm(range(0, len(df)),desc='spliting cells'):
    for i in range(0, len(df)):
        
        ids = df.iloc[i]['ID']
        gts = df.iloc[i]['gt']
        preds = df.iloc[i]['predict']
        ids = ast.literal_eval(ids)
        gts = ast.literal_eval(gts)
        preds = ast.literal_eval(preds)
        
        new_data.extend(
            [{"ID": ID, "gt": gt, "pred": pred} for ID, gt, pred in zip(ids, gts, preds)]
        )
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(csv_path, index=False)
    
def compute_single_metric(ID, gt, pred):
    single_metrics = compute_scores({0: [gt]}, {0: [pred]})
    return {'ID': ID, 'gt': gt, 'pred': pred, **single_metrics}

def add_metrics(csv_path, batch_size=100):
    max_workers = 16
    batch_size = max_workers * 1
    # 读取原始数据
    df = pd.read_csv(csv_path)
    
    # 检查是否存在临时文件
    temp_path = csv_path + '.temp'
    if os.path.exists(temp_path):
        temp_df = pd.read_csv(temp_path)
        # 获取已经处理过的ID
        processed_ids = set(temp_df['ID'])
        # 过滤掉已处理的数据
        df = df[~df['ID'].isin(processed_ids)]
        metrics = temp_df.to_dict('records')
    else:
        metrics = []
    
    # 如果所有数据都已处理完，直接返回
    if len(df) == 0:
        return
    
    assert(df.columns.equals(pd.Index(['ID', 'gt', 'pred'], dtype='object')))
    
    from concurrent.futures import as_completed, ThreadPoolExecutor
    
    # 按批次处理数据
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_metrics = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_single_metric, ID, gt, pred): (ID, gt, pred)
                for ID, gt, pred in zip(batch_df['ID'], batch_df['gt'], batch_df['pred'])
            }
            for future in tqdm(as_completed(futures), 
                             desc=f'add metrics {csv_path} ({i}/{len(df)})', 
                             total=len(batch_df)):
                batch_metrics.append(future.result())
        
        # 将新的批次结果添加到总结果中
        metrics.extend(batch_metrics)
        
        # 保存临时文件
        pd.DataFrame(metrics).to_csv(temp_path, index=False)
     
    # 所有数据处理完成后，保存最终结果并删除临时文件
    pd.DataFrame(metrics).to_csv(csv_path, index=False)
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main(args):
    fix_csv1(args.csv_path)
    pass 

if __name__ == "__main__":
    
    import glob 

    
    files = glob.glob('/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/TF_organ_classify/*_result_*.csv')
    files = [file for file in files if 'result_50.csv' in file or 'result_11.csv' in file or 'result_13.csv' in file]
    print(files)
    
    for file in tqdm(files, desc='computing metrics'):
        try:
            add_metrics(file)
        except Exception as e:
            print(f"Error in {file}: {e}")