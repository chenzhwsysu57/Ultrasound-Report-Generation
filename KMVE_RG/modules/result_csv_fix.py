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


def add_metrics(csv_path):
    df = pd.read_csv(csv_path)
    
    assert(df.columns.equals(pd.Index(['ID', 'gt', 'pred'], dtype='object')))
    metrics = []
    from concurrent.futures import  as_completed, ThreadPoolExecutor # ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(compute_single_metric, ID, gt, pred): (ID, gt, pred)
            for ID, gt, pred in zip(df['ID'], df['gt'], df['pred'])
        }
        for future in tqdm(as_completed(futures), desc='Computing metrics', total=len(df)):
            # print(future.result())
            metrics.append(future.result())
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(csv_path, index=False)

def main(args):
    fix_csv1(args.csv_path)
    pass 

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some parameters.')
    # parser.add_argument('--csv_path', type=str, default='/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/Nassir/Thyroid_test_restult_48.csv', help='Path to csv')
    # args = parser.parse_args()
    # main(args)
    import glob 

    files = glob.glob('/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/*/*_restult_*.csv')
    files = [file for file in files if '/tmp2' not in file and 'restult' in file]
    for file in tqdm(files, desc='csv fix1'):
        try:
            fix_csv1(file)
        except Exception as e:
            print(f"Error in {file}: {e}")

    files = glob.glob('/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/*/*_result_*.csv')
    for file in tqdm(files, desc='csv fix2'):
        try:
            fix_csv2(file)
        except Exception as e:
            print(f"Error in {file}: {e}")
    
    files = glob.glob('/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result/*/*_result_*.csv')
    files = [file for file in files if 'result_49' in file or 'result_30' in file or 'result_5' in file or 'result_1' in file]
    for file in tqdm(files, desc='computing metrics'):
        try:
            add_metrics(file)
        except Exception as e:
            print(f"Error in {file}: {e}")