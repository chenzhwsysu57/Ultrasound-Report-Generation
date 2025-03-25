import sys
import argparse
import os
import time
import pandas as pd
import wandb
import sys
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}/KMVE_RG')

from config_urg import Config 

# Set up argument parser
parser = argparse.ArgumentParser(description='Run WandB logging for ultrasound report generation.')
parser.add_argument('--dataset', type=str, default='all', help='Name of the dataset to log metrics for.')
parser.add_argument('--project', type=str, default="Nassir-US-Report-Gen", help='Name of the dataset to log metrics for.')
parser.add_argument('--comment', type=str, default='today', help='a very short comment without blankspace.')
parser.add_argument('--result', type=str, default='TF_only', help='result path')
args = parser.parse_args()
config=Config(dataset_name = args.dataset, result = args.result)
# Update log_file_path based on the dataset argument
log_file_path = f'{config.Result_prefix}/{args.dataset}_log.csv'

print(config)

wandb.init(
    project=args.project,
    config=config,
    name=args.comment
)

def read_all_metrics(log_file):
    return pd.read_csv(log_file)

while True:
    if os.path.exists(log_file_path):
        df = read_all_metrics(log_file_path)
        for _, row in df.iterrows():
            wandb.log({
                "train_loss": row['train_loss'],
                "val_BLEU_1": row['val_BLEU_1'],
                "val_BLEU_2": row['val_BLEU_2'],
                "val_BLEU_3": row['val_BLEU_3'],
                "val_BLEU_4": row['val_BLEU_4'],
                "val_METEOR": row['val_METEOR'],
                "val_ROUGE_L": row['val_ROUGE_L'],
                "val_CIDER": row['val_CIDER'],
                "test_BLEU_1": row['test_BLEU_1'],
                "test_BLEU_2": row['test_BLEU_2'],
                "test_BLEU_3": row['test_BLEU_3'],
                "test_BLEU_4": row['test_BLEU_4'],
                "test_METEOR": row['test_METEOR'],
                "test_ROUGE_L": row['test_ROUGE_L'],
                "test_CIDER": row['test_CIDER'],
                # "epoch": row['epoch'],  # Record the current epoch
            })
        break
    else:
        # raise FileNotFoundError(f"{log_file_path}")
        print("waiting log file")
        time.sleep(5)
    
print("Previous log done.")
# Keep track of the last logged epoch
last_epoch = df['epoch'].max() if not df.empty else 0

while True:
    # Check if the file exists
    if os.path.exists(log_file_path):
        latest_metrics = read_all_metrics(log_file_path)
        current_epoch = latest_metrics['epoch'].max()  # Get the current maximum epoch

        # Log new metrics if there are updates
        if current_epoch > last_epoch:
            new_rows = latest_metrics[latest_metrics['epoch'] > last_epoch]
            for _, row in new_rows.iterrows():
                last_epoch = row['epoch']
                wandb.log({
                    "train_loss": row['train_loss'],
                    "val_BLEU_1": row['val_BLEU_1'],
                    "val_BLEU_2": row['val_BLEU_2'],
                    "val_BLEU_3": row['val_BLEU_3'],
                    "val_BLEU_4": row['val_BLEU_4'],
                    "val_METEOR": row['val_METEOR'],
                    "val_ROUGE_L": row['val_ROUGE_L'],
                    "val_CIDER": row['val_CIDER'],
                    "test_BLEU_1": row['test_BLEU_1'],
                    "test_BLEU_2": row['test_BLEU_2'],
                    "test_BLEU_3": row['test_BLEU_3'],
                    "test_BLEU_4": row['test_BLEU_4'],
                    "test_METEOR": row['test_METEOR'],
                    "test_ROUGE_L": row['test_ROUGE_L'],
                    "test_CIDER": row['test_CIDER'],
                })
    else:
        print(f"File {log_file_path} does not exist. Waiting.")
    # Sleep for a while before checking again
    time.sleep(10)
