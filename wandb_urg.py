import sys
import argparse
import os
import time
import pandas as pd
import wandb
import sys
sys.path.append('/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/KMVE_RG')

from config_nassir_urg import Config 

# Set up argument parser
parser = argparse.ArgumentParser(description='Run WandB logging for ultrasound report generation.')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to log metrics for.')
args = parser.parse_args()
config=Config(dataset_name = args.dataset)
# Update log_file_path based on the dataset argument
log_file_path = f'{config.Result_prefix}/{args.dataset}_log.csv'

print(config)

wandb.init(
    project="Nassir-US-Report-Gen",
    config=config,
    name=f'{args.dataset}'
)

def read_all_metrics(log_file):
    return pd.read_csv(log_file)

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
else:
    raise FileNotFoundError(f"{log_file_path}")
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

    # Sleep for a while before checking again
    time.sleep(10)
