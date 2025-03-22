import re

def parse_log_file(log_path):
    # 目标字段的正则匹配
    pattern_dict = {
        "epoch": r"epoch\s*:\s*(\d+)",
        "train_loss": r"train_loss\s*:\s*([\d\.e+-]+)",
        "val_BLEU_1": r"val_BLEU_1\s*:\s*([\d\.e+-]+)",
        "val_BLEU_2": r"val_BLEU_2\s*:\s*([\d\.e+-]+)",
        "val_BLEU_3": r"val_BLEU_3\s*:\s*([\d\.e+-]+)",
        "val_BLEU_4": r"val_BLEU_4\s*:\s*([\d\.e+-]+)",
        "val_METEOR": r"val_METEOR\s*:\s*([\d\.e+-]+)",
        "val_ROUGE_L": r"val_ROUGE_L\s*:\s*([\d\.e+-]+)",
        "val_CIDER": r"val_CIDER\s*:\s*([\d\.e+-]+)",
        "test_BLEU_1": r"test_BLEU_1\s*:\s*([\d\.e+-]+)",
        "test_BLEU_2": r"test_BLEU_2\s*:\s*([\d\.e+-]+)",
        "test_BLEU_3": r"test_BLEU_3\s*:\s*([\d\.e+-]+)",
        "test_BLEU_4": r"test_BLEU_4\s*:\s*([\d\.e+-]+)",
        "test_METEOR": r"test_METEOR\s*:\s*([\d\.e+-]+)",
        "test_ROUGE_L": r"test_ROUGE_L\s*:\s*([\d\.e+-]+)",
        "test_CIDER": r"test_CIDER\s*:\s*([\d\.e+-]+)"
    }

    # 检查点标记
    checkpoint_pattern = re.compile(r"Saving checkpoint:\s*(.*)")

    results = []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        checkpoint_match = checkpoint_pattern.search(line)
        if checkpoint_match:
            checkpoint_path = checkpoint_match.group(1).strip()
            epoch_data = {"saving_checkpoint": checkpoint_path}

            # 向上查找目标信息
            for j in range(i - 1, -1, -1):
                for key, pattern in pattern_dict.items():
                    if key not in epoch_data:  # 避免重复解析
                        match = re.search(pattern, lines[j])
                        if match:
                            epoch_data[key] = int(match.group(1)) if key == "epoch" else float(match.group(1))

                # 一旦找到 `epoch` 字段，就停止向上查找（假设 `epoch` 是最上面的字段）
                if "epoch" in epoch_data:
                    break

            results.append(epoch_data)

    return results

# 使用示例
log_file_path = "/Users/chenzhiwei/Downloads/ultrasound_report_Gen/Nassir-US-Report-Gen/20250319-tf-balsam.txt"
parsed_results = parse_log_file(log_file_path)

# 打印解析结果
for entry in parsed_results:
    print(entry)
print(len(parsed_results))

# 提取数据
epochs = [entry['epoch'] for entry in parsed_results]
losses = [entry['train_loss'] for entry in parsed_results]
min_loss = min(losses)
max_loss = max(losses)
normalized_losses = [(loss - min_loss) / (max_loss - min_loss) for loss in losses]

monitor_metric = 'val_BLEU_4'
val_bleu_1_scores = [entry[monitor_metric] for entry in parsed_results]

# 计算递增 marker 位置
max_so_far = float('-inf')
marker_epochs = []
marker_values = []

for i in range(len(epochs)):
    if val_bleu_1_scores[i] > max_so_far:
        max_so_far = val_bleu_1_scores[i]
        marker_epochs.append(epochs[i])
        marker_values.append(val_bleu_1_scores[i])
import matplotlib.pyplot as plt

# 绘制曲线
plt.figure(figsize=(8, 4))
plt.plot(epochs, val_bleu_1_scores, label=f"{monitor_metric}",  linewidth=2, color='#2878B5', linestyle='dashdot')
plt.plot(epochs,normalized_losses,label=f"loss",alpha=0.5,linewidth=2, color='#2878B5',linestyle='dashdot')
# 添加递增 marker
plt.scatter(marker_epochs, marker_values, color='red', marker='X', s=50, label=f"Increasing {monitor_metric}")

# 图表设置
plt.xlabel("Epoch")
plt.ylabel(f"{monitor_metric} Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sampler_train_log.pdf',format='pdf')
# 显示图像
plt.show()