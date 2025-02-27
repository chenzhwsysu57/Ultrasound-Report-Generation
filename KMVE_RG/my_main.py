import numpy as np
import torch

from KMVE_RG.models.AllOrgan import AllOrgan
from modules.MyTrainer import TFTrainer as Trainer
from modules.dataloaders import MyDataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer


def main(args):
    # fix random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)
    print(f'tokenizer size: {tokenizer.get_vocab_size()}')
    train_dataloader = MyDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = MyDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = MyDataLoader(args, tokenizer, split='test', shuffle=False)

    model = AllOrgan(args, tokenizer)
    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()


if __name__ == '__main__':
    # for dataset in ['Liver', 'Thyroid', 'Mammary']:
    #     print(f"\033[1;35mtrain for {dataset}...\033[0m")
    #     from config_nassir_urg import Config
    #     config = Config(dataset_name = dataset)
    #     main(config)
    import argparse
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--dataset_name', type=str, default='all', help='dataset name')
    parser.add_argument('--result', type=str, default='TF_only', help='result path')
    cmd_line_args = parser.parse_args()
    from config_nassir_urg import Config
    config = Config(
        dataset_name = cmd_line_args.dataset_name, 
        result = cmd_line_args.result
                    )
    main(config)