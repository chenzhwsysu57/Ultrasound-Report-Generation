import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    model = MyModel(args, tokenizer)
    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--dataset_name', type=str, default='all', help='dataset name')
    parser.add_argument('--result', type=str, default='debug', help='folder save result. This would be a subfolder created under the folder Result/')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size to train. defaults to 5')
    parser.add_argument('--debug', type=int, default=-1, help='the index end of your dataloader, defaults to -1 means load all, 0:-1')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulate step for grad.')
    parser.add_argument('--decoderonly', type=str, default='False', help='use decoder only model.')
    cmd_line_args = parser.parse_args()
    
    from config_urg import Config
    if cmd_line_args.dataset_name == "all":
        from KMVE_RG.models.AllOrgan import AllOrgan as MyModel
        from modules.MyTrainer import TFTrainer as Trainer 
    else:
        from KMVE_RG.models.SGF import SGF as MyModel
        from modules.MyTrainer import Trainer
    config = Config(**vars(cmd_line_args))
    
    print(config)
    main(config)