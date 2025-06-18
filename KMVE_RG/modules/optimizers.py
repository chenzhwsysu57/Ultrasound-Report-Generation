import torch


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    cls_params = list(map(id, model.classification_layers.parameters()))
    text_params = list(map(id, model.text_encoder.parameters()))
    non_tf = ve_params + cls_params + text_params
    
    ed_params = filter(lambda x: id(x) not in non_tf, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': model.classification_layers.parameters(), 'lr': args.lr_ve},
         {'params': model.text_encoder.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed} # encoder decoder
         ],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer

def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
