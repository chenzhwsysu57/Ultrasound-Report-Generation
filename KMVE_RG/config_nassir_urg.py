import os
from yacs.config import CfgNode as CN

class Config(CN):
    def __init__(self, dataset_name):
        super().__init__()  # 调用父类构造函数
        self.data_prefix = '/home/chenzhw/ultrasound_report_gen/USData'
        self.Result_prefix = '/home/chenzhw/ultrasound_report_gen/Nassir-US-Report-Gen/Result'

        
        
        self.dataset_name = dataset_name
        
        
        
        self.image_dir = f'{self.data_prefix}/{self.dataset_name}_report'
        self.ann_path = f'{self.data_prefix}/new_{self.dataset_name}2.json'
        # Static configurations
        self.jieba_dir = f'{self.data_prefix}/key_technical_words.txt'
        self.technical_word = f'{self.data_prefix}/key_technical_words.txt'
        self.dict_pth = ' '

        # Training parameters
        self.max_seq_length_train = 150
        self.max_seq_length = 150
        self.threshold = 3
        self.num_workers = 0
        self.batch_size = 48
        self.evaluate_batch = 1

        # Model parameters
        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True
        self.d_model = 512
        self.d_ff = 512
        self.d_vf = 2048
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.1
        self.logit_layers = 1
        self.bos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0
        self.use_bn = 0
        self.drop_prob_lm = 0.5

        # Sampling and output parameters
        self.sample_n = 1
        self.output_logsoftmax = 1
        self.decoding_constraintt = 0

        # Training configurations
        self.n_gpu = 1
        self.epochs = 50
        self.save_dir = f'{self.Result_prefix}/Models'
        self.record_dir = f'{self.Result_prefix}/Records'
        self.save_period = 1
        self.monitor_mode = 'max'
        self.monitor_metric = 'BLEU_4'
        self.early_stop = 100
        self.image_type = '2d'

        # Optimizer parameters
        self.optim = 'Adam'
        self.lr_ve = 5e-5
        self.lr_ed = 1e-3
        self.weight_decay = 5e-5
        self.amsgrad = True

        # Learning rate scheduler
        self.lr_scheduler = 'StepLR'
        self.step_size = 28
        self.gamma = 0.1

        # Seed and checkpointing
        self.seed = 9233
       

        # RNN parameters
        self.embedding_vector = 300
        self.nhidden = 512
        self.nlayers = 1
        self.bidirectional = True
        self.rnn_type = 'LSTM'

        # Training settings
        self.cuda = True
        self.train_smooth_gamma3 = 10.0
        self.train_smooth_gamma2 = 5.0
        self.train_smooth_gamma1 = 4.0
        self.attn_pth = f'{self.Result_prefix}/Attn_pth'

    
    @property
    def resume(self):
        max_epoch = -1
        checkpoint_path = ''
        for i in range(1, self.epochs + 1):
            checkpoint_path = f'{self.Result_prefix}/Models/{self.dataset_name}_epoch_{i}_checkpoint.pth'
            if os.path.exists(checkpoint_path):
                max_epoch = i
        return (f'{self.Result_prefix}/Models/{self.dataset_name}_epoch_{max_epoch}_checkpoint.pth'
                if max_epoch != -1 else None)

    @property
    def distiller_num(self):
       
        distiller_mapping = {
            'Liver': 18,
            'Mammary': 18,
            'Thyroid': 5
        }
        return distiller_mapping.get(self.dataset_name, 0)



if __name__ == '__main__':
    
    for dataset in ['Liver', 'Mammary', 'Thyroid']:
        config = Config(dataset_name = dataset)
        print(config)
        # 这里调用你的 main 函数
        # main(config)