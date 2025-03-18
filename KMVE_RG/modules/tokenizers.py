import jieba
import json
from collections import Counter



class Tokenizer(object):
    
    def __init__(self, args):
        
        jieba.load_userdict(args.technical_word)
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        # self.dataset_name = args.dataset_name  #
        # self.clean_report = self.clean_report
        # self.ann = json.loads(open(self.ann_path, 'r').read())
        print(f"reading ann from {self.ann_path}")
        
        self.ann = json.loads(open(self.ann_path, 'r', encoding="utf_8_sig").read())
        self.dict_pth = args.dict_pth
        self.tokens = None
        self.token2idx, self.idx2token = self.create_vocabulary()
        
    def create_vocabulary(self):
        if self.dict_pth != ' ':
            word_dict = json.loads(open(self.dict_pth, 'r', encoding="utf_8_sig").read())
            return word_dict[0], word_dict[1]
        else:
            total_tokens = []
            split_list = ['train', 'test', 'val']
            for split in split_list:
                for example in self.ann[split]:
                    tokens = list(jieba.lcut(example['finding']))
                    for token in tokens:
                        total_tokens.append(token)
            counter = Counter(total_tokens)
            self.tokens = list(set(total_tokens))
            print(f"\033[1;35m len(self.tokens) = \033[0m{len(self.tokens)}")
            vocab = [k for k, v in counter.items()] + ['<unk>']
            token2idx, idx2token = {}, {}
            for idx, token in enumerate(vocab):
                token2idx[token] = idx + 1
                idx2token[idx + 1] = token
            return token2idx, idx2token


    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)


    def __call__(self, report):
        tokens = list(jieba.cut(report))
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_list(self, ids):
        txt = []
        for i, idx in enumerate(ids):
            if idx > 0:
                txt.append(self.idx2token[idx])
            else:txt.append('<start/end>')

        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def decode_batch_list(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode_list(ids))
        return out




if __name__=="__main__":
    import argparse

    args = argparse.Namespace(
        technical_word='/home/chenzhw/ultrasound_report_gen/USData/key_technical_words.txt',
        ann_path='/home/chenzhw/ultrasound_report_gen/USData/new_all2.json',
        threshold=10,
        dict_pth=' '
    )
    tokenizer = Tokenizer(args)

    # 单个文本编码
    encoded_text = tokenizer("肝脏形态饱满，包膜光滑，实质回声细密增强，门静脉系统显示欠清晰，肝肾回声对比增强。肝内外胆管未见扩张。门静脉主干内径正常范围。胆囊大小形态如常，壁不厚，光滑，腔内未见明显异常回声。胰腺大小形态如常，实质回声均匀，胰管不宽，内未见明确占位性病变。脾脏大小形态如常，实质回声均匀，内未见明显占位性病变。")
    print(encoded_text)

    # 批量文本编码
    texts = ["肝脏形态饱满，包膜光滑，实质回声细密增强，门静脉系统显示欠清晰，肝肾回声对比增强。肝内外胆管未见扩张。门静脉主干内径正常范围。胆囊大小形态如常，壁不厚，光滑，腔内未见明显异常回声。胰腺大小形态如常，实质回声均匀，胰管不宽，内未见明确占位性病变。脾脏大小形态如常，实质回声均匀，内未见明显占位性病变。", "甲状腺大小形态如常，腺体回声均匀，未见明确占位性病变，CDFI示腺体内未见异常血流信号。双侧颈部扫查可探及低回声结节，左侧可见多个，大者约_2DS_，右侧可见一个，大小约_2DS_，边界清晰，形态规整，可见“淋巴门”结构，CDFI示可探及血流信号。"]
    encoded_batch = tokenizer.decode_batch(texts)
    print(encoded_batch)

    # 解码单个文本
    decoded_text = tokenizer.decode(encoded_text)
    print(decoded_text)

    # 解码批量文本
    decoded_batch = tokenizer.decode_batch(encoded_batch)
    print(decoded_batch)
