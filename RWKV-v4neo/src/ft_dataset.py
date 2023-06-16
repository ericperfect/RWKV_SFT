import json

from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from torch.utils.data import Dataset, DataLoader
import torch
class FtDataset(Dataset):

    def __init__(self, data_path, ctx_len):
        data = []
        with open(data_path) as f:
            for item in f:
                data.append(json.loads(item))
        data_size = len(data)
        rank_zero_info('data has %d' % (data_size))
        self.data = data
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
        # self.tokenizer.eos_token = '<|endoftext|>'
        self.eod_id = 0
        self.pad_id = 1
        # self.tokenizer.eod_token = '<|endoftext|>'
        # self.tokenizer.pad_token='<|padding|>'

        self.ctx_len = ctx_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa_pair = self.data[idx]
        question = qa_pair['instruction']
        answer = qa_pair['output']

        input_ids_question = self.tokenizer.encode(question, add_special_tokens=False)
        input_ids_answer = self.tokenizer.encode(answer, add_special_tokens=False)
        input_ids_temp = input_ids_question + input_ids_answer + [0]
        labels_temp = [-100] * len(input_ids_question) + input_ids_answer + [0]
        input_size = len(input_ids_temp)

        # # padding
        # if input_size < self.ctx_len:
        #     input_ids_temp = input_ids_temp + (self.ctx_len - input_size) * [-100]
        #     labels_temp = labels_temp + (self.ctx_len - input_size) * [-100]
        # else:
        #     input_ids_temp = input_ids_temp[:self.ctx_len]
        #     labels_temp = labels_temp[:self.ctx_len]
        # # input_ids_temp = input_ids_temp[:self.ctx_len]
        # # labels_temp = labels_temp[:self.ctx_len]
        # input_ids = input_ids_temp
        # label_ids = labels_temp
        # #弃用
        # mask = [0]


        # # padding
        if input_size < self.ctx_len:
            input_ids_temp = input_ids_temp + (self.ctx_len + 1 - input_size) * [1]
        else:
            input_ids_temp = input_ids_temp[:self.ctx_len+1]
        input_ids = input_ids_temp[:-1]
        label_ids = input_ids_temp[1:]

        if self.ctx_len > len(input_ids_question):
            mask = [0] * len(input_ids_question) + [1] * (len(input_ids_answer) + 1) + [0] * abs(
                self.ctx_len - input_size)
            mask = mask[:self.ctx_len]
        else:
            mask = [0] * self.ctx_len


        # dynamic max_len会溢出？
        # input_ids_temp = input_ids_temp[:self.ctx_len+1]
        # input_ids = input_ids_temp[:-1]
        # label_ids = input_ids_temp[1:]
        #
        # if len(input_ids_temp) > self.ctx_len:
        #     mask = [0] * len(input_ids_question) + [1] * (len(input_ids_answer) + 1)
        #     mask = mask[:self.ctx_len]
        # else:
        #     mask = [0] * len(input_ids_question) + [1] * (len(input_ids_answer) + 1)
        #     mask = mask[:-1]
        #
        # assert len(input_ids) == len(mask)
        #
        mask = torch.tensor(mask, dtype=torch.bfloat16)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return input_ids, label_ids, mask

def collate_fn(batch):
    input_ids_list, label_list, mask_list = zip(*batch)
    input_ids_list = pad_sequence(input_ids_list, batch_first=True, padding_value=1)
    label_list = pad_sequence(label_list, batch_first=True, padding_value=1)
    mask_list = pad_sequence(mask_list, batch_first=True, padding_value=0)
    print(input_ids_list[0].shape)
    return input_ids_list, label_list, mask_list
if __name__ == '__main__':
    dataset = FtDataset("/home/eric/RWKV-LM-LoRA/data/traffic_gen/traffic_gen_test_rwkv_for_mydataset .jsonl", 5000)
    # dataset[0]
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for i in dataloader:
        print(i)