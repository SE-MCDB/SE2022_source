file = open("computer_paper.txt", encoding='utf-8')
#如果这个是batch_size的话，那可能太小了，我看网上说这个不宜太小，也不宜太大。另外如果在GPU上训练，设置成2的幂次，可能设置成64或128比较好？
batch_size = 128
#全部数据量的训练次数
index = 0

# 定义存储
list_ids = []
list_title = []
list_keywords = []

# 读取文件
line = file.readline()
while line:
    # 转化为json
    dict_str = eval(line)
    list_ids.append(dict_str['id'])
    list_title.append(dict_str['title'])
    list_keywords.append(";".join(item for item in dict_str['keywords']))
    
    line = file.readline()

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class ContrastiveSciBERT(nn.Module):
    def __init__(self, out_dim, tau, device='cuda'):
        """⽤于对⽐学习的SciBERT模型
        :param out_dim: int 输出特征维数
        :param tau: float 温度参数τ
        :param device: torch.device, optional 默认为CPU
        """
        super().__init__()
        self.tau = tau
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
        self.linear = nn.Linear(self.model.config.hidden_size, out_dim).to(device)


    def get_embeds(self, texts, max_length=64):
        """将⽂本编码为向量
        :param texts: List[str] 输⼊⽂本列表，⻓度为N


        # Press the green button in the gutter to run the script.
        if __name__ == '__main__':
            print("success")
        :param max_length: int, optional padding最⼤⻓度，默认为64
        :return: tensor(N, d_out)
        """
        encoded = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
        ).to(self.device)
        return self.linear(self.model(**encoded).pooler_output)

    def calc_sim(self, texts_a, texts_b):
        """计算两组⽂本的相似度
        :param texts_a: List[str] 输⼊⽂本A列表，⻓度为N
        :param texts_b: List[str] 输⼊⽂本B列表，⻓度为N
        :return: tensor(N, N) 相似度矩阵，S[i, j] = cos(a[i], b[j]) / τ
        """
        embeds_a = self.get_embeds(texts_a) # (N, d_out)
        embeds_b = self.get_embeds(texts_b) # (N, d_out)
        embeds_a = embeds_a / embeds_a.norm(dim=1, keepdim=True)
        embeds_b = embeds_b / embeds_b.norm(dim=1, keepdim=True)
        return embeds_a @ embeds_b.t() / self.tau

    def forward(self, texts_a, texts_b):
        """计算两组⽂本的对⽐损失（直接返回损失）
        :param texts_a: List[str] 输⼊⽂本A列表，⻓度为N
        :param texts_b: List[str] 输⼊⽂本B列表，⻓度为N
        :return: tensor(N, N), float A对B的相似度矩阵，对⽐损失
        """
        # logits_ab等价于预测概率，对⽐损失等价于交叉熵损失
        logits_ab = self.calc_sim(texts_a, texts_b)
        logits_ba = logits_ab.t()
        labels = torch.arange(len(texts_a), device=self.device)
        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ba, labels)
        loss = (loss_ab + loss_ba) / 2
        return loss

from torch.utils.data import Dataset, DataLoader, TensorDataset

class MyDataset(Dataset):
    def __init__(self, titles, keywords):
        self.titles = titles
        self.keywords = keywords
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        keywords = self.keywords[idx]
        return title, keywords
    
    def __len__(self):
        if len(self.titles) != len(self.keywords):
            raise Error
        return len(self.titles) 

train_dataset = MyDataset(list_title, list_keywords)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

def train():
    global model, optimizer, batch_size
    model.train()
    epoch = 2
    total_iter = len(train_loader)
    for get_time in range(epoch):
        iter_num = 0
        total_loss = 0
        for iters, batch in enumerate(train_loader):
            iter_num += 1
            loss = model.forward(list(batch[0]), list(batch[1]))
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with open(file='./log.txt', mode='a', encoding='utf-8') as f:  
                f.write("--------------------------------------------\n")
                f.write("iters: {}\n".format(iter_num))
                f.write("total_loss: {}\n".format(total_loss))
                f.write("Average training loss: %.4f\n"% (total_loss/(batch_size * iter_num)))
            if iter_num % 10000 == 0:
                torch.save(model.state_dict(), 'model_test.pt')

if __name__ == '__main__':
    model = ContrastiveSciBERT(out_dim=128, tau=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("start training")
    train()
    print('end')
    torch.save(model.state_dict(), 'model.pt')
