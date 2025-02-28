import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # tokenizer并行化
# 设置随机种子
torch.manual_seed(42)
# 设置默认的 CUDA 设备为 1
torch.cuda.set_device(1)
batch_size = 1024
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=40):
        super(PositionalEncoding, self).__init__()
        
        # Create a tensor to hold positional encodings
        pe = torch.zeros(batch_size, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine and cosine functions to alternate dimensions
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer so it's not treated as a parameter
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return x

class transformer_encode(nn.Module):
    def __init__(self, vocal_size, num_classes, embedding_dim=256, nhead=4, num_layers=3,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeding = nn.Embedding(vocal_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, nhead, batch_first=True), num_layers)
        self.linear1 = nn.Linear(40 * embedding_dim, 40 * embedding_dim * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(40 * embedding_dim * 2, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.squeeze()
        x = self.embeding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        # x = self.softmax(x)
        # return torch.argmax(x, dim=1)

class OurDataset(Dataset):
    def __init__(self, texts, labels, df_code):
        self.texts = texts
        self.labels = labels
        self.df_code = df_code
    
    def __getitem__(self, index):
        mask = self.df_code['主要编码'] == self.labels[index]
        y = self.df_code.index[mask].tolist()
        # return x['input_ids'], torch.tensor(y)
        return self.texts[index], torch.tensor(y)
    
    def __len__(self):
        return len(self.labels)   
    

file_path='/gjh/znbianma/znbianma_data.xlsx'
# 为不同的工作表定义需要读取的列
columns_dict = {
    '编码字典表': ['主要编码', '疾病名称'],
    '2021': ['诊断描述', '标准编码'],
    '2022': ['诊断描述', '标准编码'],
    '2023': ['诊断描述', '标准编码']
}
# 使用一个字典来存储每个工作表对应的DataFrame
dfs = pd.read_excel(file_path, sheet_name=list(columns_dict.keys()))
# 根据定义的列筛选每个DataFrame的内容
df_code = dfs['编码字典表'][columns_dict['编码字典表']]
all_data = pd.concat([dfs[sheet][columns_dict[sheet]] for sheet in ['2021', '2022', '2023']], ignore_index=True)
# all_data = pd.concat([t_dict['2021'], t_dict['2022'], t_dict['2023']], ignore_index=True)
all_data.dropna()
dataset = OurDataset(all_data['诊断描述'].astype(str).tolist(), all_data['标准编码'].astype(str).tolist(), df_code)

num_workers = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/gjh/znbianma/bert-base-chinese')
tokenizer.bos_token = '<BOS>'
tokenizer.eos_token = '<EOS>'

vocal_size = 21128
# 实例化模型
model = transformer_encode(vocal_size, num_classes=34713)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_list = []
best_acc = -float('inf')
best_loss = float('inf')

from tqdm import tqdm

epochs = 15
print("training")
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
        x = tokenizer(x,padding="max_length",return_tensors='pt',truncation=True, max_length=40)
        x = x['input_ids']
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # 清空梯度
        outputs = model(x)
        y = y.squeeze()
        loss = criterion(outputs, y)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()
        loss_list.append(loss.item())
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), '/gjh/znbianma/transformer_wieghts_final.pth')

import matplotlib.pyplot as plt
# 绘制损失变化图
plt.figure(figsize=(15, 7))
plt.plot(loss_list, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Over Batches')
plt.legend()

# 保存图表为文件
plt.savefig('/gjh/znbianma/Transformer_training_loss.png')
