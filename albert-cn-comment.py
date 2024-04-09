import transformers
print(transformers.__version__)

import pandas as pd
import torch
from transformers import BertTokenizer, AlbertForSequenceClassification
from torch.utils.data import TensorDataset

# 加载数据集
data_dir = r'./shop_comm'
train_df = pd.read_csv(f'{data_dir}/shop_train.csv')
test_df = pd.read_csv(f'{data_dir}/shop_test.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',cache_dir=r'D:\Python\Model')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2',cache_dir=r'D:\Python\Model').to(device)

# 将数据转换为模型输入格式 max_length默认512
train_encodings = tokenizer(list(train_df['review']), max_length=64, truncation=True, padding=True)
test_encodings = tokenizer(list(test_df['review']), max_length=64, truncation=True, padding=True)

train_labels = torch.tensor(list(train_df['label'])) #0,1
test_labels = torch.tensor(list(test_df['label']))

#input_ids：文本编码，attention_mask：文本掩码，默认为1
#如果输入文本长度小于最大长度，在末尾添加填充单词，attention_mask填0
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              train_labels)
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_encodings['attention_mask']),
                             test_labels)

# 定义训练参数
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

batch_size = 64
learning_rate = 1e-5

# TODO:定义DataLoader, 训练集采用随机采样，测试集采用顺序采样

# TODO:定义优化器

from tqdm import tqdm
# 训练模型
epochs = 1
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)  #内置了交叉熵损失函数
        loss = outputs[0] #
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    train_loss /= len(train_loader.dataset)
    print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, train_loss))

torch.save(model,"./models/albert_cn_shop.pt")

# 测试模型
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

model.eval()
test_loss = 0.0
predictions, true_labels = [], []
for batch in tqdm(test_loader, desc='Testing'):
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1] #通过 outputs[1] 获取了分类结果。logits 是一个二维张量，大小为 (batch_size, num_labels)
        test_loss += loss.item() * batch_size
        logits = logits.detach().cpu().numpy() #detach不参与梯度计算
        label_ids = inputs['labels'].to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

test_loss /= len(test_loader.dataset)
print('Testing Loss: {:.4f}'.format(test_loss))

# 计算准确率和分类报告
predictions = np.concatenate(predictions, axis=0)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.concatenate(true_labels, axis=0)
accuracy = accuracy_score(true_labels, predicted_labels)
print('Accuracy:', accuracy)