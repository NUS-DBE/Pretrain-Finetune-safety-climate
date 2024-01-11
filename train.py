from rfpimp import *
import matplotlib.pyplot as plt
import pandas as pd

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy


df = pd.read_csv('Trucking data with company (with items).csv')
df = df.copy()[['company_ID','OSC1c1','OSC1c3','OSC1c8','OSC1c10','OSC1c15','OSC1c16','OSC1c17','OSC1c18','OSC1c22','OSC1c24','OSC1c29',
                'OSC1c32','OSC1c34','OSC2c2r','OSC2c4r','OSC2c11r','OSC2c33r','OSC3c13r','OSC3c21r','OSC3c23r','GSC1g2','GSC1g3','GSC1g8r',
                'GSC1g9','GSC1g11','GSC1g14','GSC1g15','GSC1g17','GSC1g18','GSC1g19','GSC1g22','GSC1g23','GSC1g25','GSC1g6r','GSC1g7r','GSC1g10r',
                'GSC1g12','GSC1g13','GSC1g16r','GSC1g1','GSC1g5r','GSC1g21r','Total accident 2009']]
# 假设df是您的DataFrame，并且您想要二值化最后一列
# 使用df.columns[-1]获取最后一列的列名，然后直接使用列名进行赋值
df[df.columns[-1]] = (df[df.columns[-1]] > 0).astype(int)

# grouped = df.groupby('company_ID').size()
# print(grouped)

df_company_1 = df[df['company_ID'] == 1]
df_company_2 = df[df['company_ID'] == 2]
df_company_3 = df[df['company_ID'] == 3]
df_company_4 = df[df['company_ID'] == 4]
df_company_5 = df[df['company_ID'] == 5]
df_company_6 = df[df['company_ID'] == 6]
df_company_7 = df[df['company_ID'] == 7]
df_company_8 = df[df['company_ID'] == 8]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import AccidentClassifier
from dataset import prepare_data,split_dataset
from eval import evaluate_model
from torch.optim.lr_scheduler import StepLR
batch_size=4
# 训练函数
def train_model(model, train_dataset, val_dataset, batch_size=batch_size, epochs=1):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # criterion = nn.BCEWithLogitsLoss()  # Assuming binary classification
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    # 初始化学习率调整器，每30个epochs减少学习率
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    acc =-100
    acc_scores=[]

    model.train()
    for epoch in range(epochs):
        # Training phase
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            metrics = evaluate_model(model, test_dataset, batch_size=1)
            print(metrics)
            acc_scores.append(metrics['accuracy'])
        avg_val_loss = val_loss / len(val_loader)

        # If the model improved, save it!
        if acc < metrics['accuracy']:
            acc = metrics['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'c1_model.pth')  # Save the model
            print('save model')
        print(f'Epoch {epoch + 1}/{epochs} Training Loss: {loss.item()} Validation Loss: {avg_val_loss}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model,acc_scores

# 提取特征和标签
features = df_company_1.iloc[:, 1:-1]  # 假设最后一列是标签
labels = df_company_1.iloc[:, -1]
from sklearn.model_selection import train_test_split

# 分离特征和标签
# X = df_company_1.iloc[:, :-1]  # 所有行，除了最后一列
# y = df_company_1.iloc[:, -1]   # 所有行，只有最后一列

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
# clf = DecisionTreeClassifier(random_state=42)
#
# # 训练模型
# clf.fit(X_train, y_train)
# # 在测试集上进行预测
# y_pred = clf.predict(X_test)
#
# # 打印分类报告
# print(classification_report(y_test, y_pred))
# exit(0)

# 创建模型
num_features = features.shape[1]  # 特征数量
model = AccidentClassifier(num_features) #,hidden_dim= 512, n_layers = 6, n_heads = 8, dropout=0.3

# 准备数据集
dataset = prepare_data(df_company_1)
# 划分训练集和测试集
train_dataset, test_dataset = split_dataset(dataset, test_size=0.3)



# 预训练模型
model, acc_scores =train_model(model, train_dataset,test_dataset, epochs=100)

plt.plot(acc_scores)
plt.title('Accuracy Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('ACC Score')
plt.show()
# 在测试集上评估模型
model.eval()
metrics = evaluate_model(model, test_dataset,batch_size=1)
print(metrics)

# # 微调模型（使用公司2的数据）
# dataset_finetune = prepare_data(df_company_2)
# train_dataset_finetune, test_dataset_finetune = split_dataset(dataset_finetune, test_size=0.3)
# train_model(model, train_dataset_finetune, epochs=5)


# print(1)







