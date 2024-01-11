from rfpimp import *
import matplotlib.pyplot as plt
import pandas as pd
from dataset import balance_classes_with_smote
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
from sklearn.preprocessing import StandardScaler

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


from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import AccidentClassifier
from dataset import prepare_data,split_dataset
from eval import evaluate_model

batch_size=4
# 训练函数
def train_model(model, train_loader, val_loader,test_dataset, batch_size=batch_size, epochs=1,lr=0.0001):


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    acc =-0
    model.train()
    for epoch in range(epochs):
        # Training phase
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            # for inputs, labels in val_loader:
            #     outputs = model(inputs)
            #     loss = criterion(outputs, labels)
            #     val_loss += loss.item()
            metrics = evaluate_model(model, test_dataset, batch_size=1)
            # print(metrics)
        if acc < metrics['accuracy']:
            acc = metrics['accuracy']


    return model,acc

# companys={'1': df_company_1,'2':df_company_2,'3':df_company_3,'4':df_company_4,'5':df_company_5,'6':df_company_6,'7':df_company_7,'8':df_company_8}
companys={'2':df_company_2,'3':df_company_3,'4':df_company_4,'5':df_company_5,'7':df_company_7}
for cstr in companys.keys():
    df_company_fine = companys[cstr]


    pre_model = AccidentClassifier(42)  # ,hidden_dim= 512, n_layers = 6, n_heads = 8, dropout=0.3
    no_pre_model = AccidentClassifier(42)  # ,hidden_dim= 512, n_layers = 6, n_heads = 8, dropout=0.3

    pre_model.load_state_dict(torch.load('0.5993.pth'))  # 预训练模型

    # 准备数据集
    df_company_fine = df_company_fine.drop(['company_ID'], axis=1)
    balanced_df = balance_classes_with_smote(df_company_fine, df_company_fine.columns[-1])
    df_0 = balanced_df[balanced_df['Total accident 2009'] == 0]
    df_1 = balanced_df[balanced_df['Total accident 2009'] == 1]

    sample_0_test = df_0.sample(n=min(int(0.4 * len(df_0)), 100))
    sample_1_test = df_1.sample(n=min(int(0.4 * len(df_0)), 100))

    # 获取 sample_0_test 中的索引
    indices_to_remove = sample_0_test.index
    df_0 = df_0.drop(indices_to_remove)
    indices_to_remove = sample_1_test.index
    df_1 = df_1.drop(indices_to_remove)

    new_balanced_df_test = pd.concat([sample_0_test, sample_1_test])
    new_balanced_df_test_features = new_balanced_df_test.iloc[:, 0:-1].values  # 假设第一列是ID，最后一列是我们的标签
    new_balanced_df_test_labels = new_balanced_df_test.iloc[:, -1].values
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(new_balanced_df_test_features)

    # 转换为PyTorch张量
    features_tensor = torch.tensor(features).float()
    labels_tensor = torch.tensor(new_balanced_df_test_labels).long()  # 将labels转换为正确的形状

    # 创建TensorDataset
    test_dataset = TensorDataset(features_tensor, labels_tensor)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    ct = 0
    for i in range(10):
        metrics = evaluate_model(copy.deepcopy(pre_model), test_dataset, batch_size=1)
        ct = ct + metrics['accuracy']
    print(ct / 10)
    p1 = ct / 10
    ct = 0
    for i in range(10):
        metrics = evaluate_model(copy.deepcopy(no_pre_model), test_dataset, batch_size=1)
        ct = ct + metrics['accuracy']
    print(ct / 10)
    # pre_train_acc.append(metrics['accuracy'])
    metrics = evaluate_model(copy.deepcopy(no_pre_model), test_dataset, batch_size=1)
    print(metrics)
    print(len(df_0))
    p2 = ct / 10

    while True:
        pre_train_acc = [p1]
        no_pre_train_acc = [p2]
        for nt in range(8, len(df_0), 8):
            # 从每个类别中随机选择100个样本
            print(nt)
            sample_0 = df_0.sample(n=nt)
            sample_1 = df_1.sample(n=nt)
            # 合并样本
            new_balanced_df = pd.concat([sample_0, sample_1])

            new_features = new_balanced_df.iloc[:, 0:-1].values  # 假设第一列是ID，最后一列是我们的标签
            new_labels = new_balanced_df.iloc[:, -1].values
            # 标准化特征
            features = scaler.fit_transform(new_features)

            features_tensor = torch.tensor(features).float()
            labels_tensor = torch.tensor(new_labels).long()  # 将labels转换为正确的形状

            # 创建TensorDataset
            dataset = TensorDataset(features_tensor, labels_tensor)

            # 划分训练集和测试集
            # train_dataset, test_dataset = split_dataset(dataset, test_size=0.4)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            pn = 10
            # val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

            for pre in ['no pre-train', 'pre-train']:
                if pre == 'pre-train':
                    pass
                    # for i, layer in enumerate(pre_model.children()):
                    #     if i < 2:  # 假设我们想冻结前三层
                    #         for param in layer.parameters():
                    #             param.requires_grad = False
                    #     else:
                    #         break  # 一旦处理完前三层，就跳出循环
                    # print(1)
                    if nt < 20:
                        epoch_num = 20
                    else:
                        epoch_num = 20
                    acc_sc = 0
                    model_copy = copy.deepcopy(pre_model)
                    for ii in range(pn):
                        _, acc_scores = train_model(model_copy, train_loader, val_loader, test_dataset,
                                                    epochs=epoch_num,
                                                    lr=0.001)
                        acc_sc = acc_sc + acc_scores
                    pre_train_acc.append(acc_sc / pn)
                else:
                    # print(2)
                    acc_sc2 = 0
                    model_copy_no = copy.deepcopy(no_pre_model)
                    for ii in range(pn):
                        _, acc_scores = train_model(model_copy_no, train_loader, val_loader, test_dataset, epochs=20,
                                                    lr=0.001)
                        acc_sc2 = acc_sc2 + acc_scores
                    no_pre_train_acc.append(acc_sc2 / pn)
        print(1)
        break

    # pre_train_acc=[0.6,0.7151898734177216,0.7215189873417721,0.7974683544303798,0.7215189873417721,0.7468354430379747,0.810126582278481,0.8164556962025317,0.8164556962025317,0.7911392405063291]
    # no_pre_train_acc=[0.5,0.689873417721519,0.6708860759493671,0.7151898734177216,0.7341772151898734,0.7531645569620253,0.7468354430379747,0.8037974683544303,0.8354430379746836,0.8227848101265823]
    #
    # pre_train_acc=[0.6,0.7151898734177216,0.7215189873417721,0.7974683544303798,0.7974683544303798,0.7974683544303798,0.810126582278481,0.8164556962025317,0.8164556962025317]
    # no_pre_train_acc=[0.5,0.689873417721519,0.689873417721519,0.7151898734177216,0.7341772151898734,0.7531645569620253,0.7531645569620253,0.8037974683544303,0.8354430379746836]

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(df_0), 8), pre_train_acc, label='Pre-trained')
    plt.plot(range(0, len(df_0), 8), no_pre_train_acc, label='No Pre-trained')
    plt.xlabel('Sample Size')
    plt.ylabel('Precision')
    plt.title('Precision vs. Sample Size_company1-' + cstr)
    plt.legend()
    plt.show()
    print(pre_train_acc)
    print(no_pre_train_acc)











