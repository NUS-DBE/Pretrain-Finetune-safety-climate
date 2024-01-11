import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from model import AccidentClassifier


# 准备数据
def prepare_data(df):
    # 假设第一列是ID，最后一列是我们的标签
    df=df.drop(['company_ID'],axis=1)
    balanced_df = balance_classes_with_smote(df, df.columns[-1])


    features = balanced_df.iloc[:, 0:-1].values  # 假设第一列是ID，最后一列是我们的标签
    labels = balanced_df.iloc[:, -1].values
    # print(sum(labels))

    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 转换为PyTorch张量
    features_tensor = torch.tensor(features).float()
    labels_tensor = torch.tensor(labels).long()  # 将labels转换为正确的形状

    # 创建TensorDataset
    dataset = TensorDataset(features_tensor, labels_tensor)
    return dataset

#
import pandas as pd
from sklearn.utils import resample
def balance_classes(df, target_column):
    # 分离多数和少数类别
    df_majority = df[df[target_column] == df[target_column].mode()[0]]
    df_minority = df[df[target_column] != df[target_column].mode()[0]]

    # 计算多数类别与少数类别之间的样本数差
    num_to_resample = df_majority.shape[0] - df_minority.shape[0]

    # 对少数类别进行过采样
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # 采样替换
                                     n_samples=num_to_resample,  # 匹配多数类别的数量
                                     random_state=123)  # 可复制的结果

    # 合并多数类别和过采样后的少数类别DataFrame
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled

from imblearn.over_sampling import SMOTE
import pandas as pd
def balance_classes_with_smote(df, target_column):
    # 分离特征和目标
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    # 初始化SMOTE对象
    smote = SMOTE(random_state=123)

    # 进行过采样
    X_res, y_res = smote.fit_resample(X, y)

    # 将过采样的数据转换回DataFrame
    df_resampled = pd.DataFrame(X_res, columns=df.drop(target_column, axis=1).columns)
    df_resampled[target_column] = y_res

    return df_resampled



# 分割数据集

def split_dataset(dataset, test_size=0.3):
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
