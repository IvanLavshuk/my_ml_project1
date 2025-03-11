# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Загрузка данных
df = pd.read_csv("train.csv")


# Проверка пропущенных значений
print(df.isnull().sum())

# Заполнение пропущенных значений
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Нормализация данных
scaler = MinMaxScaler()
df["Age"] = scaler.fit_transform(df[["Age"]])

# One-hot encoding
df = pd.get_dummies(df, columns=["Embarked", "Sex"], drop_first=True)

# Сохранение обработанных данных
df.to_csv("processed_titanic.csv", index=False)