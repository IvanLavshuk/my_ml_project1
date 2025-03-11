# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
# Загрузка данных
df = pd.read_csv("train.csv")

#Данные из Dataset
print("Первые 20 строк датасета:")
print(df.head(20))
print("\nИнформация о датасете:")
print(df.info())
print("\nТипы данных в столбцах:")
print(df.dtypes)
print("\nНазвания столбцов:")
print(df.columns.tolist())

#3 Количество пропущенных значений для каждого столбца в Datasets
nan_matrix = df.isnull()
missing_values_count = nan_matrix.sum()
print("Количество пропущенных значений в каждом столбце:")
print(missing_values_count)

# 4. Заполнение пропущенных значений
# Определение числовых и категориальных столбцов
num_cols = ["Age", "Fare"]  # Числовые столбцы
cat_cols = ["Cabin", "Embarked"]  # Категориальные столбцы
# Заполнение пропущенных значений в числовых столбцах медианой
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
# Заполнение пропущенных значений в категориальных столбцах модой
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# Проверка, что пропущенные значения заполнены
print("\nКоличество пропущенных значений после заполнения:")
print(df.isnull().sum())

# 5. Нормализация данных
# Минимально-максимальное масштабирование для числовых столбцов
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
df['Fare'] = scaler.fit_transform(df[['Fare']])

print("\nДанные после нормализации:")
print(df[['Age', 'Fare']].head())

# 6. Преобразование категориальных данных
# One-Hot Encoding для категориальных столбцов
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
le = LabelEncoder()
df['Cabin'] = le.fit_transform(df['Cabin'])


# 7. Удаление столбцов
# Удаление столбца 'Name'
df_without_name = df.drop('Name', axis='columns')
print("\nДанные после удаления столбца 'Name':")
print(df_without_name.head())

# Удаление столбцов 'Age' и 'Fare'
df_without_age_and_fare = df.drop(['Age', 'Fare'], axis='columns')
print("\nДанные после удаления столбцов 'Age' и 'Fare':")
print(df_without_age_and_fare.head())

# 8. Выбор столбцов по типу данных
# Выбор только числовых столбцов
numeric_df = df.select_dtypes(include='number')
print("\nЧисловые столбцы:")
print(numeric_df.head())

# Выбор нечисловых столбцов
not_numeric_df = df.select_dtypes(exclude='number')
print("\nНечисловые столбцы:")
print(not_numeric_df.head())

# 9. Добавление новых столбцов
# Пример добавления нового столбца 'VIP' (предположим, что у нас есть данные в df1)
# df['VIP'] = df1['VIP_Status']

# Пример добавления двух новых столбцов 'Deck' и 'Side' (предположим, что у нас есть данные в df2)
# df[['Deck', 'Side']] = df2[['CabinDeck', 'CabinSide']]

# 10. Разделение данных на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print("\nРазмер обучающей выборки:", train_df.shape)
print("Размер тестовой выборки:", test_df.shape)

# 11. Сохранение обработанных данных
train_df.to_csv("train_titanic.csv", index=False)
test_df.to_csv("test_titanic.csv", index=False)

print("\nОбработанные данные сохранены в файлы 'train_titanic.csv' и 'test_titanic.csv'.")
