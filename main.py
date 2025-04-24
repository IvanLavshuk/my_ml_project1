import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder

df = pd.read_csv("train.csv")
print("Первые 20 строк датасета:")
print(df.head(20))
print("\nИнформация о датасете:")
print(df.info())
print("\nТипы данных в столбцах:")
print(df.dtypes)
print("\nНазвания столбцов:")
print(df.columns.tolist())

train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 42)
print("\nРазмер обучающей выборки:", train_df.shape)
print("Размер тестовой выборки:", test_df.shape)


nan_matrix_test = test_df.isnull()
nan_matrix = train_df.isnull()
missing_values_count = nan_matrix.sum()
missing_values_count_test = nan_matrix_test.sum()
print("Количество пропущенных значений в каждом столбце:")
print(missing_values_count)
print(missing_values_count_test)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

cat_cols = ["Cabin", "Embarked"]
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
print("\nКоличество пропущенных значений после заполнения:")
print(train_df.isnull().sum())
print(test_df.isnull().sum())
# Нормализация данных
# Минимально-максимальное масштабирование для числовых столбцов
scaler = MinMaxScaler()
train_df['Age'] = scaler.fit_transform(train_df[['Age']])
train_df['Fare'] = scaler.fit_transform(train_df[['Fare']])
test_df['Age'] = scaler.fit_transform(test_df[['Age']])
test_df['Fare'] = scaler.fit_transform(test_df[['Fare']])
print("\nДанные после нормализации:")
print(train_df[['Age', 'Fare']].head())


# Преобразование категориальных данных
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Cabin'] = train_df['Cabin'].fillna('Unknown').str[0]
test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df['Cabin'] = test_df['Cabin'].fillna('Unknown').str[0]
pd.set_option("display.max_columns",None)
print(train_df.head(20));
train_df= pd.get_dummies(train_df, columns=['Cabin'], drop_first=True)
test_df= pd.get_dummies(test_df, columns=['Cabin'], drop_first=True)
train_df = train_df.drop('Name', axis='columns')
train_df = train_df.drop('Ticket', axis='columns')
test_df = test_df.drop('Name', axis='columns')
test_df = test_df.drop('Ticket', axis='columns')

pd.set_option("display.max_columns",None)
print(train_df.head(20));
print(test_df.head(20));
train_df.to_csv("train_titanic.csv", index=False)
test_df.to_csv("test_titanic.csv", index=False)
print("\nОбработанные данные сохранены в файл 'train_titanic.csv'.")
