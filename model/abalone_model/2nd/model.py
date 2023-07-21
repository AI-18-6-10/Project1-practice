import numpy as np
import pandas as pd
import catboost

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from catboost import CatBoostRegressor

# Scaler 종류
# MinMaxScaler - 0, StandardScaler - 1, , MaxAbsScaler - 2, RobustScaler - 3, Normalizer - 4

def catboostregressor(df, train_size = 0.8, Scaler = None):
    
    # catboostregressor의 특성상 별도의 인코딩 과정이 필요하지 않음

    # 원핫 인코딩
    df = pd.get_dummies(df,columns=['Sex'])

    # 데이터 분리
    X = df.drop('Rings', axis=1)
    y = df['Rings'].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state = 0)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    if Scaler != None:
        # scaling
        scale = Scaler
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

    # 학습
    model = CatBoostRegressor()
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model