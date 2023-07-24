import numpy as np
import pandas as pd
import catboost

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# Scaler 종류
# MinMaxScaler - 0, StandardScaler - 1, , MaxAbsScaler - 2, RobustScaler - 3, Normalizer - 4

def linearregressor(df, train_size = 0.8, Scaler = None):

    # 성별 원핫 인코딩
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

    model=LinearRegression() # initialzing the model
    model.fit(X_train, y_train)   

    return X_train, X_test, y_train, y_test, model

def grandientboostregressor(df, train_size = 0.8, Scaler = None):
    
    # 성별 원핫 인코딩
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

    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model


def randomforestregressor(df, train_size = 0.8, Scaler = None):

    # 성별 원핫 인코딩
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
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model


def catboostregressor(df, train_size = 0.8, Scaler = None):
    
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
    model = CatBoostRegressor(random_state = 0)
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model


def lgbmregressor(df, train_size = 0.8, Scaler = None):

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
    model = LGBMRegressor(random_state = 0)
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model


def xgboostregressor(df, train_size = 0.8, Scaler = None):

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
    model = XGBRegressor(random_state = 0)
    model.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, model