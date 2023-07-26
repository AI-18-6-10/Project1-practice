# -*- coding: utf-8 -*-
"""binary_load

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fuoL8PU-F3KuXNYjX1Imn9kt1S1qQVnZ
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN



def binary_load_dataset(csv_loca, TEST_SIZE, VAL_SIZE, RANDOM_STATE):
  """
  이진용 데이터 로드법
  총 두가지가 나온다
  하나는 업스케일 (XX_ad)
  하나는 원본
  """
  # 글로벌로 만들어서 저장하기
  global df, X, y, X_train, X_val, X_test, y_train, y_val, y_test

  # 데이터 불러오기
  df = pd.read_csv(csv_loca)

  # 데이터 컬럼 간소화
  df.columns = ['Mean_i', 'SD_i', 'EK_i', 'S_i', 'Mean_curve','SD_curve', 'EK_curve', 'S_curve', 'Class']

  # 데이터 나누기
  X = df.drop('Class',axis=1)
  y = df['Class']

  # 업스케일링
  # 스모트(SMOTE) 대신에 아다신(ADASYN) 사용된 이유는 좀 더 랜덤하게 업스케일링이 되게 하게 위해 사용
  ada = ADASYN(random_state = RANDOM_STATE)
  X_ad, y_ad = ada.fit_resample(X, y)

  # 원본 훈련, 검증, 테스트 셋 나누기
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify= y, random_state=RANDOM_STATE)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, stratify= y_train, random_state=RANDOM_STATE)

  # 원본 표준화
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.fit_transform(X_val)
  X_test = scaler.transform(X_test)

  # 업스케일링 훈련, 검증, 테스트 셋 나누기
  X_train_ad, X_test_ad, y_train_ad, y_test_ad = train_test_split(X_ad, y_ad, test_size=TEST_SIZE, random_state=RANDOM_STATE)
  X_train_ad, X_val_ad, y_train_ad, y_val_ad = train_test_split(X_train_ad, y_train_ad, test_size=VAL_SIZE, random_state=RANDOM_STATE)

  # 업스케일링 표준화
  scaler = MinMaxScaler()
  X_train_ad = scaler.fit_transform(X_train_ad)
  X_val_ad = scaler.fit_transform(X_val_ad)
  
  return X_train, X_val, X_test, y_train, y_val, y_test