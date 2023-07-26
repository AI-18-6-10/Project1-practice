import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler


# 데이터셋을 불러온 뒤 이상치를 제거하는 함수
# 원데이터의 경로를 csv_path에 입력하면 전처리된 데이터프레임을 반환
# default Train:Valid:Test = 6:2:2
# split random_state 필요하면 설정하기

def load_prepro_data(csv_path='Regression_data.csv'):
    # original 데이터셋 불러오기
    df = pd.read_csv(csv_path)

    # 이상치 전처리 
    df_origin = df.copy()
    df_origin.iloc[:, 1:8] *= 200

    selected_rows = df_origin[df_origin['Diameter'] == 46.0]
    selected_rows_removed = selected_rows.drop(3996)
    df_origin.loc[3996, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Diameter'] == 71.0]
    selected_rows_removed = selected_rows.drop(2051)
    df_origin.loc[2051, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Diameter'] == 68.0]
    selected_rows_removed = selected_rows.drop(1257)
    df_origin.loc[1257, 'Height'] = selected_rows_removed['Height'].mean()

    selected_rows = df_origin[df_origin['Length'] == 141.0]
    selected_rows_removed = selected_rows.drop(1417)
    df_origin.loc[1417, 'Height'] = selected_rows_removed['Height'].mean()

    condition = (df_origin['Whole weight'] > 12.0) & (df_origin['Whole weight'] < 14.0)
    selected_rows = df_origin[condition]
    selected_rows_removed = selected_rows.drop(3522)
    df_origin.loc[3522, 'Height'] = selected_rows_removed['Height'].mean()

    df_clean = df_origin.copy()
    df_clean.iloc[:, 1:8] /= 200
    
    return df_clean


def target_split(df):

    # 원핫 인코딩
    df = pd.get_dummies(df,columns=['Sex'])

    # 데이터 분리
    X = df.drop('Rings', axis=1)
    y = df['Rings'].astype('float32')

    # 데이터 분할
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    print(X_train.shape, X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape)

    # scaling
    scale = MinMaxScaler()
    X_train = scale.fit_transform(X_train)
    X_valid = scale.transform(X_valid)
    X_test = scale.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_valid, X_test, y_train, y_valid, y_test