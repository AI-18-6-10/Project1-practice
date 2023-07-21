import pandas as pd

# 데이터셋을 불러온 뒤 이상치를 제거하는 함수
# 원데이터의 경로를 csv_path에 입력하면 전처리된 데이터프레임을 반환
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