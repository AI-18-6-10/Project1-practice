import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

def predict_neutron_star(input_data):
    # 1. 모델 불러오기
    with open('ML_model2_pickle/lg_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # scaler 불러오기
    with open('ML_model2_pickle/lg_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    # 입력 데이터를 2차원 배열로 변환하여 스케일링
    input_data_2d = np.array(input_data).reshape(1, -1)

    # 입력 데이터 스케일링
    input_data_scaled = scaler.transform(input_data_2d)

    # 예측
    prediction = model.predict(input_data_scaled)

    # 4. 결과 반환 (0이면 중성자별 아님, 1이면 중성자별)
    return prediction[0]