import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def predict_neutron_star():
    # 1. 모델 불러오기
    with open('ML_model2_pickle/lg_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # scaler 불러오기
    with open('ML_model2_pickle/lg_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    # 입력값 받기
    Mean_i = input("Enter Mean of the integrated profile: ")
    SD_i = input("Enter Standard deviation of the integrated profile: ")
    EK_i = input("Enter Excess kurtosis of the integrated profile: ")
    S_i = input("Enter Skewness of the integrated profile: ")
    Mean_curve = input("Enter Mean of the DM-SNR curve: ")
    SD_curve = input("Enter Standard deviation of the DM-SNR curve: ")
    EK_curve = input("Enter Excess kurtosis of the DM-SNR curve: ")
    S_curve = input("Enter Skewness of the DM-SNR curve: ")

    # 리스트로 변형
    input_list = [Mean_i, SD_i, EK_i, S_i, Mean_curve,SD_curve, EK_curve, S_curve]
    
    # 입력 데이터를 2차원 배열로 변환하여 스케일링
    input_data_2d = np.array(input_list).reshape(1, -1)

    # 입력 데이터 스케일링
    input_data_scaled = scaler.transform(input_data_2d)

    # 예측
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 0:
        return '중성자 별이 아닙니다.'
    else:
        return '중성자 별입니다.'