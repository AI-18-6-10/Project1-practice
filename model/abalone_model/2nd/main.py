import joblib
import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

def predict_neutron_star():
    # 1. 모델 불러오기
    model = joblib.load('saved/cat_model_v1.pkl')

    # 2. scaler 불러오기
    scaler = joblib.load('saved/minmaxscaler.pkl')
    
    # 입력값 받기
    Sex = input("Enter Sex: ")
    Length = input("Enter Length: ")
    Diameter = input("Enter Diameter: ")
    Height = input("Enter Height: ")
    Whole_weight = input("Enter Whole weight: ")
    Shucked_weight = input("Enter Shucked weight: ")
    Viscera_weight = input("Enter Viscera weight: ")
    Shell_weight = input("Enter Shell weight: ")
    Rings = input("Enter Rings(Target): ")
    Sex_F, Sex_I, Sex_M = 0, 0, 0
    
    if Sex == 'F':
        Sex_F = 1
    
    if Sex == 'M':
        Sex_I = 1
    
    if Sex == 'I':
        Sex_M = 1

    # 리스트로 변형
    input_list = [Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight, Sex_F, Sex_I, Sex_M]
    
    # 입력 데이터를 2차원 배열로 변환하여 스케일링
    input_data_2d = np.array(input_list).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_2d)

    # 예측
    prediction = model.predict(input_data_scaled)
    print(prediction, Rings)

predict_neutron_star()