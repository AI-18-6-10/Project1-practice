import star_MLmodel
import numpy as np

# 예측할 입력 데이터
input_data = [35.179688, 26.179797, 5.374898, 39.850834, 15.742475, 52.624313, 3.477182, 10.839420]

# 결과 예측
result = star_MLmodel.predict_neutron_star(input_data)

# 결과 출력
print("Result:", result)