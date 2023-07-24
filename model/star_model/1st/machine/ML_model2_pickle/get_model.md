# 모델 언피클링 (불러오기)
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)