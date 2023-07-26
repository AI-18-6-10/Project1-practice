import numpy as np
import pandas

from sklearn import metrics

# Custom Accuracy 생성
def get_Accuracy(y, y_pred):
    mdiff = (np.abs((y_pred - y) / y)).mean()
    acc = 1 - mdiff

    return acc

# Custom Accuracy 추가된 Metrics 출력
def get_metrics(model, X, y, cv_n=3):
    y_pred=model.predict(X)
    
    r2 = metrics.r2_score(y, y_pred)
    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    acc = get_Accuracy(y, y_pred)

    print('r2_score                 :', r2)
    print('Mean Absolute Error      :', mae)
    print('Mean Squared Error       :', mse)
    print('Root Mean Squared Error  :', rmse)
    print('Accuracy                 :', acc)

    return r2, mae, mse, rmse, acc

def total_score(model, X_train, X_test, y_train, y_test):

    print('훈련성능')
    get_metrics(model, X_train, y_train)
    print('-'*100)
    print('평가성능')
    get_metrics(model, X_test, y_test)