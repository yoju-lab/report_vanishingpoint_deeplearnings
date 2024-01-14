import matplotlib.pyplot as plt
import json
configs_path = 'commons/configs.json'
configs = json.load(open(configs_path))

from tensorflow.keras import backend as K

def euclidean_distance_loss(y_true, y_pred):
    y_true_x, y_true_y = y_true[..., 0], y_true[..., 1]
    y_pred_x, y_pred_y = y_pred[..., 0], y_pred[..., 1]
    return K.sum(K.square(y_pred_x - y_true_x) + K.square(y_pred_y - y_true_y))

from datetime import datetime

def show_history(history, epochs, history_file_name, save_file=True):
    # Plotting MSE
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['mean_squared_error'], label='train')
    plt.plot(history.history['val_mean_squared_error'], label='validation')
    plt.title('MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Plotting MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mean_absolute_error'], label='train')
    plt.plot(history.history['val_mean_absolute_error'], label='validation')
    plt.title('MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plotting RMSE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['root_mean_squared_error'], label='train')
    plt.plot(history.history['val_root_mean_squared_error'], label='validation')
    plt.title('RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    if (save_file):
        import os
        model_history_dir = os.path.join(
            configs['datasets_dir'], configs['model_history_dir'])
        os.makedirs(model_history_dir, exist_ok=True)

        # 현재 날짜 및 시간을 얻습니다.
        now = datetime.now()

        # 원하는 형식으로 날짜와 시간을 문자열로 변환합니다.
        date_str = now.strftime("%Y%m%d%H%M%S")

        file_path = os.path.join(
            model_history_dir, history_file_name.format(epochs, date_str))
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()

