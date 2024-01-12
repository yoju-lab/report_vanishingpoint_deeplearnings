import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras.applications as applications
import tensorflow.keras.preprocessing.image as image_processing
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

import matplotlib.pyplot as plt
import json
configs_path = 'commons/configs.json'
configs = json.load(open(configs_path))

def show_history(history, epochs, history_file_name, save_file=True):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    if (save_file):
        import os
        model_history_dir = os.path.join(
            configs['datasets_dir'], configs['model_history_dir'])
        os.makedirs(model_history_dir, exist_ok=True)

        file_path = os.path.join(
            model_history_dir, history_file_name.format(epochs))
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close()

# Load vanishing point data
data = pd.read_csv('datasets/any_informations/01122202_filtering_grayimage_by_vanishing_point.csv')
data = data[data['filtered'] == True]

BATCH_SIZE = 6
img_height = 200
img_width = 200
img_channel = 3

# Load images and coordinates
import os
images = [image_processing.load_img('datasets/preprocessings/' + fname, target_size=(img_height, img_width)) for fname in data['file_name']]
images = np.array([image_processing.img_to_array(img) for img in images])
coordinates = data[['vanishing_point_x', 'vanishing_point_y']].values

# Split the data into training and validation datasets
train_images, val_images, train_coordinates, val_coordinates = train_test_split(images, coordinates, test_size=0.2, random_state=42)

# Create the model
base_model = applications.InceptionResNetV2(include_top=False, weights='imagenet'
                                            , input_shape=(img_height, img_width, img_channel), pooling='avg')
output_layer = layers.Dense(2)
model = models.Sequential([base_model, output_layer])

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss='mse'
              , metrics=['mse', 'accuracy'])

# Train the model
epochs = 4
history = model.fit(train_images, train_coordinates, epochs=epochs, shuffle=True
          , validation_data=(val_images, val_coordinates))

history_file_name = 'tf_model_InceptionResNetV2_{}.png'
show_history(history, epochs, history_file_name, save_file=True)

