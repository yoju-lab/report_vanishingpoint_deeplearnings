from matplotlib.cbook import flatten
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.applications as applications
import tensorflow.keras.preprocessing.image as image_processing
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import sys
sys.path.append('./')

# Load vanishing point data
data = pd.read_csv('datasets/any_informations/01122202_filtering_grayimage_by_vanishing_point.csv')
data = data[data['filtered'] == True]

# Load images and coordinates
img_height = 200
img_width = 200
img_channel = 1
images = [image_processing.load_img('datasets/preprocessings/' + fname
                                    , target_size=(img_height, img_width, img_channel)
                                    , color_mode='grayscale') for fname in data['file_name']]
# 이미지를 numpy 배열로 변환하고 채널 수 조정
images = np.array([np.reshape(image_processing.img_to_array(img), (img_height, img_width, img_channel)) for img in images])
coordinates = data[['vanishing_point_x', 'vanishing_point_y']].values

# Split the data into training and validation datasets
train_images, val_images, train_coordinates, val_coordinates = train_test_split(images, coordinates
                                                                                , test_size=0.2, random_state=42)

# Create the model
# Load pre-trained InceptionResNetV2
base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False)

# Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
predictions = layers.Dense(2)(x)

# Define the model
model = models.Model(inputs=base_model.input, outputs=predictions)

# Set layers to trainable
for layer in base_model.layers:
    layer.trainable = True

# Define a function to convert 1-channel image to 3-channel image
def convert_to_3_channels(x):
    return tf.repeat(x, 3, axis=-1)

# Add a Lambda layer to do the conversion
input_shape = (200, 200, 1)  # Your input shape
inputs = layers.Input(input_shape)
x = layers.Lambda(convert_to_3_channels)(inputs)
outputs = model(x)

# Define a new model that does the conversion
model = models.Model(inputs, outputs)

# Set layers to trainable
for layer in base_model.layers:
    layer.trainable = True

# Compile the model
from commons.utils_models import euclidean_distance_loss
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
model.compile(optimizer=optimizers.Adam(), loss=euclidean_distance_loss, 
              metrics=[MeanSquaredError(), MeanAbsoluteError(), RootMeanSquaredError()])

from commons.utils_models import show_history
# Train the model
BATCH_SIZE = 6
# 모델 fitting
epochs_list = [20, 40, 60]
# epochs_list = [2, 4, 6]

import json
configs_path = 'commons/configs.json'
configs = json.load(open(configs_path))
import os
# model weights 저장
model_saves_dir = os.path.join(
    configs['datasets_dir'], configs['model_saves_dir'])
os.makedirs(model_saves_dir, exist_ok=True)

for epochs in epochs_list:
    history = model.fit(train_images, train_coordinates, epochs=epochs, shuffle=True
            , validation_data=(val_images, val_coordinates))

    history_file_name = 'tf_model_InceptionResNetV2_{}_{}.png'

    show_history(history, epochs, history_file_name, save_file=True)
    # count = 0
    # for images, labels in train_images.take(1):
    #     count += 1
    #     print(count, ', ', labels)
    #     predictions = model.predict(images)
    #     print('predictions : {}'.format(
    #         label_standardscaler.inverse_transform(predictions)))

    # # show_history(history, epochs)
    # model_saves_path = os.path.join(model_saves_dir, tf_model_name)
    # model.save(model_saves_path.format(epochs))

