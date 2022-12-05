
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
import tensorflow as tf
from random import randint, choice
import pathlib
import os

import matplotlib.pyplot as plt


def show_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
    import json
    configs_path = 'commons/configs.json'
    configs = json.load(open(configs_path))

    data_sets_path = os.path.abspath('datasets/')
    data_dir = data_sets_path + '/features/'

    # count with temporary labels
    count_dir = pathlib.Path(data_dir)
    image_count = len(list(count_dir.glob('*/*')))
    print(data_dir, image_count)
    train_labels = []

    for _ in range(image_count):
        x = choice([randint(9, 15), randint(21, 27), randint(1, 5)])
        y = choice([randint(1, 5), randint(9, 15), randint(21, 27)])
        train_labels.append([x, y])
        # train_labels.append([y])

    pass

    train_dir = os.path.join(data_dir, 'training')

    BATCH_SIZE = 6
    img_height = 200
    img_width = 200
    IMG_SHAPE = (img_height, img_width, 1)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=train_labels,
        color_mode="grayscale",
        label_mode='int',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=BATCH_SIZE)

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=train_labels,
        color_mode="grayscale",
        label_mode='int',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=BATCH_SIZE)

    # count = 0
    # for images, labels in train_ds.take(1):
    #     count += 1
    #     print(count, ', ', labels)
    # pass

    # 데이터 불러오기
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # ResNet50 가져오기
    model_finetuning = ResNet50(
        include_top=False, pooling='avg', weights='imagenet')

    # model_finetuning = InceptionResNetV2(
    #     include_top=False, pooling='avg', weights='imagenet')

    tf_model_name = 'tf_model_resnet50_{}.h5'
    # tf_model_name = 'tf_model_InceptionResNetV2_{}.h5'

    # resnet50 가중치 프리징
    model_finetuning.trainable = False

    # inputs = tf.keras.Input(shape=IMG_SHAPE)
    # input layers for grayscale
    inputs = tf.keras.Input(shape=(img_height, img_width, 1), name="input_01")
    x = tf.keras.layers.Concatenate(name="input_02")([inputs, inputs, inputs])
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    # x = tf.keras.applications.InceptionResNetV2.preprocess_input(x)

    x = model_finetuning(x, training=False)
    # add regressions layers
    x = Flatten(name="output_01")(x)
    outputs = Dense(2, name="output_02")(x)
    model_finetuning = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 모델 컴파일
    # optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum= 0.9, nesterov = True)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    # loss = tf.keras.losses.MeanSquaredError()
    loss = tf.keras.losses.MeanAbsoluteError()
    model_finetuning.compile(optimizer=optimizer, loss=loss,
                             metrics=['mse', 'accuracy'])
    model_finetuning.summary()

    # 모델 fitting
    epochs_list = [20, 50]

    # model weights 저장
    model_saves_dir = os.path.join(
        configs['datasets_dir'], configs['model_saves_dir'])
    os.makedirs(model_saves_dir, exist_ok=True)

    for epochs in epochs_list:
        history = model_finetuning.fit(train_ds, epochs=10,
                                       shuffle=True, validation_data=validation_ds)

        show_history(history, 10)
        count = 0
        for images, labels in train_ds.take(1):
            count += 1
            print(count, ', ', labels)
            predictions = model_finetuning.predict(images)
            print('predictions : {}'.format(predictions))

        # show_history(history, epochs)
        model_saves_path = os.path.join(model_saves_dir, tf_model_name)
        model_finetuning.save(model_saves_path.format(epochs))

    pass
