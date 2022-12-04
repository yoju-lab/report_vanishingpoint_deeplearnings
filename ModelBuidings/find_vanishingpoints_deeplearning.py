
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
import tensorflow as tf
from random import randint, choice
import pathlib
import os

data_sets_path = os.path.abspath('datasets/')
data_dir = data_sets_path + '/datasets/'

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

# count = 0
# for images, labels in train_ds.take(1):
#     count += 1
#     print(count, ', ', labels)
# pass


# 데이터 불러오기
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# ResNet50 가져오기
# from tensorflow.keras.applications.resnet50 import ResNet50
# model_res = ResNet50(include_top=False, pooling='avg', weights='imagenet')

model_res = InceptionResNetV2(
    include_top=False, pooling='avg', weights='imagenet')

# resnet50 가중치 프리징
model_res.trainable = False

# inputs = tf.keras.Input(shape=IMG_SHAPE)
# input layers for grayscale
inputs = tf.keras.Input(shape=(img_height, img_width, 1), name="input_01")
x = tf.keras.layers.Concatenate(name="input_02")([inputs, inputs, inputs])
x = tf.cast(x, tf.float32)
x = tf.keras.applications.resnet50.preprocess_input(x)
# x = tf.keras.applications.InceptionResNetV2.preprocess_input(x)

x = model_res(x, training=False)
# add regressions layers
x = Flatten(name="output_01")(x)
outputs = Dense(2, name="output_02")(x)
model_res = tf.keras.Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
# optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum= 0.9, nesterov = True)
optimizer = tf.keras.optimizers.Adam(lr=0.001)
# loss = tf.keras.losses.MeanSquaredError()
loss = tf.keras.losses.MeanAbsoluteError()
model_res.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_res.summary()

# 모델 fitting
history = model_res.fit(train_ds, epochs=10)
count = 0
for images, labels in train_ds.take(1):
    count += 1
    print(count, ', ', labels)
    predictions = model_res.predict(images)
    print('predictions : {}'.format(predictions))


pass
