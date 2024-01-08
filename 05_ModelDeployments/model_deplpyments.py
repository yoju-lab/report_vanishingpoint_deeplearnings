
import tensorflow as tf
import os
import json
configs_path = 'commons/configs.json'
configs = json.load(open(configs_path))

if __name__ == "__main__":
    import json
    configs_path = 'commons/configs.json'
    configs = json.load(open(configs_path))

    data_dir = os.path.join(configs['datasets_dir'], configs['features_dir'])
    import sys
    sys.path.append('./')
    from commons.utils import get_lables

    validation_labels_list = get_lables(data_dir)

    # Normalization using standard scaler
    from sklearn.preprocessing import StandardScaler
    label_standardscaler = StandardScaler()
    label_standardscaler.fit(validation_labels_list)
    # label_standardscaler.scale_, label_standardscaler.min_
    validation_labels = label_standardscaler.transform(
        validation_labels_list).tolist()

    train_dir = os.path.join(data_dir, 'training')

    BATCH_SIZE = 6
    img_height = 200
    img_width = 200
    IMG_SHAPE = (img_height, img_width, 1)

    # validation dataset from directory
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=validation_labels,
        color_mode="grayscale",
        label_mode='int',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=BATCH_SIZE)

    model_saves_dir = os.path.join(
        configs['datasets_dir'], configs['model_saves_dir'])

    # predict the vanishing point on the image with load models
    for file in os.listdir(model_saves_dir):
        file_path = os.path.join(os.path.join(model_saves_dir, file))

        if os.path.isfile(file_path):
            load_model = tf.keras.models.load_model(file_path)
            for images, labels in validation_ds.take(3):
                predictions = load_model.predict(images)
                print('file name : {} \n predictions : \n {} \n labels : \n {}'.format(
                    file, label_standardscaler.inverse_transform(predictions), labels))
