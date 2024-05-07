import tensorflow as tf
import cv2
import os
import numpy as np

class Config:
    CASIA1 = "CASIA1"
    CASIA2 = "CASIA2"
    autotune = tf.data.experimental.AUTOTUNE
    epochs = 30
    batch_size = 32
    lr = 1e-3
    name = "xception"
    n_labels = 2
    image_size = (224, 224)
    decay = 1e-6
    momentum = 0.95
    nesterov = False




preprocess = {
    "densenet": tf.keras.applications.densenet.preprocess_input,
    "xception": tf.keras.applications.xception.preprocess_input,
    "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
    "effecientnetb7": tf.keras.applications.efficientnet.preprocess_input,
    "vgg19": tf.keras.applications.vgg19.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
    "nasnetlarge": tf.keras.applications.nasnet.preprocess_input,
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet": tf.keras.applications.resnet.preprocess_input,
}


def process_image(file_path):
    QUALITY = 95
    SCALE = 15
    # Generate the image
    orig = cv2.imread(file_path.numpy().decode("utf-8"))
    # orig = cv2.imread(file_path)
    orig = cv2.resize(orig, (224, 224), interpolation=cv2.INTER_AREA)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # Augmentation
    #
    buffer = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    # print(buffer)
    # print(type(buffer))
    # get it from the buffer and decode it to a numpy array
    compressed_img = cv2.imdecode(np.frombuffer(buffer[1], np.uint8), cv2.IMREAD_COLOR)

    # Compute the absolute difference
    diff = SCALE * (cv2.absdiff(orig, compressed_img))
    img = preprocess[Config.name](diff)
    return img, 0.0

def process_image_manipulation():
    model = tf.keras.models.load_model("./manupulation_detection/model.h5")
    data_ds = tf.data.Dataset.list_files("")


    # data_ds = file.concatenate(tif_files)

    tensor_preprocess = lambda x: tf.py_function(
        process_image, [x], [tf.float32, tf.float32]
    )

    n_data = data_ds.cardinality().numpy()
    n_val = int(0.2 * n_data)
    data_ds = data_ds.shuffle(n_data)

    pred_ds = (
        data_ds.skip(n_val)
        .map(tensor_preprocess, num_parallel_calls=Config.autotune)
        .batch(Config.batch_size)
    )
    val_ds_x = []
    val_ds_y = []
    for _, (val_x_batch, val_y_batch) in enumerate(pred_ds):
        for val_x, val_y in zip(val_x_batch, val_y_batch):
            val_ds_x.append(val_x)
            val_ds_y.append(val_y)

    pred_data = (
        tf.convert_to_tensor(val_ds_x, dtype=tf.float32),
        tf.convert_to_tensor(val_ds_y, dtype=tf.float32),
    )


    test_predict = model.predict(pred_data[0], batch_size=Config.batch_size)
    fake_score = 1 - test_predict[0][0]
    return fake_score