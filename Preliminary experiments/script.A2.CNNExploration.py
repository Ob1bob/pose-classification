import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from options import exploration_options


def plot_image(img):
    """
    This function displays an image using the matplotlib library. It takes a 3D array
    representing the image's RGB pixel values and plots the image using pyplot.

    Args:
        img (numpy.ndarray):    The image represented as a 3D array of RGB pixel values.
    """
    pyplot.imshow(img)
    pyplot.show()


def create_directory(path):
    """
    This function creates a directory if it does not exist.

    Args:
        path (str):     The directory path to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_session(gpu_fraction=0.333):
    """
    Create and return a TensorFlow session with customizable GPU memory allocation settings
    based on the specified GPU fraction. The fraction determines the portion of available
    GPU memory that TensorFlow can use for a single process (session).

    Args:
        gpu_fraction (float, optional):     The fraction of GPU memory to allocate (default is 0.333).

    Returns:
        tf.Session: A TensorFlow session with the specified GPU memory allocation settings.
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  # tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == "__main__":
    opt = exploration_options()

    # Sub-directories for models and logs
    MODEL_DIR = os.path.join(opt.log_dir, opt.dataset_name, "models")
    LOG_DIR = os.path.join(opt.log_dir, opt.dataset_name, "logs")
    # create sub-directories
    create_directory(MODEL_DIR)
    create_directory(LOG_DIR)

    # set keras tensorflow session to use fraction of GPU
    ktf.set_session(get_session())

    csv_train_file = os.path.join(opt.data_dir, opt.dataset_name + "-TRAIN.csv")
    csv_test_file = os.path.join(opt.data_dir, opt.dataset_name + "-TEST.csv")
    # import data from CSV file
    df_train = pd.read_csv(csv_train_file)
    df_test = pd.read_csv(csv_test_file)

    # TRAIN samples
    train_cls_column = df_train.iloc[:, 0].values  # all rows, first column
    train_cls = []
    for val in train_cls_column:  # create a 2D array of class values
        train_cls.append([val])
    y_train = np.array(train_cls)
    # TRAIN images
    train_imgs = []
    for index, row in df_train.iloc[:, 1:].iterrows():
        arr_3D = np.reshape(np.array(row), (32, 32, 3))  # reconfigure 1D array of values as 3D array
        train_imgs.append(arr_3D)
    x_train = np.array(train_imgs)

    # TEST samples
    test_cls_column = df_test.iloc[:, 0].values  # all rows, first column
    test_cls = []
    for val in test_cls_column:  # create a 2D array of class values
        test_cls.append([val])
    y_test = np.array(test_cls)
    # TEST images
    test_imgs = []
    for index, row in df_test.iloc[:, 1:].iterrows():
        arr_3D = np.reshape(np.array(row), (32, 32, 3))  # reconfigure 1D array of values as 3D array
        test_imgs.append(arr_3D)
    x_test = np.array(test_imgs)

    # User feedback on data stats
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # debug
    # plot_image(x_train[2])

    # Convert type to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # TRAINING #
    # Data pre-processing
    # create generator (1.0/255.0 = 0.003921568627451)
    datagen = ImageDataGenerator(rescale=1.0/255.0)  # normalise
    # prepare an iterators to scale images
    train_iterator = datagen.flow(x_train, y_train, batch_size=opt.batch_size)
    test_iterator = datagen.flow(x_test, y_test, batch_size=opt.batch_size)
    # confirm data rescaling with user feedback
    batchX, batchy = train_iterator.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # range of architecture configuration values:
    dense_layers = [1, 2]
    layer_sizes = [32, 64, 128, 256]
    conv_layers = [1, 2, 3] # 0 = 1

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:

                # label model based on layer configuration
                MODEL_NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                print(MODEL_NAME)

                # compile CNN architecture
                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), input_shape=x_train.shape[1:]))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # model.add(Dropout(0.25))

                for l in range(conv_layer-1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    # model.add(Dropout(0.25))

                model.add(Flatten())

                for d in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                tensorboard = TensorBoard(log_dir=LOG_DIR + "/{}".format(MODEL_NAME))

                model.compile(loss='binary_crossentropy', # loss='categorical_crossentropy' assuming more than 2 classes
                              optimizer='adam',  # adam optimiser
                              metrics=['accuracy']
                              )

                # commence training
                model.fit_generator(train_iterator,
                          #batch_size=opt.batch_size,
                          steps_per_epoch=round(x_train.shape[0] / opt.batch_size),
                          epochs=opt.epochs,
                          #validation_split=0.3,
                          validation_data=(x_test, y_test),
                          shuffle=True,
                          callbacks=[tensorboard]
                          )

                # Save model and weights
                model_path = os.path.join(MODEL_DIR, MODEL_NAME)
                model.save(model_path)
                print('Saved trained model at %s ' % model_path)

                # Score trained model.
                scores = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=1)
                print('Test loss:', scores[0])
                print('Test accuracy:', scores[1])