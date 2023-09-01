import os
import gc
import sys
import shutil
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import sklearn.utils.class_weight as class_weight
import sklearn.metrics as metrics
import configparser


def create_directories(paths):
    """
    This function checks each path in the input list and creates the directories
    if they are not already present. It ensures that the specified paths are available
    for storing files or other data.

    Args:
        paths (list):   List of directory paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def plot_confusion_matrix(model_name, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function creates a visualizstion of a confusion matrix.
    The confusion matrix assesses the performance of the classification model and its plot can help
    identify how well the model is classifying different classes and highlight common misclassifications.

    Args:
        model_name (str):           The name of the model being evaluated.
        cm (numpy.ndarray):         The confusion matrix (2D array) of predicted vs. true labels.
        classes (list):             List of class labels for labeling the axes.
        normalize (bool, optional): Whether to normalize confusion matrix values (default is False).
        title (str, optional):      Title for the plot (default is 'Confusion matrix').
        cmap (matplotlib.colors.Colormap, optional): colour map for the plot (default is plt.cm.Blues).
    """
    image_name = model_name
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        image_name += '-confusion_NORM.png'
    else:
        image_name += '-confusion_COUNT.png'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, image_name))


def absolute_file_paths(directory):
    """
    Walks through a directory and its subdirectories to return absolute paths to each of its contained files.

    Args:
        directory (str): The directory path to start generating file paths from.

    Returns:
        str: Absolute file paths for each file found in the directory and its subdirectories.
    """
    for dir_path, _, file_names in os.walk(directory):
        for f in file_names:
            yield os.path.abspath(os.path.join(dir_path, f))


def copy_images(df, src_dir_name, tmp_dir):
    """
    This function copies image files from a source directory (generated augmentations)
    into a temp directory for each training/validation fold.

    Args:
        df (pandas.DataFrame):  A DataFrame containing information about image files.
        src_dir (str):          The source directory containing the image files.
        tmp_dri (str):          The temporary directory where the files will be copied.

    Returns:
        str: The destination directory where the files have been copied.
    """
    destination_directory = os.path.join(tmp_dir, src_dir_name)
    print("Copying [{}] files to {}...".format(src_dir_name, destination_directory))

    # remove all files from previous fold in the temp directory
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)

    # create directory for files for the current fold
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # create sub-directories for each class
    for c in set(list(df['class'])):
        if not os.path.exists(destination_directory + '/' + c):
            os.makedirs(destination_directory + '/' + c)

    # copy files for the given fold from the source directory to the temporary directory
    for i, row in df.iterrows():
        try:
            path_to = "{}/{}".format(destination_directory, row['class'])
            shutil.copy(row['filename'], path_to)
        except Exception:
            print("[Error] could not copy from {}: {}".format(dir, row['filename']))

    return destination_directory


def create_model_VGG(input_shape, output_shape):
    """
    This function defines a CNN architecture for image classification tasks, inspired by the VGGNet architecture.
    The model consists of convolutional layers with batch normalization, ReLU activation, and max-pooling.

    Args:
        input_shape (tuple):    The shape of input images (height, width, channels).
        output_shape (int):     The number of classes for classification.

    Returns:
        keras.models.Sequential: A compiled Keras sequential model.
    """
    # MODEL PROPERTIES
    filters = 16
    pad = "valid"
    model = Sequential()

    # INPUT LAYERS
    count_conv = 1
    model.add(Conv2D(input_shape=input_shape, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding=pad, name="C" + str(count_conv)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # HIDDEN LAYERS
    count_conv += 1
    model.add(Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding=pad, name="C" + str(count_conv)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # POOL
    model.add(MaxPooling2D(pool_size=(2, 2), name="P1"))

    for i in range(1, 4):  # start at 1, highest based on adhikari is 4  (MANUALLY ADJUST THIS FOR SHALLOWER CNN)
        # input shape should look like (180x320x3)
        filters = filters*2

        count_conv += 1
        model.add(Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding=pad, name="C" + str(count_conv)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        count_conv += 1
        model.add(Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding=pad, name="C" + str(count_conv)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # POOL
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="P" + str(i+1)))

    # OUTPUT LAYERS
    model.add(Flatten())
    # FC 1
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # FC 2 - output layer
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))

    return model


def compile_model(batch_size, image_size, model_name, train_dir, valid_dir):
    """
    This function prepares and trains a NN model for image classification
    using the specified training and validation directories. It compiles the model with
    a specified optimizer, loss function, and metrics, and trains it using data generators.

    Args:
        batch_size (int):   Batch size for training and validation.
        image_size (tuple): Target image size (height, width) for input images.
        model_name (str):   Name to save the trained model.
        train_dir (str):    Path to the training data directory.
        valid_dir (str):    Path to the validation data directory.

    Returns:
        keras.models.Sequential: The trained Keras sequential model.
        float: Loss value after evaluation on validation data.
        float: Accuracy value after evaluation on validation data.)
    """

    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory(train_dir,
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 color_mode='rgb',
                                                 target_size=image_size)
    valid_it = valid_datagen.flow_from_directory(valid_dir,
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 color_mode='rgb',
                                                 target_size=image_size)

    batch_images, batch_classes = train_it.next()
    input_shape = batch_images.shape[1:]
    output_shape = len(train_it.class_indices.keys())

    opt = SGD(lr=0.015, momentum=0.95)
    model = create_model_VGG(input_shape=input_shape, output_shape=output_shape)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # training
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_it.classes), train_it.classes)
    tensorboard = TensorBoard(log_dir=logs_dir + "/{}".format(model_name))
    earlystop = EarlyStopping(patience=30)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.1,
                                                min_lr=0.0001)

    # fit model
    history = model.fit_generator(train_it,
                                  steps_per_epoch=len(train_it),
                                  validation_data=valid_it,
                                  validation_steps=len(valid_it),
                                  epochs=epochs,
                                  callbacks=[tensorboard, earlystop, learning_rate_reduction],
                                  verbose=1,
                                  class_weight=class_weights)

    # save model
    model_path = os.path.join(models_dir, model_name)
    model.save(model_path)

    # evaluate model
    loss, acc = model.evaluate_generator(valid_it,
                                         steps=len(valid_it),
                                         verbose=1)
    print('Validation Accuracy: %.3f' % (acc * 100.0))

    return model, loss, acc


def predict(model, batch_size, image_size, test_dir):
    """
    This function uses a trained NN model to predict class labels for images
    in a specified test directory. It returns the list of class labels, predicted classes,
    and true classes for further analysis and evaluation.

    Args:
        model (keras.models.Sequential):    The trained Keras sequential model.
        batch_size (int):       Batch size for prediction.
        image_size (tuple):     dimensions of input image size (height, width).
        test_dir (str):         Path to the test data directory.

    Returns:
        list: List of class labels for mapping predicted and true class indices.
        numpy.ndarray: Predicted class indices for each test image.
        numpy.ndarray: True class indices for each test image.
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_it = test_datagen.flow_from_directory(test_dir,
                                               class_mode='categorical',
                                               batch_size=1,
                                               color_mode='rgb',
                                               target_size=image_size,
                                               shuffle=False)

    predictions = model.predict_generator(test_it, steps=len(test_it))
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_it.classes

    return list(test_it.class_indices.keys()), predicted_classes, true_classes


def interpret_metrics(conf_matrix):
    """
    This function takes a confusion matrix as input and calculates several performance
    metrics such as accuracy, sensitivity, specificity, precision, and more.

    Args:
        conf_matrix (numpy.ndarray):    The confusion matrix as a 2D array.

    Returns:
        tuple: A tuple containing multiple metrics.
    """
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Overall accuracy for each class
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = np.nan_to_num(TP / (TP + FN), nan=-999)
    # Specificity or true negative rate
    TNR = np.nan_to_num(TN / (TN + FP), nan=-999)
    # Precision or positive predictive value
    PPV = np.nan_to_num(TP / (TP + FP), nan=-999)
    # Negative predictive value
    NPV = np.nan_to_num(TN / (TN + FN), nan=-999)
    # Fall out or false positive rate
    FPR = np.nan_to_num(FP / (FP + TN), nan=-999)
    # False negative rate
    FNR = np.nan_to_num(FN / (TP + FN), nan=-999)
    # False discovery rate
    FDR = np.nan_to_num(FP / (TP + FP), nan=-999)

    return ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR, TP, TN, FP, FN


def get_script_arguments():
    """
    This function reads the command-line arguments and retrieves and
    validates the dataset ID provided by the user.

    Raises:
        SystemExit: If the script is not executed with the correct arguments or if
                    the dataset ID is not within the range of 0 to 5.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_id>")
        sys.exit(1)

    try:
        arg_val = int(sys.argv[1])
        if arg_val < 0 or arg_val > 5:
            print("[Error: Script Argument] Dataset ID should be between 0 and 5.")
            sys.exit(1)
        else:
            return arg_val
    except ValueError:
        print("[Error: Script Argument] Invalid value. ID should be a number.")


if __name__ == "__main__":
    # read arguments from config file
    config = configparser.ConfigParser()
    config.read('parameters.conf')
    # names of each dataset (0_Baseline, 1_JointColour, 2_RadialGradient, etc.)
    data_types = list(config.get('Global', 'data_types').split(', '))

    dataset_id = get_script_arguments()
    dataset_name = data_types[dataset_id]  # e.g. 0_Baseline
    print("Selected dataset:\n\tID: {}\n\tName: {}\n".format(dataset_id, dataset_name))

    # directory paths to training, validation, testing datasets (i.e. the generated RGBA images)
    generated_dir = os.path.abspath(config.get('CNN', 'generated_dir'))
    dataset_dir = os.path.join(generated_dir, dataset_name)  # e.g. 3_generated-data/156x108/0_Baseline
    results_dir = os.path.abspath(config.get('CNN', 'results_dir'))
    epochs = config.get('CNN', 'epochs')
    batch_size = config.get('CNN', 'batch_size')
    # the cropped dimensions of generated video frames
    x_axis = int(config.get('CNN', 'x_axis'))
    y_axis = int(config.get('CNN', 'y_axis'))

    tmp_fold_dir = os.path.join(results_dir, "tmp")  # ten-fold datasets are copied into tmp directories during training
    models_dir = os.path.join(results_dir, "models")
    logs_dir = os.path.join(results_dir, "logs")

    # files & directories
    create_directories([results_dir, models_dir, logs_dir])
    test_results = open(os.path.join(results_dir, "Scores.EvaluationResults.txt"), "a+")

    # combine all training and validation directory files into a single dataframe for ten fold training/validation
    filenames, classes = [], []
    for basedir in ['Training', 'Validation']:
        for subdir in ["empty", "lying", "sitting", "crawling", "bending", "standing"]:
            path = os.path.join(os.path.join(os.path.join(os.getcwd(), dataset_dir), basedir), subdir)
            for file in absolute_file_paths(path):
                filenames.append(file)
                classes.append(subdir)
    df = pd.DataFrame({
        'filename': filenames,
        'class': classes
    })
    df_y = df['class']
    df_x = df['filename']
    # feedback on the size of training data sample set
    print("Reading data from directory:\n  {}\n", dataset_dir)
    print("\t{:<20}: \t{}".format("TOTAL SAMPLE SIZE", str(df['class'].value_counts().sum())))
    print("\t{}\n".format(str(df['class'].value_counts()).replace('\n', '\n\t')))


    skf = StratifiedShuffleSplit(n_splits=10, test_size=0.17, random_state=0)  # take %17 of training+validation data as the ten-fold validation set
    total_actual, total_predicted, total_val_accuracy, total_val_loss, total_test_accuracy, classes = [], [], [], [], [], []

    for i, (train_index, test_index) in enumerate(skf.split(df_x, df_y)):
        x_train, x_test = df_x.iloc[train_index], df_x.iloc[test_index]
        y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]

        train = pd.concat([x_train, y_train], axis=1)
        validation = pd.concat([x_test, y_test], axis=1)

        # feedback on training and validation set sample sizes
        print("\n>>> TRAINING FOLD [{}/{}]\n".format(str(i+1), str(skf.n_splits)))
        print("\t{:<20}: \t{}".format("TRAIN SET SAMPLE SIZE", str(train['class'].value_counts().sum())))
        print("\t{}\n".format(str(train['class'].value_counts()).replace('\n', '\n\t')))
        print("\t{:<20}: \t{}".format("VALIDATION SET SAMPLE SIZE", str(validation['class'].value_counts().sum())))
        print("\t{}\n".format(str(validation['class'].value_counts()).replace('\n', '\n\t')))

        # safeguard: make sure validation data does not include training data
        train = train[~train['filename'].isin(list(validation['filename']))]

        # copy the images according to the given fold over to tmp directory
        train_dir = copy_images(train, 'Training', tmp_fold_dir)
        valid_dir = copy_images(validation, 'Validation', tmp_fold_dir)
        test_dir = path = os.path.join(dataset_dir, "Testing")

        model_name = "{}-Fold_{}.h5".format(dataset_name, int(i))

        # create and train the model - returns the validation accuracy and validation loss
        model, val_loss, val_accuracy = compile_model(batch_size=batch_size, image_size=(y_axis, x_axis), model_name=model_name, train_dir=train_dir, valid_dir=valid_dir)
        # store validation accuracy and loss - to be used to calculate average over all 10 folds
        total_val_accuracy.append(val_accuracy)
        total_val_loss.append(val_loss)

        # the predict() method executes the model to guess the class of images in the test set directory
        # this function returns the actual classes and the predicted classes in the same order
        classes, predicted, actual = predict(model=model, batch_size=epochs, image_size=(y_axis, x_axis), test_dir=test_dir)

        # append accuracy from the predictions on the test data
        total_test_accuracy.append(accuracy_score(actual, predicted))

        # append all of the actual and predicted classes for your final evaluation
        total_actual = [*total_actual, *list(actual)]
        total_predicted = [*total_predicted, *list(predicted)]

    # EVALUATE MODEL
    test_results.write(">>> DATASET:\t{}\n".format(dataset_name))
    # validation data partition
    test_results.write("\nVALIDATION ACCURACY:\t{:.2f}%\n".format(np.mean(total_val_accuracy)*100))
    test_results.write("avg acc:\t{}\n".format(np.mean(total_val_accuracy)))
    test_results.write("avg loss:\t{}\n".format(np.mean(total_val_loss)))
    test_results.write("\tfor each fold:\n")
    for i, val in enumerate(total_val_accuracy):
        test_results.write("\t\t[{}]\t{}\n".format(i+1, val))
    # test data partition
    test_results.write("\nTEST ACCURACY:\t{:.2f}%\n".format(np.mean(total_test_accuracy)*100))
    test_results.write("\tfor each fold:\n")
    for i, val in enumerate(total_test_accuracy):
        test_results.write("\t\t[{}]\t{}\n".format(i, val))

    # EVALUATE MODEL: set-aside TEST data partition
    # CONFUSION MATRIX 1: overview
    predicted_classes = total_predicted
    true_classes = total_actual
    class_labels = classes
    # count correct and incorrect predictions
    correct = sum(x == y for x, y in zip(predicted_classes, true_classes))
    incorrect = len(predicted_classes) - correct
    test_results.write("\tCorrectly predicted test set instances: \t{}\n".format(correct))
    test_results.write("\tIncorrectly predicted test set instances: \t{}\n\n".format(incorrect))
    # write confusion matrix on precision, accuracy and F1
    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    test_results.write(report+"\n")

    # CONFUSION MATRIX 2: per class
    lookup = {0: class_labels[0], 1: class_labels[1], 2: class_labels[2], 3: class_labels[3], 4: class_labels[4], 5: class_labels[5]}
    y_true = pd.Series([lookup[_] for _ in true_classes])
    y_pred = pd.Series([lookup[_] for _ in predicted_classes])
    # dataframe CM 1
    df = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    df.to_csv(test_results, sep='\t', encoding='utf-8')
    # dataframe CM 2
    df_confusion = pd.crosstab(pd.Series(true_classes, name='Actual'), pd.Series(predicted_classes, name='Predicted'), rownames=['Actual'], colnames=['Predicted'], margins=True)
    CM_file_path = os.path.join(results_dir, "Scores.ConfusionMatrix.txt")
    with open(CM_file_path, "a") as file:
        file.write("\n" + dataset_name + "\n")
        df_confusion.to_csv(file, header=True, index=True, sep="\t", mode='a')

    # CONFUSION MATRIX 2: (visualised in blue)
    categorical_test_labels = pd.DataFrame(true_classes)
    categorical_preds = pd.DataFrame(predicted_classes)
    confusion_matrix = metrics.confusion_matrix(categorical_test_labels, categorical_preds)
    plot_confusion_matrix(dataset_name, confusion_matrix, class_labels, normalize=False)

    # CM: Blues image
    y_true = np.array([lookup[_] for _ in true_classes])
    y_pred = np.array([lookup[_] for _ in predicted_classes])
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=class_labels)

    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(dataset_name, conf_matrix, classes=class_labels, normalize=True, title='Confusion matrix (normalised)')
    plt.close()

    other_metrics = interpret_metrics(confusion_matrix)
    test_results.write("Overall accuracy for each class:\n{}\n".format(other_metrics[0]))
    test_results.write("Sensitivity, hit rate, recall, or true positive rate:\n{}\n".format(other_metrics[1]))
    test_results.write("Specificity or true negative rate:\n{}\n".format(other_metrics[2]))
    test_results.write("Precision or positive predictive value:\n{}\n".format(other_metrics[3]))
    test_results.write("Negative predictive value:\n{}\n".format(other_metrics[4]))
    test_results.write("Fall out or false positive rate:\n{}\n".format(other_metrics[5]))
    test_results.write("False negative rate:\n{}\n".format(other_metrics[6]))
    test_results.write("False discovery rate:\n{}\n\n".format(other_metrics[7]))

    test_results.write("TP:\n{}\n".format(other_metrics[8]))
    test_results.write("TN:\n{}\n".format(other_metrics[9]))
    test_results.write("FP:\n{}\n".format(other_metrics[10]))
    test_results.write("FN:\n{}\n".format(other_metrics[11]))

    # remove the temp directory
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)

    gc.collect()
