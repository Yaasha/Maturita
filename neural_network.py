from config import *
from os import environ

# set keras backend
environ['KERAS_BACKEND'] = BACKEND
import dataset
from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import LambdaCallback
import pickle
import os.path
import numpy as np


def create_net(data):
    """
    create model of the neural network using keras
    :param
        data: data from the dataset (we need only the number of classes)
    :return:
        the model of the neural network
    """
    print("Creating model...")

    # separate data
    (train_images, train_labels), (test_images, test_labels), classes = data

    # create sequential net model
    model = Sequential()

    # first convolution layer with relu activation function
    model.add(Convolution2D(NB_FILTERS,
                            KERNEL_SIZE,
                            padding='valid',
                            input_shape=(HEIGHT, WIDTH, 1),
                            activation='relu'))

    # second convolution layer with relu activation function
    model.add(Convolution2D(NB_FILTERS, KERNEL_SIZE, activation='relu'))

    # max pooling layer (downscaling input 2x)
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    # add dropout to prevent overfitting
    model.add(Dropout(DROPOUT))

    # 3th and 4th convolution layer with relu activation function and 2 times the filters
    model.add(Convolution2D(2 * NB_FILTERS, KERNEL_SIZE, activation='relu'))
    model.add(Convolution2D(2 * NB_FILTERS, KERNEL_SIZE, activation='relu'))

    # second max pooling layer (downscaling input 2x)
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    # add dropout to prevent overfitting
    model.add(Dropout(DROPOUT))

    # flatten the data (from multi-dimensional array to 1-dimension)
    model.add(Flatten())

    # narrow down the results to 512 classes using relu activation function
    model.add(Dense(512, activation='relu'))
    # add dropout to prevent overfitting
    model.add(Dropout(2 * DROPOUT))
    # narrow the results down to 62 classes using softmax activation function
    model.add(Dense(classes, activation='softmax'))

    # calculate loss and accuracy of the model and optimize it accordingly
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # save model graph as an image
    print("Model visualization...")
    plot_model(model, to_file=visualize_path, show_shapes=True)

    # save the model to file
    model.save(model_path)

    return model


def get_model():
    if os.path.isfile(model_path):
        model = load_model(model_path)
    else:
        data = dataset.load()
        model = create_net(data)

    return model


def get_batch(data, batch):
    batch = (batch + 1) % int(len(data) / BATCH_SIZE)
    start = (batch - 1) * BATCH_SIZE
    end = batch * BATCH_SIZE
    return data[start:end]


def train(model, data, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    train the neural network and save it to file
    :param
        model: model of the neural network
    :param
        data: data from the dataset (only training and testing dataset needed)
    :param
        batch_size: amount of input data in each epoch (default specified in config)
    :param
        epochs: number of training cycles (default specified in config)
    :return:
        training accuracy
    """
    # load current epoch of the model
    if os.path.isfile(cur_epoch_path):
        cur_epoch = pickle.load(open(cur_epoch_path, 'rb'))
    else:
        cur_epoch = 0

    # separate data
    (train_images, train_labels), (test_images, test_labels), classes = data

    history_keys = ['loss', 'acc', 'val_loss', 'val_acc']
    # if batch history of model already exists load it
    if os.path.isfile(history_batch_path):
        history_batch = pickle.load(open(history_batch_path, 'rb'))
    # if not create new array with column titles
    else:
        history_batch = [history_keys]

    # callback for saving loss and history each batch
    score_callback = LambdaCallback(on_batch_end=lambda batch, logs: history_batch.append(
        [logs['loss'], logs['acc']] + list(model.evaluate(get_batch(test_images, batch), get_batch(test_labels, batch),
                                                          batch_size=BATCH_SIZE, verbose=0))))

    # train the model
    history_dict = model.fit(train_images,
                             train_labels,
                             batch_size=batch_size,
                             epochs=cur_epoch + epochs,
                             validation_data=(test_images, test_labels),
                             initial_epoch=cur_epoch,
                             verbose=2,
                             callbacks=[score_callback])

    # save the model to file
    model.save(model_path)

    history_dict = history_dict.history
    # if history of model already exists load it
    if os.path.isfile(history_path):
        history = pickle.load(open(history_path, 'rb'))
    # if not create new array with column titles
    else:
        history = [history_keys]

    # add the new history to the old one
    for i in range(len(history_dict[history_keys[0]])):
        epoch_history = []
        for key in history_keys:
            epoch_history.append(history_dict[key][i])

        history.append(epoch_history)

    # save history and batch history into a file
    pickle.dump(history, open(history_path, 'wb'))
    pickle.dump(history_batch, open(history_batch_path, 'wb'))
    # save current epoch
    pickle.dump(cur_epoch + epochs, open(cur_epoch_path, 'wb'))

    # get the model`s accuracy
    score = history[-1][3]
    print('Test accuracy:', score)

    return score


def predict(model, input, ignore_classes=()):
    """
    predict text based on input
    :param
        model: model to use for the prediction
    :param
        input: input data for the prediction
    :param
        ignore_classes: classes to be ignored in the result
    :return:
        array based on predictions for inputs
    """
    # load mapping of classes to the ASCII characters
    mapping = dataset.load_mapping()

    result = []
    # for every input
    for i in input:
        # predict character
        prediction = model.predict(np.expand_dims(i, axis=0))

        # top 3 results for each input
        top_3 = []

        # average of black pixels
        mean = np.mean(i)
        if mean < 0.03:  # if the input is blank it is space
            top_3.append([" ", "mean of black pixels"])
        elif mean > 0.5:  # if the input is filled it is blank
            top_3.append(["", "mean of black pixels"])
        else:
            # set probability to 0 for ignored classes
            if ignore_classes:
                for j in ignore_classes:
                    prediction[0][j] = 0

            # find the most possible class
            class_index = prediction.argmax()
            probability = "%.2f" % (prediction[0][class_index] * 100) + '%'
            # set the probability of that class to 0 (so it cannot be selected as 2nd or 3rd)
            prediction[0][class_index] = 0

            top_3.append([chr(mapping[class_index]), probability])

        # second and third prediction
        for j in range(2):
            # find the most possible class
            class_index = prediction.argmax()
            probability = "%.2f" % (prediction[0][class_index] * 100) + '%'
            # set the probability of that class to 0 (so it cannot be selected as 2nd or 3rd)
            prediction[0][class_index] = 0

            top_3.append([chr(mapping[class_index]), probability])

        # add input's prediction to the result
        result.append(top_3)

    return result


def evaluate(model, inputs, ignore_classes=()):
    """
    get model's accuracy for specified input
    :param
        model: model to use for the evaluation
    :param
        inputs: input data and labels for the evaluation
    :param
        ignore_classes: classes to be ignored in accuracy evaluation
    :return:
        model's loss and accuracy
    """
    # load mapping of classes to the ASCII characters
    mapping = dataset.load_mapping()
    # convert labels to one-hot (e.g. 5 to 0,0,0,0,0,1,0...)
    classes = len(mapping)  # number of classes
    labels = np_utils.to_categorical(inputs[1], classes)
    # evaluate model and get loss
    score = model.evaluate(inputs[0], labels, batch_size=BATCH_SIZE, verbose=0)
    loss = "%.2f" % score[0]

    # get accuracy of the model for each selected class
    total = np.zeros(classes)
    correct = np.zeros(classes)
    # predict classes
    prediction = model.predict(np.array(inputs[0]))
    # set probability to 0 for ignored classes
    for i in range(len(inputs[0])):
        current_class = labels[i].argmax()
        for j in ignore_classes:
            prediction[i][j] = 0
        # check if prediction matches the label
        if current_class == prediction[i].argmax():
            correct[current_class] += 1
        total[current_class] += 1
    # total accuracy
    acc = "%.2f" % ((np.sum(correct) / np.sum(total)) * 100) + '%'

    # accuracy for each class
    acc_class = []
    for i in range(classes):
        # only add the selected classes
        if total[i] != 0:
            acc_class.append({"char": chr(mapping[i]), "acc": "%.2f" % ((correct[i] / total[i]) * 100) + '%'})

    return loss, acc, acc_class


if __name__ == '__main__':

    data = dataset.load()
    model = get_model()
    acc = 0
    while acc < 0.88:
        acc = train(model, data, epochs=1)
