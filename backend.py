import tensorflow as tf
import numpy as np
import neural_network
import pdf_read
import dataset
import pickle
from config import *
import base64

model = None
graph = None


def server_start():
    global model
    global graph
    model = neural_network.get_model()
    # workaround for tensorflow bug with django
    if BACKEND == 'tensorflow':
        model._make_predict_function()
        graph = tf.get_default_graph()


def handle_request(request, mode=""):
    data = {}
    post = request.POST
    if post:
        if post['read']:
            file = request.FILES.get('read_file', None)
            mode = post['read_type']
            up = post.get('read_up', 0)
            low = post.get('read_low', 0)
            digit = post.get('read_digit', 0)
            # if no checkboxes were checked, check all of them
            if not up and not low and not digit:
                up = low = digit = 1

            if file and file.name.endswith('.jpg'):
                data.update(read_doc(file, mode, up, low, digit))

    if mode == 'info':
        data.update(get_info())

    return data


def split_array(array, elements):
    """
    split array into subarrays of specified length
    :param
        array: original array
    :param
        elements: length of subarrays
    :return:
        array of subarrays of specified length
    """
    length = len(array)
    index = 0
    end = index + elements
    result = []
    while end < length:
        result.append(array[index:end])
        index = end
        end = index + elements
    result.append(array[index:])

    return result


def get_ignored_classes(uppercase, lowercase, digit):
    """
    get tuple of ignored classes based on selected classes
    :param
        uppercase: whether to keep uppercase classes
    :param
        lowercase: whether to keep lowercase classes
    :param
        digit: whether to keep digit classes
    :return:
        tuple of ignored classes
    """
    # result placeholder
    ignored = []
    # add digit classes to the ignore list
    if not digit:
        ignored.append(dataset.get_classes('digit'))
    # add uppercase classes to the ignore list
    if not uppercase:
        ignored.append(dataset.get_classes('uppercase'))
    # add lowercase classes to the ignore list
    if not lowercase:
        ignored.append(dataset.get_classes('lowercase'))
    # return tuple
    return tuple(ignored)


def train_evaluate(data, read_up, read_low, read_digit):
    """
    evaluate model on training data
    :param
        data: training data read from the training document
    :param
        read_up: whether to work with uppercase classes
    :param
        read_low: whether to work with lowercase classes
    :param
        read_digit: whether to work with digit classes
    :return:
        dictionary of results for the server
    """
    # result dictionary
    result = {}

    # total loss and accuracy
    total = [np.empty((0, 28, 28, 1)), []]
    # if reading uppercase
    if read_up:
        # uppercase data and labels
        uppercase = [data[0][0][0], dataset.get_classes('uppercase') * (len(data[0][0][0]) // 26)]
        # uppercase loss and accuracy
        uppercase_score = neural_network.evaluate(model, uppercase,
                                                  get_ignored_classes(read_up, read_low, read_digit))
        # add uppercase score to result
        result.update({'uppercase_loss': uppercase_score[0], 'uppercase_acc': uppercase_score[1]})
        total = [np.append(total[0], uppercase[0], axis=0), total[1] + uppercase[1]]

    # if reading lowercase
    if read_low:
        # lowercase data and labels
        lowercase = [data[0][1][0], dataset.get_classes('lowercase') * (len(data[0][1][0]) // 26)]
        # digit loss and accuracy
        lowercase_score = neural_network.evaluate(model, lowercase,
                                                  get_ignored_classes(read_up, read_low, read_digit))
        # add lowercase score to result
        result.update({'lowercase_loss': lowercase_score[0], 'lowercase_acc': lowercase_score[1]})
        total = [np.append(total[0], lowercase[0], axis=0), total[1] + lowercase[1]]

    # if reading digits
    if read_digit:
        # digit data and labels
        digit = [data[0][2][0], dataset.get_classes('digit') * (len(data[0][2][0]) // 10)]
        # digit loss and accuracy
        digit_score = neural_network.evaluate(model, digit,
                                              get_ignored_classes(read_up, read_low, read_digit))
        # add digits score to result
        result.update({'digit_loss': digit_score[0], 'digit_acc': digit_score[1]})
        total = [np.append(total[0], digit[0], axis=0), total[1] + digit[1]]

    # add total accuracy to the result
    total_score = neural_network.evaluate(model, total, get_ignored_classes(read_up, read_low, read_digit))
    result.update({'total_loss': total_score[0], 'total_acc': total_score[1], 'class_acc': total_score[2]})

    return result


def read_doc(file, mode, read_up, read_low, read_digit):
    """
    read data from specified document
    :param
        file: the image to be read
    :param
        mode: type of the document (train/test)
    :param
        read_up: whether to work with uppercase classes
    :param
        read_low: whether to work with lowercase classes
    :param
        read_digit: whether to work with digit classes
    :return:
        dictionary of results for the server
    """

    # get cnn model and graph
    global model
    global graph

    # dictionary with result
    result = {}

    # convert file to numpy array
    file = np.fromstring(file.read(), np.uint8)

    # extract data to predict and original image to show
    if mode == 'train':
        data = pdf_read.read_train(file)
        org_images = data[1][1]
        inputs = data[1][0]

        # evaluate the model on the training data
        if BACKEND == 'tensorflow':  # workaround for tensorflow bug with django
            with graph.as_default():
                result.update(train_evaluate(data, read_up, read_low, read_digit))
        else:
            result.update(train_evaluate(data, read_up, read_low, read_digit))

    else:
        data = pdf_read.read_test(file)
        org_images = data[1]
        inputs = data[0]

    # predict the results
    if BACKEND == 'tensorflow':  # workaround for tensorflow bug with django
        with graph.as_default():
            text = neural_network.predict(model, inputs, get_ignored_classes(read_up, read_low, read_digit))
    else:
        text = neural_network.predict(model, inputs, get_ignored_classes(read_up, read_low, read_digit))

    # add text and images into one list
    read_data = list(map(list, zip(text, org_images)))

    # split data into lines
    read_data = split_array(read_data, LINE_BREAK)

    # add reading results
    result.update({'read_data': read_data})
    return result


def insert_index(array):
    """
    add first column to 2D array with index
    :param
        array: array to which we want to add index
    :return:
    array with indexes in in first column
    """
    for i in range(len(array)):
        array[i].insert(0, i)
    return array


def dense_array(array, divide):
    """
    delete inconsistent elements of 2D array and reduce its size by creating smaller array
    :param
        array: array to dense
    :param
        divide: number to divide the array size
    :return:
    2D divided array with consistent shape
    """
    result = []
    # get the right size from first element
    size = len(array[0])
    print(size)
    for e in array:
        # if the element has different size delete it
        if len(e) != size:
            i = array.index(e)
            del array[i]
    # divide the array into smaller array
    for i in range(len(array)):
        if i % divide == 0:
            result.append(array[i])
    return result


def get_info():
    """
    gets project info
    :return:
    dictionary with project information
    """

    # get number of epochs
    epochs = pickle.load(open(cur_epoch_path, "rb"))

    # get current backend
    backend = BACKEND
    # get batch size
    batch_size = BATCH_SIZE

    # get epoch history
    epoch_history = pickle.load(open(history_path, "rb"))

    # get training accuracy
    train_acc = "%.2f" % (epoch_history[-1][1] * 100) + '%'
    # get testing accuracy
    val_acc = "%.2f" % (epoch_history[-1][3] * 100) + '%'

    # add index column
    epoch_history = insert_index(epoch_history)
    # set headers
    epoch_history[0] = ['Epoch', 'Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy']

    # get batch history
    batch_history = pickle.load(open(history_batch_path, "rb"))
    # add index column
    batch_history = insert_index(batch_history)
    # delete inconsistent elements and dense the array
    batch_history = dense_array(batch_history, 25)
    # set headers
    batch_history[0] = ['Batch', 'Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy']

    # create result dictionary
    result = {'cur_epoch': epochs, 'backend': backend, 'train_acc': train_acc, 'val_acc': val_acc,
              'epoch_history': epoch_history, 'batch_history': batch_history, 'batch_size': batch_size}

    # add image of the cnn model
    with open(visualize_path, "rb") as image:
        # get image
        image = image.read()
        # encode it to base64
        image = base64.b64encode(image)
        # convert it to string
        image = image.decode('utf-8')
        result.update({'model_image': image})

    return result
