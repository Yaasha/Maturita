from config import *
from scipy.io import loadmat
from keras.utils import np_utils
import pickle


def load(file_path=dataset_path):
    """
    load dataset from a .mat file and save mapping to ASCII into a file
    :param
        file_path: path to the .mat file (default value specified in config)
    :return:
        training and testing datasets and the number of classes
        format: (train_data, train_labels), (test_data, test_labels), number_of_classes
    """
    print("Loading dataset...")
    # load dataset from file in matlab format
    dataset_mat = loadmat(file_path)

    # map classes (0 - 62) to ASCII codes for 0-9, A-Z, a-z
    mapping = {char[0]: char[1] for char in dataset_mat['dataset'][0][0][2]}
    # save mapping to a file
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    pickle.dump(mapping, open(mapping_path, 'wb'))

    # load training data
    # reshape flattened image to 2D array in matlab order (because of the format of the file)
    train_images = dataset_mat['dataset'][0][0][0][0][0][0].reshape(-1, HEIGHT, WIDTH, 1, order='A')
    train_labels = dataset_mat['dataset'][0][0][0][0][0][1]

    # load testing data
    # reshape flattened image to 2D array in matlab order (because of the format of the file)
    test_images = dataset_mat['dataset'][0][0][1][0][0][0].reshape(-1, HEIGHT, WIDTH, 1, order='A')
    test_labels = dataset_mat['dataset'][0][0][1][0][0][1]

    # convert type to float32 (from int) and normalize (e.g. 255 to 1, 128 to 0.5, etc.)
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # convert labels to one-hot (e.g. 5 to 0,0,0,0,0,1,0...)
    classes = len(mapping)      # number of classes
    train_labels = np_utils.to_categorical(train_labels, classes)
    test_labels = np_utils.to_categorical(test_labels, classes)

    return (train_images, train_labels), (test_images, test_labels), classes


def load_mapping():
    """
    load mapping of the dataset from file specified in config
    :return:
        the mapping of the dataset
    """
    mapping = pickle.load(open(mapping_path, 'rb'))
    return mapping


def get_classes(type=''):
    """
    get list of classes based on selected type of chars
    :param
        type: type of characters (uppercase, lowercase, digit, blank for all classes)
    :return:
        list of classes
    """
    # load mapping to convert chars to classes
    mapping = load_mapping()
    # result list
    classes = []
    # get keys from the mapping dictionary
    keys = list(mapping.keys())
    if type == 'digit':
        # for each digit ASCII code add its class to the result
        for i in range(48, 58):
            # get the key by the value
            classes.append(keys[list(mapping.values()).index(i)])

    elif type == 'uppercase':
        # for each uppercase ASCII code add its class to the result
        for i in range(65, 91):
            classes.append(keys[list(mapping.values()).index(i)])

    elif type == 'lowercase':
        # for each lowercase ASCII code add its class to the result
        for i in range(97, 123):
            classes.append(keys[list(mapping.values()).index(i)])

    else:
        # return all classes
        classes = keys

    return classes
