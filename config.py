import os.path

# filepaths
DATASET_DIRECTORY = 'dataset'           # directory of the dataset and mapping files
DATASET_FILE = 'emnist-byclass.mat'     # name of the dataset file
DATASET_MAPPING_FILE = 'mapping.p'      # name of the mapping file

MODEL_DIRECTORY = 'saves_tensorflow'               # directory of the model and weights
MODEL_FILE = 'model.h5'                 # name of the model file
HISTORY_FILE = 'history.p'              # name of the model`s history file
HISTORY_BATCH_FILE = 'history_batch.p'  # name of the model`s batch history file
CUR_E_FILE = 'epoch.p'                  # name of the current epoch file
MODEL_VISUALIZE = 'model.png'           # name of the image of the model

# joined directories and filenames
dataset_path = os.path.join(DATASET_DIRECTORY, DATASET_FILE)
mapping_path = os.path.join(DATASET_DIRECTORY, DATASET_MAPPING_FILE)

model_path = os.path.join(MODEL_DIRECTORY, MODEL_FILE)
history_path = os.path.join(MODEL_DIRECTORY, HISTORY_FILE)
history_batch_path = os.path.join(MODEL_DIRECTORY, HISTORY_BATCH_FILE)
cur_epoch_path = os.path.join(MODEL_DIRECTORY, CUR_E_FILE)

visualize_path = os.path.join(MODEL_DIRECTORY, MODEL_VISUALIZE)

# input data parameters
HEIGHT = 28     # height of the images
WIDTH = 28      # width of the images

# neural net model parameters
BACKEND = "tensorflow"      # backend for keras (tensorflow, theano, cntk)
NB_FILTERS = 64         # number of convolutional filters
POOL_SIZE = (2, 2)      # size of pooling area for max pooling
KERNEL_SIZE = (3, 3)    # size of the convolutional kernel
DROPOUT = 0.25          # dropout chance
BATCH_SIZE = 1000       # number of inputs in one epoch
EPOCHS = 10             # number of training cycles

# constants for creating pdf
# offset of each side of document in cm
X_OFFSET_CM = 2  # horizontal
Y_OFFSET_CM = 2  # vertical

# dimensions of cell in cm
CELL_WIDTH_CM = 0.65
CELL_HEIGHT_CM = 0.65

FONT_SIZE = 20

# constants for reading pdf
# number of characters per line
LINE_BREAK = 26
