from config import *
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4, cm
import base64


def show(img):
    """
    show opencv image in resized window (for debugging)
    :param
        img: image to be shown
    """
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    ratio = np.size(img, 0)/np.size(img, 1)
    cv2.resizeWindow('image', 550, int(550*ratio))
    cv2.waitKey(0)


def prepare_image(file):    # splits the image into grid
    # load image
    img = cv2.imdecode(file, cv2.IMREAD_COLOR)
    original = img
    # turn into gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create binary image (inverted)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 2)
    # find shapes
    var, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # anchor squares
    squares = []
    # anchor shape to find
    template = np.array([[[0, 0]], [[0, 2]], [[1, 2]], [[1, 1]], [[2, 1]], [[2, 0]]])
    for cnt in contours:    # search the shape in contours
        if cv2.matchShapes(template, cnt, 1, 0.0) < 0.05:
            squares.append(cnt)

    # if more than four anchor shapes were found pick the ones on the outside
    if len(squares) > 4:
        squares = [squares[0], squares[1], squares[len(squares) - 1], squares[len(squares) - 2]]

    # list of centroids of the squares
    edges = []
    for square in squares:  # calculate centroid for each square
        m = cv2.moments(square)
        centroid_x = int(m["m10"] / m["m00"])
        centroid_y = int(m["m01"] / m["m00"])
        edges.append([centroid_x, centroid_y])

    # sort edges from left top to right bottom
    edges.sort(key=lambda e: e[0]+e[1])

    # calculate width and height of the document
    width = edges[1][0] - edges[0][0]
    height = edges[2][1] - edges[0][1]

    # create points for transforming
    edges = np.array(edges, dtype="float32")
    points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype="float32")

    # transform image according to squares
    m = cv2.getPerspectiveTransform(edges, points)
    img = cv2.warpPerspective(img, m, (width, height))
    original = cv2.warpPerspective(original, m, (width, height))    # also transform the original image

    return img, original


def read_grid(img, original, start, columns, rows, vertical_space=0):
    """
    extract grid from binary image and original image
    :param
        img: binary image
    :param
        original: original image
    :param
        start: top left corner of the grid
    :param
        columns: number of columns in the grid
    :param
        rows: number of rows in the grid
    :param
        vertical_space: is there a vertical space between rows
    :return:
        list of extracted images for binary and original image
        format: [binary images, original images]
    """
    # horizontal and vertical ratio to the generated pdf document
    ratio_x = len(img[0]) / (A4[0] - 84)
    ratio_y = len(img) / (A4[1] - 84)
    ratio = ratio_x, ratio_y

    # calculate size of one cell of the grid
    cell_width = CELL_WIDTH_CM * cm * ratio[0]
    cell_height = CELL_HEIGHT_CM * cm * ratio[1]

    # relative starting point to absolute
    start = (start[0] * ratio[0], start[1] * ratio[1])

    # placeholder for our result (binary images, original images)
    result = [[], []]
    # for every row and column of the grid
    for i in range(rows):
        for j in range(columns):
            # top left corner of the cell
            x1 = round(start[0] + j * cell_width)
            y1 = start[1] + i * cell_height
            # add vertical space if selected
            if vertical_space > 0:
                y1 += 4.23 * i * ratio[1]
            y1 = round(y1)

            # bottom right corner of the cell
            x2 = round(x1 + cell_width)
            y2 = round(y1 + cell_height)

            # extract cell from images
            result[0].append(img[y1:y2, x1:x2])
            result[1].append(original[y1:y2, x1:x2])

    return result


def remove_borders(image):
    """
    removes white borders from the images
    :param
        image: a pair of binary and original image
    :return:
        cropped binary and original image without borders
    """
    # smooth the image to average scanning error
    smooth = cv2.GaussianBlur(image[0], (5, 5), 0)
    # mask to find dark pixels
    mask = smooth < 0.5
    # indicies of the dark pixels
    coords = np.argwhere(mask)
    # if there are any dark pixels
    if coords.any():
        # set points for the cropped image
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 2
    else:
        # just set the points to include the full image
        x0, y0 = 0, 0
        x1, y1 = len(image[0][0]), len(image[0][1])

    # crop the images to remove white border
    image[0] = image[0][x0:x1, y0:y1]
    image[1] = image[1][x0:x1, y0:y1]

    return image

def prepare_for_cnn(images):
    """
    edit images to be suitable for the cnn
        - crop borders
        - resize it to preferred size
        - convert binary image to normalized numpy array
    :param
        images: list of images to be edited
    :return:
        list of edited images
    """
    # for each pair of images
    for i in range(len(images[0])):
        # crop grid borders
        images[0][i], images[1][i] = remove_borders([images[0][i], images[1][i]])

        # for each image in pair (binary : original)
        for j in range(2):
            # resize image to format specified in config
            images[j][i] = cv2.resize(images[j][i], (WIDTH, HEIGHT))

    # convert binary image to numpy array
    inputs = np.array(images[0])
    # reshape array to (28, 28, 1)
    inputs = inputs.reshape(inputs.shape + (1,))
    # normalize array
    inputs = inputs.astype('float32') / 255
    # transform original image into string so it can be shown in html
    org_images = []
    for img in images[1]:
        # convert opencv mat into jpg image
        i, org = cv2.imencode(".jpg", img)
        # encode image into base64
        org = base64.b64encode(org)
        # convert to string
        org = org.decode('utf-8')
        org_images.append(org)

    # add array and original images together
    images = [inputs, org_images]
    return images


def read_train(filename):
    """
    reads data from the training paper
    :param
        filename: filepath to the .jpg file
    :return:
        dataset of characters as array and image
        format: training data(uppercase(bin, img), lowercase(b, i), digits(b, i)), testing data(b, i)
    """
    # load and prepare our image
    img, original = prepare_image(filename)
    # read uppercase letters
    start_uppercase = (16.4, 43)    # starting point for uppercase grid
    uppercase = read_grid(img, original, start_uppercase, 26, 6)
    uppercase = prepare_for_cnn(uppercase)
    # read lowercase letters
    start_lowercase = (16.4, 185)   # starting point for lowercase grid
    lowercase = read_grid(img, original, start_lowercase, 26, 6)
    lowercase = prepare_for_cnn(lowercase)
    # read digits
    start_digits_1 = (44.4, 326.8)   # starting point for first digit grid
    digits = read_grid(img, original, start_digits_1, 10, 3)
    start_digits_2 = (279.5, 326.8)  # starting point for second digit grid
    digits_2 = read_grid(img, original, start_digits_2, 10, 3)

    digits = [digits[0] + digits_2[0], digits[1] + digits_2[1]]

    digits = prepare_for_cnn(digits)
    # combine uppercase, lowercase and digits into training dataset
    train = (uppercase, lowercase, digits)

    #testing dataset
    start_test = (16.4, 398)    # starting point for testing grid
    test = read_grid(img, original, start_test, 26, 15, vertical_space=1)
    test = prepare_for_cnn(test)

    return train, test


def read_test(filename):
    """
    reads data from the testing paper
    :param
        filename: filepath to the .jpg file
    :return:
        dataset of characters as array and image
        format: testing data(bin, img)
    """
    # load and prepare our image
    img, original = prepare_image(filename)
    # read grid cells
    start_test = (16.4, 15.3)   # starting poin of the grid
    test = read_grid(img, original, start_test, 26, 32, vertical_space=1)
    test = prepare_for_cnn(test)

    return test
