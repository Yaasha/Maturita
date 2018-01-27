from config import *
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, cm
import string

# dimensions of the document
PAGE_WIDTH, PAGE_HEIGHT = A4


def grid(items, cell_size, offset, center, center_value):  # calculates the location of each cell
    # list of cells
    result = []
    for i in range(items + 1):  # for each cell
        # cell's order * real cell size + real offset size
        cell = i * cm * cell_size + offset * cm
        if center == 1:  # if I want to center the cells on the page
            # add offset needed to center
            cell += center_value
        # add cell to the list
        result.append(cell)
    return result


'''
function for creating tables
    columns (int) - number of columns (if 0 number of columns will be set to fill the page available)
    rows (int) - number of rows (if 0 number of rows will be set to fill the page available)
    horizontal_center (1/0) - 1 = center the table on the page horizontally, 0 = don't center horizontally
    vertical_center (1/0) - 1 = center the table on the page vertically, 0 = don't center vertically
    x (int) - explicit horizontal offset (if 0 default offset will be used)
    y (int) - explicit vertical offset (if 0 default offset will be used)
'''


def table(pdf, columns=0, rows=0, horizontal_center=1, vertical_center=1, x=0.0, y=0.0):
    # width of the available page (width - offset)
    if x > 0:
        width = PAGE_WIDTH - (x + X_OFFSET_CM) * cm
    else:
        width = PAGE_WIDTH - 2 * X_OFFSET_CM * cm

    # height of the available page (height - offset)
    if y > 0:
        height = PAGE_HEIGHT - (y + Y_OFFSET_CM) * cm
    else:
        height = PAGE_HEIGHT - 2 * Y_OFFSET_CM * cm

    # set number of columns and rows to fit the page (if not already specified)
    if columns == 0:
        columns = int(width / (CELL_WIDTH_CM * cm))
    if rows == 0:
        rows = int(height / (CELL_HEIGHT_CM * cm))

    # calculate the offset needed to center the table
    x_center = (width - columns * CELL_WIDTH_CM * cm) / 2
    y_center = (height - rows * CELL_HEIGHT_CM * cm) / 2

    # if explicit offset is specified, override default
    x_offset = x if x > 0 else X_OFFSET_CM
    y_offset = y if y > 0 else Y_OFFSET_CM

    # create list of cell positions (horizontal and vertical)
    x_grid = grid(columns, CELL_WIDTH_CM, x_offset, horizontal_center, x_center)
    y_grid = grid(rows, CELL_HEIGHT_CM, y_offset, vertical_center, y_center)

    # draw grid into document
    pdf.grid(x_grid, y_grid)


def create_base_pdf(filename):  # creates blank page with anchor squares
    # create new document
    pdf = canvas.Canvas(filename, bottomup=0, pagesize=A4)
    # initial setting of the writer
    pdf.setFontSize(FONT_SIZE)

    # draw rectangles for scanning
    pdf.rect(25, 25, 20, 40, fill=1)
    pdf.rect(45, 25, 20, 20, fill=1)

    pdf.rect(PAGE_WIDTH - 45, 25, 20, 40, fill=1)
    pdf.rect(PAGE_WIDTH - 65, 25, 20, 20, fill=1)

    pdf.rect(25, PAGE_HEIGHT - 65, 20, 40, fill=1)
    pdf.rect(45, PAGE_HEIGHT - 45, 20, 20, fill=1)

    pdf.rect(PAGE_WIDTH - 45, PAGE_HEIGHT - 65, 20, 40, fill=1)
    pdf.rect(PAGE_WIDTH - 65, PAGE_HEIGHT - 45, 20, 20, fill=1)
    return pdf


def create_train(filename):    # creates grids for training and testing data
    pdf = create_base_pdf(filename)

    # offset ideal for text
    text_offset_x = X_OFFSET_CM * cm + 4
    text_offset_y = Y_OFFSET_CM * cm + FONT_SIZE + 1
    y = 0

    # grid for capital letters
    capital = string.ascii_uppercase
    x = text_offset_x
    for ch in capital:
        pdf.drawString(x, text_offset_y, ch)
        x += CELL_WIDTH_CM * cm
    y += 1
    table(pdf, 26, 6, y=Y_OFFSET_CM + y, vertical_center=0)
    y += 4

    # grid for lowercase letters
    lowercase = string.ascii_lowercase
    x = text_offset_x
    for ch in lowercase:
        pdf.drawString(x, text_offset_y + y * cm, ch)
        x += CELL_WIDTH_CM * cm
    y += 1
    table(pdf, 26, 6, y=Y_OFFSET_CM + y, vertical_center=0)
    y += 4

    # grid for lowercase letters
    digits = string.digits
    x = text_offset_x + cm
    for i in range(2):
        table(pdf, 10, 3, x=x/cm - 0.1, y=Y_OFFSET_CM + y + 1, horizontal_center=0, vertical_center=0)
        for ch in digits:
            pdf.drawString(x, text_offset_y + y * cm, ch)
            x += CELL_WIDTH_CM * cm
        x += 1.8 * cm
    y += 3.5

    # grid for free text with vertical space
    y = Y_OFFSET_CM + y
    while y < int(PAGE_HEIGHT / cm) - Y_OFFSET_CM:
        table(pdf, rows=1, vertical_center=0, y=y)
        y += 0.8

    # output file
    pdf.save()


def create_test(filename):  # creates grids for testing data
    pdf = create_base_pdf(filename)

    # grid for free text with vertical space
    y = Y_OFFSET_CM
    while y < int(PAGE_HEIGHT / cm) - Y_OFFSET_CM:
        table(pdf, rows=1, vertical_center=0, y=y)
        y += 0.8
    # output file
    pdf.save()


def train_grid(pdf, i, text, rows=1, columns=26):   # create training grid
    # offset ideal for text
    text_offset_x = X_OFFSET_CM * cm + 10
    text_offset_y = Y_OFFSET_CM * cm + FONT_SIZE - 1

    # grid for capital letters
    pdf.drawString(text_offset_x, text_offset_y + cm * i, text)
    table(pdf, columns, rows, x=X_OFFSET_CM + 3.75, y=Y_OFFSET_CM + i, horizontal_center=0, vertical_center=0)
    return i + rows
