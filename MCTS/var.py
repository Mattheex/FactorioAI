from math import floor


RIGHT = "Right"
DOWN = "Down"
TOP = "Top"
LEFT = "Left"
WIDTH, HEIGHT = 600, 600  # Window dimensions
CELL_NUMBER = 2
CELL_SIZE = round(WIDTH / CELL_NUMBER)  # Size of each cell
ITEM_SIZE = round(30 * WIDTH / 450)
SPEED = round(100 * WIDTH / 600)
FONT_SIZE = 24  # Font size for text
BACKGROUND_COLOR = (30, 30, 30)  # Background color for window
TEXT_COLOR = (255, 255, 255)  # Text color for numbers
RENDER = None