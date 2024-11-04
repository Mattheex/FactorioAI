import numpy as np
import pygame
import sys
from random import randint, uniform, choice

from entities import *
from var import *


def twoDprint(A):
    for i in A:
        print("\t".join(map(str, i)))


def get_neighbors(array, row, col):
    rows, cols = array.shape

    row_start = max(0, row - 1)
    row_end = min(rows, row + 2)
    col_start = max(0, col - 1)
    col_end = min(cols, col + 2)

    neighbors = array[row_start:row_end, col_start:col_end]

    center_position = np.array([row - row_start, col - col_start])

    return neighbors, center_position


class Agent:
    def __init__(self, states) -> None:
        self.learningRate = 0.1
        self.learningFactor = 0
        self.states = states
        self.action = [["B", "RIGHT"], ["B", "DOWN"], ["B", "TOP"], ["B", "LEFT"]]

    def createEntities(self, c):
        if c[0] == "B":
            return Belt(c[1])

    def turn(self, g):
        learning = uniform(0, 1)

        if learning > self.learningFactor:
            self.createEntities(choice(self.action))
        else:
            self.createEntities(np.max(self.states, 0))

        self.learningFactor += self.learningRate

    def reward(self, g):
        r = 0
        for row in range(n):
            for col in range(n):
                if isinstance(g[row, col], Belt):

                    neighbors, center_position = get_neighbors(g, row, col)

                    contains_instance = any(
                        isinstance(item, Mine) for items in neighbors for item in items
                    )

                    if contains_instance:
                        r += 10

        return r


pygame.init()

# Define some constants
WIDTH, HEIGHT = 800, 600  # Window dimensions
CELL_SIZE = 90  # Size of each cell
FONT_SIZE = 24  # Font size for text
BACKGROUND_COLOR = (30, 30, 30)  # Background color for window
TEXT_COLOR = (255, 255, 255)  # Text color for numbers

# Set up the pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Array Display")
font = pygame.font.Font(None, FONT_SIZE)

# Main loop

running = True
n = 5
g = np.array([[Dirt() for _ in range(n)] for _ in range(n)])
g[0, 1] = Mine()
g[0, 2] = Belt(LEFT, DOWN)
g[1, 2] = Belt(TOP,DOWN)
g[2, 2] = Belt(TOP,RIGHT)
# g[2, 3] = Belt("RIGHT")
g[2, 3] = Chest()

states = np.array([[0 for _ in range(n)] for _ in range(n)])

agent = Agent(states)

print(agent.reward(g))


while running:
    screen.fill(BACKGROUND_COLOR)

    #agent.turn(g)

    for row in range(n):
        for col in range(n):

            nearby, center_pos = get_neighbors(g, row, col)
            g[row, col].turn(nearby, center_pos)

            entity = g[row, col]

            text_surface = font.render(str(entity), True, TEXT_COLOR)
            x = col * CELL_SIZE + (WIDTH - n * CELL_SIZE) // 2
            y = row * CELL_SIZE + (HEIGHT - n * CELL_SIZE) // 2

            text_width = font.size(str(entity))[0]
            text_height = font.size(str(entity))[1]
            x_text = x + (CELL_SIZE - text_width) // 2
            y_text = y + (CELL_SIZE - text_height) // 2
            screen.blit(text_surface, (x_text, y_text))

            pygame.draw.rect(screen, TEXT_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1)

    #agent.observe(g)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()
