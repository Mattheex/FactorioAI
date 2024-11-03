import time
import numpy as np
import pygame
import sys


def twoDprint(A):
    for i in A:
        print("\t".join(map(str, i)))


class Case:
    def __init__(self, name, speed=0, limit=0) -> None:
        self.item = 0
        self.then = time.time()
        self.name = name
        self.speed = speed
        self.limit = limit

    def turn(self, nearby, center_pos):
        """Load in the file for extracting text."""
        pass

    def action(self):
        duration_in_s = time.time() - self.then
        return duration_in_s * 1000 > self.speed


class Dirt(Case):
    def __init__(self) -> None:
        super().__init__("dirt")

    def __str__(self) -> str:
        return self.name


class Mine(Case):
    def __init__(self) -> None:
        super().__init__("mine", 200, 100)

    def turn(self, nearby, center_pos):
        if self.action() and self.item < self.limit:
            self.item += 1
            self.then = time.time()

    def __str__(self) -> str:
        return f"{self.name} {self.item}"


class Belt(Case):
    def __init__(self, direction) -> None:
        super().__init__("belt", 600, 2)
        self.direction = direction

    def turn(self, nearby, center_pos):
        if self.action():
            self.then = time.time()
            start = None

            if self.direction == "RIGHT":
                start = (center_pos[0], center_pos[1] - 1)
            elif self.direction == "DOWN":
                start = (center_pos[0] - 1, center_pos[1])

            if nearby[start].item > 0 and self.item < self.limit:
                nearby[start].item -= 1
                self.item += 1

    def __str__(self) -> str:
        return f"{self.name} {self.item}"


class Chest(Case):
    def __init__(self) -> None:
        super().__init__("chest", 1000, 20)

    def turn(self, nearby, center_pos):
        if self.action():
            self.then = time.time()
            for row in range(nearby.shape[0]):
                for col in range(nearby.shape[1]):
                    if (row != center_pos[0] or col != center_pos[1]) and row != col:
                        if nearby[row, col].item > 0 and self.item < self.limit:
                            nearby[row, col].item -= 1
                            self.item += 1

    def __str__(self) -> str:
        return f"{self.name} {self.item}"


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
g = np.array([[Dirt() for i in range(n)] for j in range(n)])
g[0, 1] = Mine()
g[0, 2] = Belt("RIGHT")
g[1, 2] = Belt("DOWN")
g[2, 2] = Belt("DOWN")
g[2, 3] = Belt("RIGHT")
g[2, 3] = Chest()
g[0, 3] = Chest()


def get_neighbors(array, row, col):
    rows, cols = array.shape

    row_start = max(0, row - 1)
    row_end = min(rows, row + 2)
    col_start = max(0, col - 1)
    col_end = min(cols, col + 2)

    neighbors = array[row_start:row_end, col_start:col_end]

    center_position = (row - row_start, col - col_start)

    return neighbors, center_position


while running:
    # Fill the background color
    screen.fill(BACKGROUND_COLOR)

    # Loop through the array and render each number
    for row in range(n):
        for col in range(n):

            nearby, center_pos = get_neighbors(g, row, col)
            g[row, col].turn(nearby, center_pos)

            number = g[row, col]
            text_surface = font.render(str(number), True, TEXT_COLOR)
            x = col * CELL_SIZE + (WIDTH - n * CELL_SIZE) // 2
            y = row * CELL_SIZE + (HEIGHT - n * CELL_SIZE) // 2
            # Draw the text onto the screen
            screen.blit(text_surface, (x + CELL_SIZE // 4, y + CELL_SIZE // 4))
            # Draw a border around each cell
            pygame.draw.rect(screen, TEXT_COLOR, (x, y, CELL_SIZE, CELL_SIZE), 1)

    # Check for quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()
