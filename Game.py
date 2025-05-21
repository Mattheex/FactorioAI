from itertools import count

import numpy as np
from numpy import linalg as LA
import pygame
import sys
import gymnasium as gym
from typing import Optional
from gymnasium.spaces import Tuple, Box, Discrete

from entities import Belt, Case, Supplier, Vendor, Maker, ItemList
from var import (
    TOP,
    RIGHT,
    DOWN,
    LEFT,
    WIDTH,
    HEIGHT,
    CELL_SIZE,
    FONT_SIZE,
    BACKGROUND_COLOR,
    TEXT_COLOR,
    CELL_NUMBER,
    ITEM_SIZE,
    SPEED,
)

"""def twoDprint(A):
  
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

    return neighbors, center_position"""

# pygame.init()

# Set up the pygame window


# Main loop


def decode_id(unique_id):
    a = unique_id // (5 * 4 * 4)
    b = (unique_id % (5 * 4 * 4)) // (4 * 4)
    c = (unique_id % (4 * 4)) // 4
    d = unique_id % 4
    return a, b, c, d


class Game(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(Game, self).__init__()
        self.n = CELL_NUMBER
        self.earn = 0
        self.grid = np.empty((self.n, self.n)).tolist()
        self.items = ItemList()
        self.running = True
        self.clock = pygame.time.Clock()
        self.tolerance = 2
        self.action_space = gym.spaces.Tuple(
            (Discrete(2), Discrete(self.n * self.n), Discrete(4), Discrete(4))
        )
        self.observation_space = Box(
            low=0, high=4, shape=(self.n, self.n), dtype=np.int64
        )
        self.window = None

    def add_items(self, x, y, obj, direction=None):
        coor = np.array((x, y))
        self.grid[x][y] = obj(self, coor, direction=direction)

    def exemple_game(self, ex=0):
        for x in range(self.n):
            for y in range(self.n):
                self.add_items(x, y, Case)
        if ex == 1:
            self.add_items(0, 1, Supplier, DOWN)
            self.add_items(0, 2, Belt, RIGHT)
            self.add_items(1, 2, Belt, RIGHT)
            self.add_items(2, 2, Belt, DOWN)
            self.add_items(2, 3, Vendor)

    def case_not_null(self):
        return sum(1 for obj in sum(self.grid, []) if type(obj) is not Case)

    def action(self, a, b, c, d):
        if a != 0:
            x = b // CELL_NUMBER
            y = b % CELL_NUMBER
            rot_map = [LEFT, TOP, DOWN, RIGHT]
            cell_map = [Case, Belt, Supplier, Vendor]
            self.add_items(x, y, cell_map[c], rot_map[d])

    def step(self, code):
        # a, b, c, d = decode_id(action)
        pre_earn = self.earn

        self.action(*code)

        # 0 0 0 0 -> Case (0,0)
        # 0 0 0 1 ->
        # 1 0 1 1 -> Belt (1,0) LEFT
        # 4 4 3 0 -> Vendor (4,4)

        # pre_earn = self.earn
        delta = self.clock.tick() * 2
        self.tolerance = delta * SPEED / 1000

        # delta = 6
        for obj in sum(self.grid, []):
            for item in self.items:
                if not item.action:
                    if isinstance(obj, Belt):
                        dist = abs(obj.distance(item))
                        if dist < self.tolerance:
                            if item not in obj.items:
                                item.set_vector(obj.speed_vector)
                                item.set_center(obj.get_center())
                                obj.items.append(item)
                                item.action = True
                        else:
                            if dist < obj.size:
                                item.action = True
                            if item in obj.items:
                                obj.items.remove(item)

                    elif isinstance(obj, Vendor):
                        dist = abs(obj.distance(item))
                        if dist < self.tolerance:
                            obj.sell(item)
                            self.items.remove(item)
                        if dist < obj.size:
                            item.action = True

                    elif type(obj) is Case:
                        dist = abs(obj.distance(item))
                        if dist < self.tolerance:
                            self.items.remove(item)
                        if dist < obj.size:
                            item.action = True

            obj.turn(delta)

        for item in self.items:
            item.move(delta)
            item.action = False

        # An environment is completed if and only if the agent has reached the target
        terminated = self.earn >= 10
        truncated = False
        reward = self.calculate_time()
        if len(reward) == 0:
            reward = -1
        else:
            total_reward = 0
            for r in reward:  
                total_reward += r - 1 / (CELL_NUMBER * 2 - 1)
            total_reward /= len([item for row in self.grid for item in row if type(item) is not Case])
        observation = self._get_obs()
        info = {}

        # print(reward)

        return observation, reward, terminated, truncated, info

    def board_to_image(self):
        rotate_map = {TOP: 0, LEFT: 1, DOWN: 2, RIGHT: 3, None: 0}
        img = np.zeros((3, HEIGHT, WIDTH), dtype=int)
        for obj in sum(self.grid, []):
            if type(obj) is not Case:
                coor = obj.coor.astype(int)
                img[0, coor[0] : coor[0] + obj.size, coor[1] : coor[1] + obj.size] = (
                    obj.class_id
                )
        for obj in sum(self.grid, []):
            img[1, coor[0] : coor[0] + obj.size, coor[1] : coor[1] + obj.size] = (
                rotate_map[obj.direction]
            )
        for item in self.items:
            coor = item.coor.astype(int)
            img[2, coor[0] : coor[0] + item.size, coor[1] : coor[1] + item.size] = (
                item.class_id
            )

        trunc = round(ITEM_SIZE)
        img = img[:,::trunc, ::trunc]

        """for i in img[2]:
            print(" ".join(map(str, i)))
        print()"""
        #img = np.expand_dims(img, axis=0)
        return img

    def _get_obs(self):
        return self.board_to_image()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.exemple_game(1)
        self.items = ItemList()
        self.clock = pygame.time.Clock()
        self.earn = 0

        observation = self._get_obs()

        return observation, {}

    def calculate_time(self):
        times = []
        # Flatten grid and create list of suppliers once
        flat_grid = [item for row in self.grid for item in row]
        suppliers = [s for s in flat_grid if isinstance(s, Supplier)]
        objects = flat_grid  # Keep reference to all objects for collision checking

        for supplier in suppliers:
            # Initialize starting position
            item = np.array(supplier.get_center())
            direction = supplier.map[supplier.direction]
            item += direction * np.array((CELL_SIZE, CELL_SIZE))

            # Track item movement
            t = 0
            while t < CELL_NUMBER * 2:  # Maximum movement limit
                # Use numpy broadcasting for faster distance calculation
                item_pos = np.array([item])  # Reshape for broadcasting
                obj_positions = np.array([obj.get_center() for obj in objects])
                distances = np.abs(np.linalg.norm(obj_positions - item_pos, axis=1))

                # Find first collision
                collision_indices = np.where(distances < self.tolerance)[0]
                #print(collision_indices)
                if len(collision_indices) > 0:
                    collided_obj = objects[collision_indices[0]]

                    if type(collided_obj) is Case :
                        break
                    elif type(collided_obj) is Vendor:
                        times.append(t + 1)
                        break

                    # Update direction based on collision
                    direction = collided_obj.map[collided_obj.direction]

                # Update position
                item += direction * np.array((CELL_SIZE, CELL_SIZE))
                t += 1

        return times

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Array Display")
            font = pygame.font.Font(None, FONT_SIZE)
        self.window.fill(BACKGROUND_COLOR)

        # agent.turn(self.grid,self.items)

        text_surface = font.render(str(self.earn), True, TEXT_COLOR)
        x = 10
        y = 10
        self.window.blit(text_surface, (x, y))

        # agent.observe(g)

        for obj in sum(self.grid, []):
            self.window.blit(obj.img, tuple(obj.coor))

        for item in self.items:
            self.window.blit(item.img, tuple(item.coor))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()