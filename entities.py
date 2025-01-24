from __future__ import annotations

import time
from collections import UserList

import pygame

import numpy as np

from var import TOP, RIGHT, DOWN, LEFT, CELL_SIZE, ITEM_SIZE,SPEED


def get_center(coor, size):
    return coor + (np.array((size, size))) // 2


class Case:
    def __init__(self, game, coor, class_id=0, path="case.png", name="Nothing", direction=None):
        self.game = game
        self.size = CELL_SIZE
        self.class_id = class_id
        self.coor = coor * np.array((self.size, self.size))
        self.name = name
        self.direction = direction
        self.img = pygame.image.load(path).convert_alpha()
        self.img = pygame.transform.scale(self.img, (self.size, self.size))

        self.map = {
            LEFT: np.array([-1, 0]),
            TOP: np.array([0, -1]),
            DOWN: np.array([0, 1]),
            RIGHT: np.array([1, 0]),
        }
        self.speed = SPEED

        if direction is not None:
            self.speed_vector = self.map[direction] * np.array((self.speed, self.speed))

            if direction != TOP:
                rotate_map = {
                    LEFT: 90,
                    DOWN: 180,
                    RIGHT: -90
                }
                self.img = pygame.transform.rotate(self.img, rotate_map[direction])

    def turn(self):
        pass

    def get_center(self):
        return get_center(self.coor, self.size)

    def distance(self, item):
        center_obj = self.get_center()
        center_item = item.get_center()
        return np.linalg.norm(center_obj - center_item)

    def __str__(self) -> str:
        return self.name


class Belt(Case):
    def __init__(self, game, coor, direction) -> None:
        super().__init__(game, coor, 1, "belt.png", "Belt", direction)
        self.items = []

    def __str__(self) -> str:
        return f"{self.name} {self.coor}"


class Maker(Case):
    def __init__(self, game, coor, class_id, path, name, direction=None, prod_time=1000) -> None:
        super().__init__(game, coor, class_id, path, name, direction)
        self.then = time.time()
        self.prod_time = prod_time

    def action(self) -> bool:
        duration_in_s = time.time() - self.then
        return duration_in_s * 1000 > self.prod_time


class Supplier(Maker):
    def __init__(self, game, coor, direction) -> None:
        super().__init__(game, coor, 2, "supplier.png", "Supplier", direction, 1000)

    def turn(self) -> None:
        if self.action():
            self.then = time.time()
            item = Item()
            item.set_center(self.get_center())
            item.set_vector(self.speed_vector)
            self.game.items.append(item)

    def __str__(self) -> str:
        return f"{self.name} {self.coor}"


class Vendor(Case):
    def __init__(self, game, coor, direction) -> None:
        super().__init__(game, coor, 3, "vendor.png", "Vendor")

    def sell(self, item):
        self.game.earn += 10
        del item

    def __str__(self) -> str:
        return f"{self.name} {self.coor}"


class Item:
    def __init__(self, coor=(0, 0)) -> None:
        self.class_id = 4
        self.id = np.random.randint(100)
        self.coor = coor
        self.vector = None
        self.size = ITEM_SIZE
        self.stock = False
        self.overlap = False
        self.action = False
        self.img = pygame.image.load("gold.png").convert_alpha()
        self.img = pygame.transform.scale(self.img, (self.size, self.size))

    def move(self, delta) -> None:
        if not self.stock and self.vector is not None:
            if self.vector[0] == 50:
                pass
            self.coor = self.coor + self.vector * delta / 1000

    def set_vector(self, vector):
        self.vector = vector

    def set_center(self, center):
        self.coor = center - self.size // 2

    def get_center(self):
        return get_center(self.coor, self.size)

    def __str__(self):
        return f'item {self.id} {self.coor} {self.vector} {self.stock} {self.overlap}'


class ItemList(UserList):
    def remove(self, s=None):
        super().remove(s)
        del s
