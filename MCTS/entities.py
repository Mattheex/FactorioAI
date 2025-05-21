from __future__ import annotations

import numpy as np

from var import CELL_NUMBER, TOP, RIGHT, DOWN, LEFT


class Case:
    def __init__(self, coor,empty=False) -> None:
        self.name = "Nothing"
        self.coor = coor
        self.id = 0
        self.empty = empty

    def details(self) -> str:
        return f"{self.name} {self.coor}"
    
    def get_empty(self):
        return self.empty

    def __str__(self) -> str:
            return f'{self.name[0]}o'

class Machine:
    def __init__(self, name, coor, id, ratio, direction=None) -> None:
        self.name = name
        self.coor = coor
        self.id = id
        self.ratio = ratio
        self.direction = direction

        self.map = {
            LEFT: -1,
            TOP: -CELL_NUMBER,
            DOWN: CELL_NUMBER,  
            RIGHT: 1,
        }

        if direction:
            self.speed = self.map[direction]
        else:
            self.speed = None
    
    def details(self) -> str:
        return f"{self.name} {self.coor} {self.direction}"
    
    def get_empty(self):
         return False

    def __str__(self) -> str:
            self.map_str = {
            LEFT: '>',
            TOP: '^',
            DOWN: 'v',  
            RIGHT: '<',
            None:'o'
        }
            return f'{self.name[0]}{self.map_str[self.direction]}'


class Belt(Machine):
    def __init__(self, coor, direction) -> None:
        super().__init__("Belt", coor, 1, 1, direction)


class Supplier(Machine):
    def __init__(self, coor, direction) -> None:
        super().__init__("Supplier", coor, 2, 1, direction)


class Vendor(Machine):
    def __init__(self, coor,direction) -> None:
        super().__init__("Vendor", coor, 3, 1)
