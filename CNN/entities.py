from __future__ import annotations

import numpy as np
from pyparsing import C

from var import CELL_NUMBER, TOP, RIGHT, DOWN, LEFT


class Case:
    def __init__(self, coor,id=None,direction=None) -> None:
        self.name = "Nothing"
        self.coor = coor
        self.id = id

    def details(self) -> str:
        return f"{self.name} {self.coor}"

    def get_empty(self):
        return self.id == -1

    def __str__(self) -> str:
        return f"{self.name[0]}o"


class Machine:
    def __init__(self, name, coor, id, ratio, direction=None) -> None:
        self.name = name
        self.coor = coor
        self.id = id
        self.ratio = ratio
        self.direction = direction

        self.map = {
            LEFT: -1 if coor % CELL_NUMBER != 0 else -CELL_NUMBER * CELL_NUMBER,
            TOP: -CELL_NUMBER,
            DOWN: CELL_NUMBER,
            RIGHT: 1 if coor % CELL_NUMBER != CELL_NUMBER - 1 else CELL_NUMBER * CELL_NUMBER,
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
        self.map_str = {LEFT: "<", TOP: "^", DOWN: "v", RIGHT: ">", None: "o"}
        return f"{self.name[0]}{self.map_str[self.direction]}"


class Belt(Machine):
    def __init__(self, coor, id,direction) -> None:
        super().__init__("Belt", coor, id, 1, direction)


class Supplier(Machine):
    def __init__(self, coor, id,direction) -> None:
        super().__init__("Supplier", coor, id, 1, direction)


class Vendor(Machine):
    def __init__(self, coor, id,direction) -> None:
        super().__init__("Vendor", coor, id, 1)
