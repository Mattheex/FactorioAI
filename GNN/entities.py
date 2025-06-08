from __future__ import annotations
import copy
from var import (
    CELL_NUMBER,
    COPPER,
    COPPER_CABLE,
    RIGHT_DIRECTION,
    TOP,
    RIGHT,
    DOWN,
    LEFT,
    OPPOSITE_DIRECTION,
)


class Item:
    def __init__(self, coor, quantity, object,pre_cell):
        self.coor = coor
        self.quantity = quantity
        self.object = object
        self.finish = False
        self.reward = 0
        self.pre_cell = pre_cell

    def move(self, speed):
        self.coor += speed

    def div_q(self, ratio):
        self.quantity = self.quantity * ratio

    def transform(self, object):
        self.object = object

    def set_finish(self, reward):
        self.finish = True
        self.reward = reward * self.quantity

    def __str__(self) -> str:
        return f"Item {self.coor} {self.quantity} {self.object} {self.finish}"


class Case:
    def __init__(self, coor, id, direction=None) -> None:
        self.name = "Case"
        self.coor = coor
        self.id = id

    def details(self) -> str:
        return f"{self.name} {self.coor}"

    def get_empty(self):
        return self.id == -1
    
    def is_supplier(self):
        return False

    def action(self, item: Item, _) -> Item:
        if self.get_empty():
            item.set_finish(0)
        else:
            item.set_finish(0)
        return item

    def __str__(self) -> str:
        return f"{self.name[0]}o"


class Machine:
    def __init__(self, name, coor, id, ratio, direction=None) -> None:
        self.name = name
        self.coor = coor
        self.id = id
        self.ratio = ratio
        self.direction = {"start": None, "end": None}
        self.speed = None

        self.map = {
            LEFT: -1 if coor % CELL_NUMBER != 0 else -CELL_NUMBER * CELL_NUMBER,
            TOP: -CELL_NUMBER,
            DOWN: CELL_NUMBER,
            RIGHT: (
                1
                if coor % CELL_NUMBER != CELL_NUMBER - 1
                else CELL_NUMBER * CELL_NUMBER
            ),
        }

        if direction:
            self.direction = {"start": OPPOSITE_DIRECTION[direction], "end": direction}
            self.speed = self.map[self.direction["end"]]

    def details(self) -> str:
        return f"{self.name} {self.coor} {self.direction}"

    def get_empty(self):
        return False

    def is_supplier(self):
        return type(self) == Supplier

    def action(self, item: Item) -> Item:
        item.move(self.speed)
        item.div_q(self.ratio)
        return item

    def check_entry(self, pre_cell: Machine):
        if self.direction["start"] and pre_cell.direction["end"]:
            return (
                self.direction["start"] == OPPOSITE_DIRECTION[pre_cell.direction["end"]]
            )
        print(f"error check_entry {self.direction} / {pre_cell.direction}")
        return None

    def __str__(self) -> str:
        self.map_str = {LEFT: "<", TOP: "^", DOWN: "v", RIGHT: ">", None: "o"}
        return f"{self.name[0]}{self.map_str[self.direction['end']]}"


class Belt(Machine):
    def __init__(self, coor, id, direction) -> None:
        super().__init__("Belt", coor, id, 1, direction)

    def action(self, item, pre_cell):
        if self.direction["start"] == pre_cell.direction["end"]:
            item.set_finish(0)
            return item
        return super().action(item)


class Supplier(Machine):
    def __init__(self, coor, id, direction) -> None:
        super().__init__("Mine", coor, id, 1, direction)

    def action(self, item=None, _=None):
        if item:
            item.set_finish(0)
            return item
        else:
            return Item(coor=self.coor + self.speed, quantity=self.ratio, object=COPPER, pre_cell=self)


class Vendor(Machine):
    def __init__(self, coor, id, _) -> None:
        super().__init__("Vendor", coor, id, 1)
        self.price = {COPPER: 1, COPPER_CABLE: 10}

    def action(self, item, _):
        item.set_finish(self.price[item.object])
        return item


class Transformer(Machine):
    def __init__(self, coor, id, direction) -> None:
        super().__init__("Transformer", coor, id, 0.5, direction)

    def action(self, item, pre_cell):
        if item.object == COPPER and self.check_entry(pre_cell):
            item = super().action(item)
            item.transform(COPPER_CABLE)
        else:
            item.set_finish(0)
        return item


class Splitter(Machine):
    def __init__(self, coor, id, direction) -> None:
        super().__init__("Splitter", coor, id, 0.5, direction)
        self.split_direction = [self.direction["end"], RIGHT_DIRECTION[self.direction["end"]]]  # type: ignore
        self.split_speed = [self.map[direct] for direct in self.split_direction]  # type: ignore

    def action(self, item: Item, _) -> list[Item]:
        items = [item, copy.copy(item)]
        for speed, it in zip(self.split_speed, items):
            self.speed = speed
            it = super().action(it)
        return items
