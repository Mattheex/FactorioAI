import copy
from csv import Error
from typing import overload
from pyparsing import C
from regex import B
from traitlets import Instance
from mcts import MCTS
from var import CELL_NUMBER, DOWN, LEFT, RIGHT, TOP
from entities import Belt, Case, Supplier, Vendor
import numpy as np


class Board:
    def __init__(self, board = None) -> None:
        self.all_items = self._all_items()
        self.board = board if board else self.reset()
        self.length_penality = 1
        self.last_move = None
        self.inverse_direction = {TOP: DOWN, DOWN: TOP, LEFT: RIGHT, RIGHT: LEFT}

    def reset(self) -> list[Case]:
        return [
            Case(x + y * CELL_NUMBER, True)
            for y in range(CELL_NUMBER)
            for x in range(CELL_NUMBER)
        ]

    def _all_items(self):
        machine = [Supplier, Vendor, Belt]
        direction = [DOWN, RIGHT, TOP, LEFT]
        product = []
        for m in machine:
            if m == Case or m == Vendor:
                product.append((m, None))
            else:
                for d in direction:
                    product.append((m, d))
        return product

    def terminate(self):
        for cell in self.board:
            if isinstance(cell, Case):
                return False
        return True

    def add2(self, entity, x, y, direction=None):
        coor = x + y * CELL_NUMBER
        self.add(entity, coor, direction)

    def add(self, entity, coor, direction=None, empty=False):
        entity = copy.deepcopy(entity)
        if entity != Case:
            self.board[coor] = entity(coor, direction)
        else:
            self.board[coor] = entity(coor, empty)

    def __str__(self):
        s = ""
        for y in range(CELL_NUMBER):
            for x in range(CELL_NUMBER):
                s += str(self.board[x + y * CELL_NUMBER]) + " "
            s += "\n"
        return s

    def total_reward(self):
        t_r = []
        total_cell = 0
        #print(self)
        for start_cell in self.board:
            if isinstance(start_cell, Supplier):
                if start_cell.speed is not None:
                    path = start_cell.coor + start_cell.speed
                    r = self.reward(path, start_cell, length=self.length_penality)
                    if r != -1:
                        t_r.append(r)
            if not start_cell.get_empty():
                total_cell += 0.05
        return sum(t_r) - total_cell

    def reward(self, path, previous, length, reward=0.0):
        # print(f"Reward path {path} previous {previous} length {length} reward {reward}")
        if (
            path >= len(self.board)
            or path < 0
            or length / self.length_penality > CELL_NUMBER * CELL_NUMBER
        ):
            return -1
        new_cell = self.board[path]
        if type(new_cell) == Supplier:
            return -1
        if type(new_cell) == Case:
            return 0
        if type(new_cell) == Vendor:
            return 1
        elif type(new_cell) == Belt:
            if type(previous) == Supplier:
                reward += 0.5
            if type(previous) == Belt:
                if self.inverse_direction[previous.direction] == new_cell.direction:  # type: ignore
                    return -1
                reward += 0.25
            return self.reward(
                path + new_cell.speed,
                new_cell,
                length + self.length_penality,
                reward=reward,
            )
        else:
            raise Exception(f"Error path {new_cell}")

    def make_move(self, position: int, item) -> "Board":
        board = Board(copy.deepcopy(self.board))
        board.last_move = (position, item)
        board.add(item[0], position, direction=item[1])
        # print(board)

        return board

    def get_empty_pos(self) -> list[int]:  # -> list[Any]:# -> list[Any]:# -> list[Any]:
        return [cell.coor for cell in self.board if cell.get_empty()]

    def actions(self):
        actions = []
        empty_pos = self.get_empty_pos()
        for pos in empty_pos:
            for item in self.all_items:
                actions.append(self.make_move(pos, item))
                #print(actions[-1].last_move)
        return actions

    def game(self):
        mcts = MCTS()
        for turn in range(CELL_NUMBER * CELL_NUMBER):
            print(f"Turn {turn + 1}")

            best_move = mcts.search(self)
            self = best_move.board
            print(self)


# print(product)
b = Board()
b.game()
"""b.add(Supplier, 2, DOWN)
b.add(Belt, 2 + CELL_NUMBER, RIGHT)
b.add(Belt, 2 + CELL_NUMBER + 1, RIGHT)
b.add(Vendor, 2 + CELL_NUMBER + 1 + 1)
print(b)
print(b.total_reward())"""
