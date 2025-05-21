import copy
from csv import Error
from typing import overload
from regex import B
from var import CELL_NUMBER, DOWN, LEFT, RIGHT, TOP
from entities import Belt, Case, Supplier, Vendor
import numpy as np


class Board:
    def __init__(self):
        self.board = []
        self.reset()

    def reset(self):
        self.board = [
            Case(x + y * CELL_NUMBER, True)
            for y in range(CELL_NUMBER)
            for x in range(CELL_NUMBER)
        ]

    def add2(self, entity, x, y, direction=None):
        coor = x + y * CELL_NUMBER
        self.add(entity, coor, direction)

    def add(self, entity, coor, direction=None, empty=False):
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
        for start_cell in self.board:
            if isinstance(start_cell, Supplier):
                if start_cell.speed is not None:
                    path = start_cell.coor + start_cell.speed
                    r = self.reward(path, 1)
                    if r != -1:
                        t_r.append(r)
        return t_r

    def reward(self, path, length):
        new_cell = self.board[path]
        if type(new_cell) == Case:
            return -1
        if type(new_cell) == Vendor:
            return 1 / length
        elif type(new_cell) == Belt:
            return self.reward(path + new_cell.speed, length + 1)
        else:
            raise Exception(f"Error path {new_cell}")


class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []

    def possible_move(self):
        pass

    def addChildren(self, node):
        self.children.append(node)

    def get_empty_pos(self):
        return [cell.coor for cell in self.board.board if cell.get_empty()]


class Tree:
    def __init__(self, root,machine_dir):
        self.root = root
        self.machine_dir = machine_dir

    def search_recursive(self, node, depth=0, total_node=0):
        empty_pos = node.get_empty_pos()

        #print(node.board)

        if len(empty_pos) == 0:
            print("depth", depth, 'total_node',total_node)
            return total_node

        for pos in empty_pos:
            new_board = copy.deepcopy(node.board)
            for (machine,direction) in self.machine_dir:
                new_board.add(machine, pos, direction)
                child = Node(new_board, node)
                node.addChildren(child)
                last_total = self.search_recursive(
                    child, depth + 1, total_node + 1
                )
                total_node = max(last_total,total_node)

                if total_node == 10000000:
                    raise Error('Stop')

                
                #print("depth", depth, 'total_node',total_node)
                
                # print(f"{pos / CELL_NUMBER*CELL_NUMBER}.3f")
        
        return total_node


if __name__ == "__main__":
    machine = [Case, Supplier, Vendor, Belt]
    direction = [DOWN, RIGHT, TOP, LEFT]
    product = []
    for m in machine:
        if m == Case or m == Vendor:
            product.append((m,None))
        else:
            for d in direction:
                product.append((m,d))

    print(product)
    tree = Tree(Node(Board(), 0),product)
    tree.search_recursive(tree.root)
