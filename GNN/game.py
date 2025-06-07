import copy
from io import StringIO
import json
from mcts import MCTS
from var import CELL_NUMBER, DOWN, LEFT, RIGHT, TOP
from entities import Belt, Case, Item, Splitter, Supplier, Transformer, Vendor
import numpy as np

class Board:
    def __init__(self, all_items=None, board=None) -> None:
        self.all_items = all_items if all_items else self._all_items()
        self.board = board if board else self.reset()
        self.max_path = CELL_NUMBER + CELL_NUMBER
        self.length_penality = 1
        self.last_move = None
        # self.inverse_direction = {TOP: DOWN, DOWN: TOP, LEFT: RIGHT, RIGHT: LEFT}

    def reset(self) -> list[Case]:
        return [Case(case, -1) for case in range(CELL_NUMBER * CELL_NUMBER)]

    def _all_items(self):
        machine = [Supplier, Vendor, Belt, Splitter, Transformer]
        direction = [DOWN, RIGHT, TOP, LEFT]
        product = []
        for m in machine:
            if m == Case or m == Vendor:
                product.append(
                    {
                        "id": len(product),
                        "entity": m,
                        "direction": None,
                        "str": m.__name__,
                    }
                )
            else:
                for d in direction:
                    product.append(
                        {
                            "id": len(product),
                            "entity": m,
                            "direction": d,
                            "str": m.__name__,
                        }
                    )

        #print(product)
        cleaned_data = [{k: v for k, v in obj.items() if k != "entity"} for obj in product]
        json_object = json.dumps(cleaned_data, indent=4)
        with open("sample.json", "w") as outfile:
            outfile.write(json_object)
        return product

    def terminate(self):
        for cell in self.board:
            if cell.get_empty():
                return False
        return True

    def add(self, entity, coor, id=-1, direction=None):
        self.board[coor] = entity(coor, id, direction)

    def __str__(self):
        s = ""
        for y in range(CELL_NUMBER):
            for x in range(CELL_NUMBER):
                s += str(self.board[x + y * CELL_NUMBER]) + " "
            s += "\n"
        return s

    def display_index(self):
        """Display the board with entity IDs"""
        # Use StringIO for efficient string concatenation
        s = StringIO()
        for y in range(CELL_NUMBER):
            for x in range(CELL_NUMBER):
                s.write(f"{self.board[x + y * CELL_NUMBER].id} ")
            s.write("\n")
        return s.getvalue()
    
    def board_from_index(self, lines:list[str]):
        """Create a board from a string representation of indices"""
        # Split the input string into lines

        
        for y, line in enumerate(lines):
            if type(line) is str or type(line) is np.str_:
                line  = line.strip().split(" ")
                #print(f"Processing line {y}: {line}")
            for x, cell_id in enumerate(line):
                #print(f"Processing cell at ({x}, {y}): {cell_id}")
                cell_id = int(cell_id)
                if cell_id >= len(self.all_items):
                    raise ValueError(f"Invalid cell ID {cell_id} at position ({x}, {y})")
                for item in self.all_items:
                    if item["id"] == cell_id:
                        self.add(item["entity"], x + y * CELL_NUMBER, id=cell_id, direction=item["direction"])
                        break
                #item = self.all_items[cell_id]
        

    def total_reward(self):
        t_r = []
        total_cell = 0
        for cell in self.board:
            if cell.is_supplier():
                item = cell.action()
                r = self.reward(item, cell, 0)
                t_r.append(r)
                #print(f"reward end {cell.coor} {item} {r}")
            if not cell.get_empty():
                total_cell += 0.02
        return sum(t_r) - total_cell

    def reward(self, item: Item, pre_cell, length):
        if item.coor >= self.max_path or item.coor < 0 or length > self.max_path:
            return -1 * item.quantity

        #print(f"reward {pre_cell} {item} {length}")

        cell = self.board[item.coor]
        item = cell.action(item, pre_cell)

        if isinstance(item, list):
            total = 0
            for it in item:
                total += self.reward(it, cell, length + 1)
            return total
        elif item.finish:
            return item.reward
        else:
            return self.reward(item, cell, length + 1)

    def make_move(self, position: int, item) -> "Board":
        board = Board(board=copy.deepcopy(self.board), all_items=self.all_items)
        board.last_move = (position, item)
        board.add(item["entity"], position, direction=item["direction"], id=item["id"])

        return board

    def get_empty_pos(self) -> list[int]:  # -> list[Any]:# -> list[Any]:# -> list[Any]:
        return [cell.coor for cell in self.board if cell.get_empty()]

    def actions(self):
        actions = []
        empty_pos = self.get_empty_pos()
        for pos in empty_pos:
            for item in self.all_items:
                actions.append(self.make_move(pos, item))
                # print(actions[-1].last_move)
        return actions

    def game(self):
        mcts = MCTS()
        saves = ''
        for turn in range(CELL_NUMBER * CELL_NUMBER):
            print(f"Turn {turn + 1}")

            best_move = mcts.search(self)
            self = best_move.board

            saves += f"{self.display_index()}\n"


            print(f"reward {self.total_reward()}")
            print(self)

        return saves


# print(product)
#b = Board()
#b.game()
"""b.add(Supplier, 2, -1, DOWN)
b.add(Splitter, 2 + CELL_NUMBER, -1, DOWN)
b.add(Belt, 2 + 2 * CELL_NUMBER, -1, RIGHT)
b.add(Transformer, 3 + 2 * CELL_NUMBER, -1, RIGHT)
b.add(Vendor, 4 + 2 * CELL_NUMBER, -1)
b.add(Vendor, 1 + CELL_NUMBER, -1)

print(b)
print(b.total_reward())"""
