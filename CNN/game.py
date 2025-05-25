import copy
import os
import threading
from random import choice
from var import CELL_NUMBER, DOWN, LEFT, RIGHT, TOP
from entities import Belt, Case, Supplier, Vendor
import numpy as np
from collections import deque
from io import StringIO


class Board:
    # Static inverse direction mapping - no need to recreate for each instance
    INVERSE_DIRECTION = {TOP: DOWN, DOWN: TOP, LEFT: RIGHT, RIGHT: LEFT}
    
    def __init__(self, board=None, all_items=None) -> None:
        self.all_items = all_items if all_items is not None else self._all_items()
        self.board = board if board is not None else self.reset()
        self.length_penality = 1
        self.last_move = None
        # Cache empty positions to avoid recalculating
        self._empty_positions = None
        
    def reset(self) -> list[Case]:
        """Create a new empty board with all cells initialized as empty"""
        return [
            Case(x + y * CELL_NUMBER, -1)
            for y in range(CELL_NUMBER)
            for x in range(CELL_NUMBER)
        ]

    def _all_items(self):
        """Generate all possible items that can be placed on the board"""
        machine = [Case, Supplier, Vendor, Belt]
        direction = [DOWN, RIGHT, TOP, LEFT]
        product = []
        
        # Pre-calculate the product list size to avoid repeated len() calls
        product_id = 0
        
        for m in machine:
            if m == Case or m == Vendor:
                product.append({"id": product_id, "entity": m, "direction": None})
                product_id += 1
            else:
                for d in direction:
                    product.append({"id": product_id, "entity": m, "direction": d})
                    product_id += 1
        return product

    def terminate(self) -> bool:
        """Check if the board is completely filled"""
        # Use cached empty positions if available
        if self._empty_positions is not None:
            return len(self._empty_positions) == 0
            
        # Otherwise check each cell
        for cell in self.board:
            if cell.get_empty():
                return False
        return True

    def add(self, entity, coor, id=None, direction=None):
        """Add an entity to the board at the specified coordinate"""
        self.board[coor] = entity(coor, id, direction)
        # Invalidate empty positions cache
        self._empty_positions = None

    def __str__(self):
        """String representation of the board"""
        # Use StringIO for efficient string concatenation
        s = StringIO()
        for y in range(CELL_NUMBER):
            for x in range(CELL_NUMBER):
                s.write(str(self.board[x + y * CELL_NUMBER]) + " ")
            s.write("\n")
        return s.getvalue()

    def display_index(self):
        """Display the board with entity IDs"""
        # Use StringIO for efficient string concatenation
        s = StringIO()
        for y in range(CELL_NUMBER):
            for x in range(CELL_NUMBER):
                s.write(f"{self.board[x + y * CELL_NUMBER].id} ")
            s.write("\n")
        return s.getvalue()

    def total_reward(self):
        """Calculate the total reward for the current board state"""
        rewards = []
        total_cell_penalty = 0
        
        for start_cell in self.board:
            if isinstance(start_cell, Supplier) and start_cell.speed is not None:
                path = start_cell.coor + start_cell.speed
                r = self.reward(path, start_cell, length=self.length_penality)
                if r != -1:
                    rewards.append(r)
                    
            # Count non-empty cells for penalty
            if not start_cell.get_empty():
                total_cell_penalty += 0.25
                
        return sum(rewards) - total_cell_penalty

    def reward(self, path, previous, length, reward=0.0):
        """Calculate reward for a path starting from a supplier"""
        # Iterative implementation to avoid stack overflow for long paths
        max_path_length = CELL_NUMBER * CELL_NUMBER
        
        # Use a loop instead of recursion
        while True:
            # Check boundary conditions
            if (path >= len(self.board) or path < 0 or
                length / self.length_penality > max_path_length):
                return -1
                
            new_cell = self.board[path]
            
            # Check cell type and calculate reward
            if isinstance(new_cell, Supplier):
                return -1
            elif isinstance(new_cell, Case):
                return reward
            elif isinstance(new_cell, Vendor):
                return 50 / length + reward
            elif isinstance(new_cell, Belt):
                # Handle belt connections
                if isinstance(previous, Supplier):
                    reward += 0.5
                if isinstance(previous, Belt) and previous.direction is not None:
                    if Board.INVERSE_DIRECTION[previous.direction] == new_cell.direction:
                        return -1
                    reward += 0.25
                    
                # Continue to next cell in path
                path += new_cell.speed
                previous = new_cell
                length += self.length_penality
                # Loop continues
            else:
                raise Exception(f"Error path {new_cell}")

    def make_move(self, position: int, item) -> "Board":
        """Create a new board with the specified move applied"""
        # Shallow copy is sufficient for most attributes
        board = copy.copy(self)
        # Deep copy only the board list
        board.board = copy.deepcopy(self.board)
        board.last_move = (position, item)
        board.add(item["entity"], position, direction=item["direction"], id=item["id"])
        # Reset the empty positions cache
        board._empty_positions = None
        return board

    def get_empty_pos(self) -> list[int]:
        """Get a list of empty positions on the board"""
        # Use cached empty positions if available
        if self._empty_positions is None:
            self._empty_positions = [cell.coor for cell in self.board if cell.get_empty()]
        return self._empty_positions

    def actions(self):
        """Generate all possible actions (moves) from the current board state"""
        actions = []
        empty_pos = self.get_empty_pos()
        
        for pos in empty_pos:
            for item in self.all_items:
                actions.append(self.make_move(pos, item))
                
        return actions

    def generate_random_board(self, thread_id, num_boards, file_lock):
        """Generate random boards in a separate thread"""
        # Buffer for storing results before writing to file
        buffer = StringIO()
        buffer_count = 0
        buffer_limit = 10  # Write to file every 10 boards
        
        for i in range(num_boards):
            board = Board(all_items=self.all_items)
            while not board.terminate():
                possible_actions = board.actions()
                if possible_actions:  # Make sure we have valid actions
                    board = choice(possible_actions)
                else:
                    break  # No valid actions available

            # Add to buffer instead of writing directly
            reward = board.total_reward()
            buffer.write(f"reward={reward}\n{board.display_index()}")
            buffer_count += 1
            
            # Write buffer to file when limit reached
            if buffer_count >= buffer_limit:
                with file_lock:
                    with open("game.txt", "a") as f:
                        f.write(buffer.getvalue())
                # Reset buffer
                buffer = StringIO()
                buffer_count = 0

            # Print progress less frequently
            if i % 50 == 0:
                print(f"Thread {thread_id}: Generated {i + 1}/{num_boards} boards")
        
        # Write any remaining boards in buffer
        if buffer_count > 0:
            with file_lock:
                with open("game.txt", "a") as f:
                    f.write(buffer.getvalue())

    def game(self, num_threads=4, total_boards=1000):
        """Generate random boards using multiple threads"""
        if os.path.exists("game.txt"):
            os.remove("game.txt")

        # Create a lock for thread-safe file operations
        file_lock = threading.Lock()

        # Calculate optimal thread count based on CPU cores if not specified
        if num_threads <= 0:
            import multiprocessing
            num_threads = multiprocessing.cpu_count()

        # Calculate boards per thread
        boards_per_thread = total_boards // num_threads
        remaining_boards = total_boards % num_threads

        # Create and start threads
        threads = []
        for i in range(num_threads):
            # Distribute remaining boards among the first few threads
            thread_boards = boards_per_thread + (1 if i < remaining_boards else 0)
            thread = threading.Thread(
                target=self.generate_random_board, args=(i, thread_boards, file_lock)
            )
            threads.append(thread)
            thread.start()
            print(f"Started thread {i} to generate {thread_boards} boards")

        # Wait for all threads to complete
        for i, thread in enumerate(threads):
            thread.join()
            print(f"Thread {i} completed")

        print(f"All {total_boards} boards generated successfully")


if __name__ == "__main__":
    import time
    
    # Measure execution time
    start_time = time.time()
    
    b = Board()
    # Use optimal number of threads based on CPU cores
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    print(f"Starting board generation using {num_cores} threads...")
    b.game(num_threads=num_cores, total_boards=10000)
    
    # Report execution time
    elapsed_time = time.time() - start_time
    print(f"Execution completed in {elapsed_time:.2f} seconds")
