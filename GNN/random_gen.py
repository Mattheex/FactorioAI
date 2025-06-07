from copy import copy
from io import StringIO
import os
from random import choice, random
import threading

from mcts import MCTS
from var import CELL_NUMBER
from game import Board


def generate_random_board(thread_id, num_boards, file_lock):
    """Generate random boards in a separate thread"""
    # Buffer for storing results before writing to file
    buffer = StringIO()
    buffer_count = 0
    board = Board()
    
    for i in range(num_boards):
        board = Board(all_items=board.all_items)
        saves = board.game()
            

        # Add to buffer instead of writing directly
        #reward = board.total_reward()
        buffer.write(saves)
        
        # Write buffer to file when limit reached
        with file_lock:
            with open("game.txt", "a") as f:
                f.write(buffer.getvalue())
        # Reset buffer
        buffer = StringIO()

        # Print progress less frequently
        if i % 50 == 0:
            print(f"Thread {thread_id}: Generated {i + 1}/{num_boards} boards")
    
    # Write any remaining boards in buffer
    if buffer_count > 0:
        with file_lock:
            with open("game.txt", "a") as f:
                f.write(buffer.getvalue())

def game(num_threads=4, total_boards=1000):
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
            target=generate_random_board, args=(i, thread_boards, file_lock)
        )
        threads.append(thread)
        thread.start()
        print(f"Started thread {i} to generate {thread_boards} boards")

    # Wait for all threads to complete
    for i, thread in enumerate(threads):
        thread.join()
        print(f"Thread {i} completed")

    print(f"All {total_boards} boards generated successfully")

game(total_boards=1000)