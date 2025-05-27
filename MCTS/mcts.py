from random import choice, shuffle

from math import sqrt, log

from typing import TYPE_CHECKING

from attr import dataclass
import threading
from queue import Queue
import time

if TYPE_CHECKING:
    from game import Board


class TreeNode:
    def __init__(self, board: "Board", parent=None):
        self.board = board

        self.terminal = self.board.terminate()

        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.fully_expanded = self.terminal


@dataclass
class MCTS:
    debug = True
    lock = threading.Lock()
    current_iteration: int = 0
    total_iterations : int = 500

    def search(self, initial_state: "Board"):

        self.root = TreeNode(initial_state)

        iterations = 200
        # Perform the MCTS iterations
        for i in range(iterations):  # Number of iterations
            if i % 25 == 0 and self.debug:
                print("Iteration: ", i)
            # Perform the MCTS iteration
            node = self.selection(self.root)

            score = self.roolout(node.board)

            self.backpropagation(node, score)

        for c in self.root.children:
            print(
                f"{c.wins} / {c.visits} : utc {c.wins / c.visits+ 0 * sqrt(log(node.visits) / c.visits)} : {c.board.last_move}"
            )

        return self.best_child(self.root, 0)

    def parallel_search(self, initial_state: "Board", num_threads=27):
        self.root = TreeNode(initial_state)
        work_queue = Queue()

        # Create worker threads
        workers = []
        for _ in range(num_threads):
            worker = threading.Thread(target=self._worker, args=(work_queue,),daemon=True)
            worker.start()
            workers.append(worker)

        # Distribute work
        for i in range(self.total_iterations):  # Total iterations
            work_queue.put(i)

        while self.current_iteration < self.total_iterations:
            time.sleep(0.1)  # Check every 100ms
            with self.lock:
                if self.current_iteration % 25 == 0 and self.debug:
                    print(f"\rIteration: {self.current_iteration}/{self.total_iterations}", end="", flush=True)

        # Wait for completion
        work_queue.join()
        for _ in range(num_threads):
            work_queue.put(None)
        for worker in workers:
            worker.join()

        for c in self.root.children:
            print(
                f"{c.wins} / {c.visits} : utc {c.wins / c.visits+ 0 * sqrt(log(self.root.visits) / c.visits)} : {c.board.last_move}"
            )

        return self.best_child(self.root, 0)

    def _worker(self, queue):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                break

            node = self.selection(self.root)
            score = self.roolout(node.board)

            with self.lock:
                self.backpropagation(node, score)
                self.current_iteration += 1

            queue.task_done()

    def selection(self, node: TreeNode):
        while not node.terminal:
            if node.fully_expanded:
                node = self.best_child(node)
            else:
                return self.expand(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        legal_moves = node.board.actions()
        shuffle(legal_moves)
        # print(f"Legal moves: {len(legal_moves)}")
        for move in legal_moves:
            # print(move)
            # print([str(c.board.board) for c in node.children])
            if str(move) not in [str(c.board) for c in node.children]:

                new_node = TreeNode(move, node)
                node.children.append(new_node)

                if len(legal_moves) == len(node.children):
                    node.fully_expanded = True

                return new_node

        print("No legal moves available")
        return node

    def roolout(self, board: "Board") -> float:
        # Simulate a random game from the current node
        total_reward = 0
        while not board.terminate():
            board = choice(seq=board.actions())
            total_reward += board.total_reward()

        return total_reward

    def backpropagation(self, node, score):
        # Backpropagate the score up the tree
        while node is not None:
            node.visits += 1
            node.wins += score
            node = node.parent

    def best_child(self, node: TreeNode, exploration_weight=sqrt(2)) -> TreeNode:
        return max(
            node.children,
            key=lambda c: c.wins / c.visits
            + exploration_weight * sqrt(log(node.visits) / c.visits),
        )
