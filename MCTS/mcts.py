from random import choice

from math import sqrt, log

from typing import TYPE_CHECKING

from attr import dataclass
from networkx import nodes_with_selfloops

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

    def search(self, initial_state: "Board"):

        self.root = TreeNode(initial_state)

        iterations = 100
        # Perform the MCTS iterations
        for i in range(iterations):  # Number of iterations
            if i % 50 == 0 and self.debug:
                print("Iteration: ", i)
            # Perform the MCTS iteration
            node = self.selection(self.root)

            score = self.roolout(node.board)

            self.backpropagation(node, score)

        for c in self.root.children:
            print(
                f"{c.wins} / {c.visits} : utc {c.wins / c.visits
                + 0 * sqrt(log(node.visits) / c.visits)} : {c.board.last_move}"
            )

        return self.best_child(self.root, 0)

    def selection(self, node:TreeNode):
        while not node.terminal:
            if node.fully_expanded:
                node = self.best_child(node)
            else:
                return self.expand(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        legal_moves = node.board.actions()
        #print(f"Legal moves: {len(legal_moves)}")
        for move in legal_moves:
            #print(move)
            # print([str(c.board.board) for c in node.children])
            if str(move) not in [str(c.board) for c in node.children]:

                new_node = TreeNode(move, node)
                node.children.append(new_node)

                if len(legal_moves) == len(node.children):
                    node.fully_expanded = True

                return new_node

        print("No legal moves available")
        return node

    def roolout(self, board: "Board",max_steps=500) -> float:
        # Simulate a random game from the current node
        total_reward = 0
        step = 0
        while not board.terminate() and step < max_steps:
            board = choice(seq=board.actions())
            total_reward += board.total_reward()
            step += 1
        if step >= max_steps:
            print("Max steps reached, terminating rollout")
        
        return total_reward

    def backpropagation(self, node, score):
        # Backpropagate the score up the tree
        while node is not None:
            node.visits += 1
            node.wins += score
            node = node.parent

    def best_child(self, node: TreeNode, exploration_weight=sqrt(2)) -> TreeNode:
        if False and self.debug:
            print("---print children---")
            for c in node.children:
                # c.board.print_board()
                print(
                    f"{c.wins} / {c.visits} : utc {c.wins / c.visits
                + exploration_weight * sqrt(log(node.visits) / c.visits)}"
                )
        return max(
            node.children,
            key=lambda c: c.wins / c.visits
            + exploration_weight * sqrt(log(node.visits) / c.visits),
        )
