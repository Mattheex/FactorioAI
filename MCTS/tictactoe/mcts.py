from random import choice

from math import sqrt, log

from typing import TYPE_CHECKING

from attr import dataclass
from networkx import nodes_with_selfloops

if TYPE_CHECKING:
    from tictactoe import TicTacToe


class TreeNode:
    def __init__(self, board: "TicTacToe", parent=None):
        self.board = board

        self.terminal = self.board.win() or self.board.draw()

        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.fully_expanded = self.terminal


@dataclass
class MCTS:
    debug = False

    def search(self, initial_state: "TicTacToe"):

        self.root = TreeNode(initial_state)

        iterations = 50000
        # Perform the MCTS iterations
        for i in range(iterations):  # Number of iterations
            # print("Iteration: ", i)
            # Perform the MCTS iteration
            node = self.selection(self.root)

            score = self.roolout(node.board)

            self.backpropagation(node, score)

        for c in self.root.children:
            print(
                f"{c.wins} / {c.visits} : utc {c.wins / c.visits
                + 0 * sqrt(log(node.visits) / c.visits)}"
            )

        return self.best_child(self.root, 0)

    def selection(self, node:"TreeNode"):
        while not node.terminal:
            if node.fully_expanded:
                node = self.best_child(node)
            else:
                return self.expand(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        legal_moves = node.board.actions()
        # print(len(legal_moves))
        for move in legal_moves:
            # print([str(c.board.board) for c in node.children])
            if move.board not in [c.board.board for c in node.children]:

                new_node = TreeNode(move, node)
                
                # Check for moves that would allow the opponent to win in the next turn
                if not move.win() and not move.draw() and node == self.root:
                    for opponent_move in move.actions():
                        if opponent_move.win():
                            # This is an immediate loss if the opponent can win in one move
                            print(f"Detected potential immediate loss for {move.player2}")
                            opponent_move.print_board()
                            # Heavily penalize this move
                            if opponent_move.winner == 'x':  # x would win next (bad for o)
                                new_node.wins += -50000  # penalize this move for o
                            if opponent_move.winner == 'o':  # o would win next (bad for x)
                                new_node.wins += 500  # reward this move for o
                            break  # One losing path is enough to penalize

                node.children.append(new_node)

                if len(legal_moves) == len(node.children):
                    node.fully_expanded = True

                return new_node

        print("No legal moves available")
        return node

    def roolout(self, board: "TicTacToe"):
        # Simulate a random game from the current node
        while not board.win() and not board.draw():
            board = choice(seq=board.actions())
        # board.print_board()
        if board.winner == "x":
            # print("Player 1 wins")
            return -1
        elif board.winner == "o":
            # print("Player 2 wins")
            return 1
        else:
            # print("Draw")
            return 0

    def backpropagation(self, node, score):
        # Backpropagate the score up the tree
        while node is not None:
            node.visits += 1
            node.wins += score
            node = node.parent

    def best_child(self, node: TreeNode, exploration_weight=sqrt(2)) -> TreeNode:
        if self.debug:
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
