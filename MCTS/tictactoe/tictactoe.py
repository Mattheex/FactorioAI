from copy import deepcopy

from regex import T
from mcts import MCTS


class TicTacToe:
    def __init__(self):
        
        self.player1 = 'x'
        self.player2 = 'o'
        self.empty_square = ' '

        self.board = [self.empty_square  for _ in range(9)]  # A list to hold the board state
        self.winner = None

    def x_perspective(self) -> int:
        # Return the perspective of player 1 (X)
        if self.player1 == 'x':
            return 1
        elif self.player1 == 'o':
            return -1
        else:
            raise ValueError("Invalid player perspective")

    def win(self) -> bool:
        # Check all possible winning combinations
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Horizontal
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Vertical
            (0, 4, 8), (2, 4, 6)              # Diagonal
        ]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != self.empty_square:
                self.winner = self.board[combo[0]]
                return True
        return False
    
    def draw(self) -> bool:
        # Check if the board is full and there is no winner
        if all(square != self.empty_square for square in self.board) and not self.win():
            return True
        return False
    
    def make_move(self, position: int) -> "TicTacToe":

        board = deepcopy(self)
        board.board[position] = board.player1

        (board.player1, board.player2) = (board.player2, board.player1)
    
        return board
    
    def actions(self) -> list["TicTacToe"]:
        # Return a list of valid moves

        actions = []

        for position in range(9):
            if self.board[position] == self.empty_square:
                actions.append(self.make_move(position))

        return actions
    
    
    def game(self):
        # Main game loop

        mcts = MCTS()

        
        while True:
            self.print_board()
            if self.win():
                print(f"Player {self.winner} wins!")
                break
            elif self.draw():
                print("It's a draw!")
                break

            # Player 1's turn
            position = int(input("Player 1 (X), enter your move (0-8): "))
            if self.board[position] != self.empty_square:
                print("Invalid move. Try again.")
                continue

            self = self.make_move(position)
                

            self.print_board()
            if self.win():
                print(f"Player {self.winner} wins!")
                break
            elif self.draw():
                print("It's a draw!")
                break


            best_move = mcts.search(self)
            self = best_move.board

            #print(self.player1)
            #print(self.player1, self.player2)


    def print_board(self):
        # Print the board in a 3x3 grid format
        print(f"{self.board[0]} | {self.board[1]} | {self.board[2]}")
        print("---------")
        print(f"{self.board[3]} | {self.board[4]} | {self.board[5]}")
        print("---------")
        print(f"{self.board[6]} | {self.board[7]} | {self.board[8]}")


TicTacToe().game()