import logic
import random
import math
import numpy as np
from copy import copy
from AbstractPlayers import *
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}

### global functions and consts
victory_value = 2048


def empty_cells_calc(board) -> int:
    empty_cells = 0
    for i in range(0, len(board)):
        for j in range(0, len(board)):
            if board[i][j] == 0:
                empty_cells += 1
    return empty_cells


def evaluation_function(board, score):
    return math.log2(score + 2) * (empty_cells_calc(board)) \
    + 2 * board[0][0] + 0.5 * board[0][1] + 0.25 * board[0][2]


def get_empty_indices(board) -> [(int,int)]:
    empty = []
    for i in range(0,len(board)):
        for j in range(0, len(board)):
            if board[i][j] == 0:
                empty.append((i, j))
    return empty


def is_final_state(board) -> bool:
    empty_cells = 0
    max_val = 0
    for i in range(0,len(board)):
        for j in range(0, len(board)):
            if board[i][j] == 0:
                empty_cells += 1
            if board[i][j] > max_val:
                max_val = board[i][j]
    return (empty_cells == 0) or (max_val == victory_value)


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.previous_score = 0

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        next_score = self.previous_score
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = (score - self.previous_score) + math.log2(score + 2) * (empty_cells_calc(new_board))\
                                             + 2 * new_board [0][0] + 0.5 * new_board [0][1] + 0.25 * new_board [0][2]
                #optional_moves_score[move] = self.moveGrader(new_board)
                if score > next_score:
                    next_score = score
        self.previous_score = next_score
        return max(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed

    """ Returns the maximal tile. """

    def highestTile(self, board):
        maximal =  max(max(x) for x in board)
        return maximal

    """ Calculates the inverse sum of all the log values of the tiles. """

    def score(self, board):
        sum = 0
        for row in range(0, 4):
            for col in range(0, 4):
                if board[row][col] != 0:
                    sum -= math.log2(board[row][col])
        return sum

    def moveGrader(self, board):
        emptFact = 25
        highestFact = 10
        scoregrade = 14
        highest = self.highestTile(board)
        empty = empty_cells_calc(board)
        val = emptFact * empty + highestFact * highest + self.score(board) * scoregrade
        return val


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        optional_moves_score = {}
        depth = 1
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                move_value = self.rec_minmax(depth - 0.5, new_board, evaluation_function(new_board, score))
                if move_value >= -np.inf:
                    optional_moves_score[move] = move_value
        return max(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed
    def rec_minmax(self, depth, board, eval_value=0):
        '''
        a recursive algorithm to calculate minimax algorithm
        :param depth: tracking our depth in the tree
        :param game_state: A stateof the game
        :param player: 0 - player, 1 - opponent
        :return: value of the state according to minimax
        '''
        if depth == 0 or is_final_state(board):
            return eval_value
        # maximize
        if round(depth) == depth:  # depth = round number
            optional_moves_score = {}
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = new_board, evaluation_function(new_board,score)
            return self.return_max([self.rec_minmax(depth - 0.5, new_board, new_score) \
                                    for new_board, new_score in optional_moves_score])
        # minimize
        else:
            index_next_boards = []
            for tup in get_empty_indices(board):
                new_board = copy(board)
                i = int(tup[0])
                j = int(tup[1])
                new_board[i][j] = 2
                index_next_boards.append(new_board)
            return self.return_min([self.rec_minmax(depth - 0.5, new_board, eval_value) \
                                    for new_board in index_next_boards])

    def return_max(self, successors_values):
        '''
        :param successors_values: list of tuple: (successor, value)
        :return: tuple with maximal value
        '''
        if len(successors_values) == 0:
            return -np.inf
        return max(successors_values)

    def return_min(self, successors_values):
        '''
        :param successors_values: list of tuple: (successor, value)
        :return: tuple with maximal value
        '''
        if len(successors_values) == 0:
            return np.inf
        return min(successors_values)

    def get_action(self, board):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1
        Action.STOP:
            The stop direction, which is always legal
        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        empty_cells = get_empty_indices(board)
        optional_moves_score = {}
        for move in empty_cells:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score
        return min(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed
    def rec_minmax(self, depth, board, eval_value=0):
        '''
        a recursive algorithm to calculate minimax algorithm
        :param depth: tracking our depth in the tree
        :param game_state: A stateof the game
        :param player: 0 - player, 1 - opponent
        :return: value of the state according to minimax
        '''
        if depth == 0 or is_final_state(board):
            return eval_value
        # maximize
        if round(depth) != depth:  # depth = round number
            optional_moves_score = {}
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = new_board, evaluation_function(new_board,score)
            return self.return_max([self.rec_minmax(depth - 0.5, new_board, new_score) \
                                    for new_board, new_score in optional_moves_score])
        # minimize
        else:
            index_next_boards = {}
            for i, j in get_empty_indices(board):
                new_board = copy(board)
                new_board[i][j] = 2
                index_next_boards[i][j] = new_board
            return self.return_min([self.rec_minmax(depth - 0.5, new_board, eval_value) \
                                    for new_board in index_next_boards])

# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed

