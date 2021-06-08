import logic
import random
import math
import numpy as np
from copy import deepcopy
from AbstractPlayers import *
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}

# global functions and consts

victory_value = 2048
time_margin = 0.05


def empty_cells_calc(board) -> int:
    empty_cells = 0
    for i in range(0, len(board)):
        for j in range(0, len(board)):
            if board[i][j] == 0:
                empty_cells += 1
    return empty_cells


def board_score_calc(board) -> int:
    score = 0
    for i in range(0, len(board)):
        for j in range(0, len(board)):
            current = board[i][j]
            while current > 2:
                score += current
                current = current / 2
    return score


def evaluation_function(board, score):
    return math.log2(score + 2) * (empty_cells_calc(board)) \
    + 2 * board[0][0] + 0.5 * board[0][1] + 0.25 * board[0][2]


def evaluation_function2(board):
    return math.log2(board_score_calc(board) + 2) * (empty_cells_calc(board)) \
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


def return_max(successors_values):
    """
    :param successors_values: list of tuple: (successor, value)
    :return: tuple with maximal value
    """
    if len(successors_values) == 0:
        return -np.inf
    return max(successors_values)


def return_min(successors_values):
    """
    :param successors_values: list of tuple: (successor, value)
    :return: tuple with maximal value
    """
    if len(successors_values) == 0:
        return np.inf
    return min(successors_values)


# functions for the second heuristic function

def gradient(board):
    """
    Calculates how much the grid is uniforaml - if all the cells
    are equal, the function will return 0. Otherwise, it will return
    the inverse of sum of all differences (in absolute value) between neighbours in the grid.
    """
    simpleGrid = [[0 for i in range(4)] for j in range(4)]
    for row in range(0, 4):
        for col in range(0, 4):
            simpleGrid[row][col] = board[row][col]

    bonus = 0
    for row in range(4):
        for col in range(4):
            if simpleGrid[row][col] != 0:
                if simpleGrid[row][3] != 0:
                    bonus += abs(math.log2(simpleGrid[row][col]) - math.log2(
                        simpleGrid[row][3]))
                if simpleGrid[3][col] != 0:
                    bonus += abs(math.log2(simpleGrid[row][col]) - math.log2(
                        simpleGrid[3][col]))

    return -bonus


def direction(board):
    """ Calculates how much the grid is organized in some direction. """
    simpleGrid = [[0 for i in range(4)] for j in range(4)]
    for row in range(0,4):
        for col in range(0,4):
            if(board[row][col] != 0):
                simpleGrid[row][col] = math.log2(board[row][col])
            else:
                simpleGrid[row][col] = 0

    asndud = 0
    dsndud = 0
    asndlr = 0
    dsndlr = 0
    for row in range(4):
        for col in range(4):
            if col + 1 < 4:
                if simpleGrid[row][col] > simpleGrid[row][col + 1]:
                    dsndlr -= simpleGrid[row][col] - simpleGrid[row][col + 1]
                else:
                    asndlr += simpleGrid[row][col] - simpleGrid[row][col + 1]
            if row + 1 < 4:
                if simpleGrid[row][col] > simpleGrid[row + 1][col]:
                    dsndud -= simpleGrid[row][col] - simpleGrid[row + 1][col]
                else:
                    asndud += simpleGrid[row][col] - simpleGrid[row + 1][col]
    return max(dsndlr, asndlr) + max(dsndud, asndud)


def highestTile(board):
    """ Returns the maximal tile. """
    maximal =  max(max(x) for x in board)
    return maximal


def h_score(board):
    """ Calculates the inverse sum of all the log values of the tiles. """
    sum = 0
    for row in range(0, 4):
        for col in range(0, 4):
            if board[row][col] != 0:
                sum -= math.log2(board[row][col])
    return sum


def moveGrader(board):
    emptFact = 25
    highestFact = 10
    scoregrade = 14
    gradientFac = 1
    directionFac = 15
    highest = highestTile(board)
    empty = empty_cells_calc(board)
    if empty != 0:
        empty = math.log(empty)
    val = emptFact * empty + highestFact * highest + h_score(board) * scoregrade + direction(board) * directionFac\
          + gradient(board) * gradientFac
    return val


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
                if score > next_score:
                    next_score = score
        self.previous_score = next_score
        return max(optional_moves_score, key=optional_moves_score.get)

    # TODO: add here helper functions in class, if needed


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.iteration_start_time = None
        self.iteration_time = 1
        self.timeout_flag = False

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        temp_max = Move.UP
        depth = 0
        self.iteration_start_time = time.time()
        self.iteration_time = time_limit
        self.timeout_flag = False
        while time.time() - self.iteration_start_time < time_limit - time_margin:
            optional_moves_score = {}
            depth += 1
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    #move_value = self.rec_minmax(depth - 0.5, new_board, evaluation_function2(new_board))
                    if time.time() - self.iteration_start_time < time_limit - time_margin:
                        local_start = time.time()
                        move_value = self.rec_minmax(depth - 0.5, new_board, moveGrader(new_board))
                        #print("Move time: " + str(time.time()-local_start) + " depth: " + str(depth))
                        if move_value >= -np.inf:
                            optional_moves_score[move] = move_value
                    else:
                        return temp_max
            if self.timeout_flag is False:
                temp_max = max(optional_moves_score, key=optional_moves_score.get)
        return temp_max


    # TODO: add here helper functions in class, if needed
    def rec_minmax(self, depth, board, eval_value=0):
        """
        a recursive algorithm to calculate minimax algorithm
        :param depth: tracking our depth in the tree
        :param game_state: A stateof the game
        :param player: 0 - player, 1 - opponent
        :return: value of the state according to minimax
        """
        # if no time left
        if time.time() - self.iteration_start_time >= self.iteration_time - time_margin:
            self.timeout_flag = True
            return eval_value

        if depth == 0:  #or is_final_state(board):
            return eval_value
        # maximize
        if round(depth) == depth:  # depth = round number
            optional_moves_score = {}
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    #optional_moves_score[move] = new_board, evaluation_function2(new_board)
                    optional_moves_score[move] = new_board, moveGrader(new_board)
            return return_max([self.rec_minmax(depth - 0.5, optional_moves_score[move1][0], optional_moves_score[move1][1])\
                               for move1 in optional_moves_score])
        # minimize
        else:
            index_next_boards = []
            for tup in get_empty_indices(board):
                new_board = deepcopy(board)
                i = int(tup[0])
                j = int(tup[1])
                new_board[i][j] = 2
                index_next_boards.append(new_board)
            return return_min([self.rec_minmax(depth - 0.5, new_board, eval_value) \
                                    for new_board in index_next_boards])

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
        self.iteration_start_time = None
        self.iteration_time = 1
        self.timeout_flag = False

    def get_indices(self, board, value, time_limit) -> (int, int):
        depth = 0
        empty = get_empty_indices(board)
        temp_min = empty[0]
        self.iteration_start_time = time.time()
        self.iteration_time = time_limit
        self.timeout_flag = False
        while time.time() - self.iteration_start_time < time_limit - time_margin:
            optional_index_score = {}
            depth += 1
            # score = board_score_calc(board)
            for tup in empty:
                new_board = deepcopy(board)
                i = int(tup[0])
                j = int(tup[1])
                new_board[i][j] = 2
                #move_value = self.rec_minmax(depth - 0.5, new_board, evaluation_function(new_board, score))
                if time.time() - self.iteration_start_time < time_limit - time_margin:
                    local_start = time.time()
                    move_value = self.rec_minmax(depth - 0.5, new_board, moveGrader(new_board))
                    #print("Index time: " + str(time.time() - local_start) + " depth: " + str(depth))
                    if move_value <= np.inf:
                        optional_index_score[(i,j)] = move_value
                else:
                    return temp_min
            if self.timeout_flag is False:
                temp_min = min(optional_index_score, key= optional_index_score.get)
        return temp_min

    # TODO: add here helper functions in class, if needed
    def rec_minmax(self, depth, board, eval_value=0):
        """
        a recursive algorithm to calculate minimax algorithm
        :param depth: tracking our depth in the tree
        :param game_state: A stateof the game
        :param player: 0 - player, 1 - opponent
        :return: value of the state according to minimax
        """

        # if no time left
        if time.time() - self.iteration_start_time >= self.iteration_time - time_margin:
            self.timeout_flag = True
            return eval_value

        if depth == 0: # or is_final_state(board):
            return eval_value

        # minimize
        if round(depth) == depth:  # depth = round number
            index_next_boards = []
            for tup in get_empty_indices(board):
                new_board = deepcopy(board)
                i = int(tup[0])
                j = int(tup[1])
                new_board[i][j] = 2
                index_next_boards.append(new_board)
            return return_min([self.rec_minmax(depth - 0.5, new_board, eval_value)\
                               for new_board in index_next_boards])
        # maximize
        else:
            optional_moves_score = {}
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    #optional_moves_score[move] = new_board, evaluation_function(new_board, score)
                    optional_moves_score[move] = new_board, moveGrader(new_board)
            return return_max([self.rec_minmax(depth - 0.5, optional_moves_score[move1][0], optional_moves_score[move1][1])\
                               for move1 in optional_moves_score])


# part C
class ABMovePlayer(AbstractMovePlayer):
    """
    Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed
        self.iteration_start_time = None
        self.iteration_time = 1
        self.timeout_flag = False

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        file = open(r'C:\Users\kfir2\Documents\ab_player_1_sec.txt', 'a')
        temp_max = Move.UP
        depth = 0
        alpha = -np.inf
        beta = np.inf
        self.iteration_start_time = time.time()
        self.iteration_time = time_limit
        self.timeout_flag = False
        while time.time() - self.iteration_start_time < time_limit - time_margin:
            optional_moves_score = {}
            depth += 1
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    if time.time() - self.iteration_start_time < time_limit - time_margin:
                        move_value = self.rec_ab(depth - 0.5, new_board, alpha, beta, evaluation_function2(new_board))
                        #move_value = self.rec_ab(depth - 0.5, new_board, alpha, beta, moveGrader(new_board))
                        if move_value >= -np.inf:
                            optional_moves_score[move] = move_value
                    else:
                        print(depth, file=file)
                        file.close()
                        return temp_max
            if self.timeout_flag is False:
                temp_max = max(optional_moves_score, key=optional_moves_score.get)
        print(depth, file=file)
        file.close()
        return temp_max

    # TODO: add here helper functions in class, if needed
    def rec_ab(self, depth, board, alpha, beta, eval_value=0):
        """
        a recursive algorithm to calculate minimax algorithm
        :param depth: tracking our depth in the tree
        :param game_state: A stateof the game
        :param player: 0 - player, 1 - opponent
        :return: value of the state according to minimax
        """
        # if no time left
        if time.time() - self.iteration_start_time >= self.iteration_time - time_margin:
            self.timeout_flag = True
            return eval_value

        if depth == 0:  #or is_final_state(board):
            return eval_value

        # maximize
        if round(depth) == depth:  # depth = round number
            curMax = -np.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    v = self.rec_ab(depth - 0.5, new_board, alpha, beta, evaluation_function2(new_board))
                    #v = self.rec_ab(depth - 0.5, new_board, alpha, beta, moveGrader(new_board))
                    curMax = max(curMax,v)
                    alpha = max(curMax,alpha)
                    if curMax >= beta:
                        return np.inf
            return curMax

        # minimize
        else:
            curMin = np.inf
            for tup in get_empty_indices(board):
                new_board = deepcopy(board)
                i = int(tup[0])
                j = int(tup[1])
                new_board[i][j] = 2
                v = self.rec_ab(depth - 0.5, new_board, alpha, beta, moveGrader(new_board))
                curMin = min(curMin, v)
                beta = min(curMin, beta)
                if curMin <= alpha:
                    return -np.inf
            return curMin

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

