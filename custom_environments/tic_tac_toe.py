import numpy as np
from types import SimpleNamespace 


class TicTacToe():

  def __init__(self):
    self.board = np.zeros(9, dtype=np.int32)
    self._elapsed_steps = 0
    self.turn = 1
    
    self.action_space = SimpleNamespace(**{'n':9})
    self.observation_space = self.board.copy()

    self.prev_board = None

  def step(self, action):
    terminal = False
    reward = 0
    result = None

    self.board[action] = self.turn
    if self.did_win(action):
      terminal = True
      reward = 1
      if self.turn == 1:
        result = "player 1 wins"
      else:
        result = "player 2 wins"
    elif self._elapsed_steps == 8:
      terminal = True
      result = "draw"

    self._elapsed_steps += 1
    self.turn *= -1

    obs = self.turn * self.board.copy()
    self.prev_board = self.board.copy()
    return obs, reward, terminal, {"result": result}

  def legal_actions(self):
    return np.where(self.board==0)[0]

  def seed(self, seed):
    return
  
  def reset(self):
    self.board = np.zeros(9, dtype=np.int32)
    self._elapsed_steps = 0
    self.turn = 1
    return self.board.copy()

  def did_win(self, move):
    win = False
    board = self.board.reshape(3,3)
    i, j = np.unravel_index(move, (3,3))
    if abs(np.sum(board[i,:])) == 3:
      win = True
    elif abs(np.sum(board[:,j])) == 3:
      win = True
    elif (i, j) in [(0,0), (2,2)]:
      if abs(np.sum(board.diagonal())) == 3:
        win = True
    elif (i, j) in [(0,2), (2,0)]:
      if abs(np.sum(np.rot90(board).diagonal())) == 3:
        win = True
    elif (i, j) == (1, 1):
      if abs(np.sum(board.diagonal())) == 3:
        win = True
      elif abs(np.sum(np.rot90(board).diagonal())) == 3:
        win = True
    return win

  def render(self, mode=''):
    print("\n{}\n".format(self.prev_board.reshape(3,3)))

