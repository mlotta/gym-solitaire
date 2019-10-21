import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

_ind_to_rowcol = {'0': (0, 2), '1':(0, 3), '2':(0, 4),
                  '3': (1, 2), '4':(1, 3), '5':(1, 4),
                  '6': (2, 0), '7':(2, 1), '8':(2, 2), '9': (2, 3), '10':(2, 4), '11':(2, 5), '12': (2, 6),
                  '13': (3, 0), '14':(3, 1), '15':(3, 2), '16': (3, 3), '17':(3, 4), '18':(3, 5), '19': (3, 6),
                  '20': (4, 0), '21':(4, 1), '22':(4, 2), '23': (4, 3), '24':(4, 4), '25':(4, 5), '26': (4, 6),
                  '27': (5, 2), '28':(5, 3), '29':(5, 4),
                  '30': (6, 2), '31':(6, 3), '32':(6, 4),
                  }

_rowcol_to_ind = {str(item[0])+str(item[1]):int(key) for key, item in _ind_to_rowcol.items()}

class SolitaireEnv(gym.Env):
  """
        0  1  2
        3  4  5
  6  7  8  9  10 11 12
  13 14 15 16 17 18 19
  20 21 22 23 24 25 26
        27 28 29 
        30 31 32 
  """


  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.holes = [(0.4, 0.8), (0.5, 0.8), (0.6, 0.8),
                  (0.4, 0.7), (0.5, 0.7), (0.6, 0.7),
                  (0.2, 0.6), (0.3, 0.6), (0.4, 0.6), (0.5, 0.6), (0.6, 0.6), (0.7, 0.6), (0.8, 0.6),
                  (0.2, 0.5), (0.3, 0.5), (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.5), (0.8, 0.5),
                  (0.2, 0.4), (0.3, 0.4), (0.4, 0.4), (0.5, 0.4), (0.6, 0.4), (0.7, 0.4), (0.8, 0.4),
                  (0.4, 0.3), (0.5, 0.3), (0.6, 0.3),
                  (0.4, 0.2), (0.5, 0.2), (0.6, 0.2),
                  ]
    self.hole_number = len(self.holes)
    self.peg_number = self.hole_number - 1
    self.center_index = 16

    self.action_space = spaces.Tuple((spaces.Discrete(self.hole_number),  # hole id  
                                      spaces.Discrete(4)))               # 0 = up, 1 = right, 2 = bottom, 3 = left
    self.observation_space = spaces.Tuple(tuple([spaces.Discrete(2) for _ in range(self.hole_number)]))

    self.viewer = None
    self.state = None


  def step(self, action):
    assert self.action_space.contains(action)
    state = self.state
    # check if the action is possible
    first_hole = _ind_to_rowcol[str(action[0])]
    self.round += 1
    reward = 10


    try:
      dest_hole = _rowcol_to_ind[str(first_hole[0] + 2*(action[1]%2 == 0)*(action[1]-1))+str(first_hole[1] + 2*(action[1]%2 == 1)*(2-action[1]))]
      inter_hole = _rowcol_to_ind[str(first_hole[0] + (action[1]%2 == 0)*(action[1]-1))+str(first_hole[1] + (action[1]%2 == 1)*(2-action[1]))]
    except KeyError:
      reward = -1
    else:
      if state[action[0]] == 1 and state[inter_hole] == 1 and state[dest_hole] == 0:
        state[action[0]] = 0
        state[inter_hole] = 0
        state[dest_hole] = 1
      else:
        reward = -1


    observation = state
    done = False
    if sum(state) == 1:
      done = True
      reward = 200 - self.round
    elif self.round > 999:
      done = True
      reward = -sum(state)


    return observation, reward, done, {}

  def reset(self):
    self.round = 0
    self.state = [1 for _ in range(self.hole_number)]
    #self.state = [1 for _ in range(21)]
    self.state[self.center_index] = 0
    return np.array(self.state)

  def render(self, mode='human'):
    screen_width = 600
    screen_height = 600
    board_radius = (min(screen_height, screen_width) - 50) / 2
    hole_radius = 10
    holes = [(int(hole[0] * screen_width), int(hole[1] * screen_height)) for hole in self.holes]


    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)

      board = rendering.make_circle(radius=board_radius, res=50)
      self.boardtrans = rendering.Transform(translation=(screen_width/2, screen_height/2))
      board.add_attr(self.boardtrans)
      board.set_color(0.2, 0.2, 0.2)
      self.viewer.add_geom(board)

      self.holetranss = [rendering.Transform(translation=position) for position in holes]
      self.holes_render = [rendering.make_circle(radius=hole_radius, filled=True) for value in self.state]
      for ind, value in enumerate(self.state):
        self.holes_render[ind].add_attr(self.holetranss[ind])
        self.viewer.add_geom(self.holes_render[ind])

    for ind, value in enumerate(self.state):
      self.holes_render[ind].set_color(value, value, value)


    return self.viewer.render()


  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
