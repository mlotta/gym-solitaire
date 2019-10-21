# coding: utf-8
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import time

ENV_NAME = 'gym_solitaire:solitaire-v0'

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9999


#TRAINING_TYPE = "auto_learning"
#TRAINING_TYPE = "player"
TRAINING_TYPE = "guided"

solution = [(4, 2), (11, 3), (2, 2), (17, 0), (0, 1), (2, 2), (29, 0), (26, 3), (23, 1), 
            (12, 2), (26, 3), (21, 1), (30, 0), (15, 2), (32, 3), (30, 0), (3, 2), (6, 1), 
            (9, 3), (20, 0), (6, 1), (23, 3), (21, 0), (7, 1), (9, 1), (11, 2), (25, 3), 
            (16, 1), (28, 0), (15, 1), (18, 3)]

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(1, input_shape=(observation_space, ), activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        #self.model.add(Dense(sum(sel + f.action_space), activation="linear"))
        self.model.add(Dense(self.action_space[0]*self.action_space[1], activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            #return tuple([random.randrange(a) for a in self.action_space])
            return divmod(random.randrange(self.action_space[0] * self.action_space[1]), self.action_space[1])
        q_values = self.model.predict(np.reshape(state, [1, self.observation_space]))
        #return np.argmax(q_values[0][:self.action_space[0]]), np.argmax(q_values[0][self.action_space[0]:])
        return divmod(np.argmax(q_values[0]), self.action_space[1])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            #q_update = (reward, reward)
            q_update = reward
            if not terminal:
                prediction = self.model.predict(state_next)[0]
                #q_update = ((reward + GAMMA * np.amax(prediction[:self.action_space[0]])), (reward + GAMMA * np.amax(prediction[self.action_space[0]:])))
                q_update = reward + GAMMA * np.amax(prediction)
            q_values = self.model.predict(np.reshape(state, [1, self.observation_space]))
            index = self.action_space[1]*action[0] + action[1]
            q_values[0][index] = LEARNING_RATE * q_values[0][index] + (1-LEARNING_RATE)*q_update
            #q_values[0][action[0]] = q_update[0]
            #q_values[0][action[1]] = q_update[1]
            self.model.fit(np.reshape(state, [1, self.observation_space]), q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



def solitaire():
    env = gym.make(ENV_NAME)

    observation_space = len(env.observation_space)
    action_space =  (env.action_space[0].n, env.action_space[1].n) 
    dqn_solver = DQNSolver(observation_space, action_space)

    run = 0
    while True:
        run += 1    
        state = env.reset()
        done = False
        step = 0 
        sol_iter = iter(solution)
        cumulated_reward = 0

        while not done:
            step += 1
            env.render()
            if TRAINING_TYPE == "guided":
                if run < 10:
                    action = next(sol_iter)

                else:
                    action = dqn_solver.act(state)

            elif TRAINING_TYPE == "player":
                action = (int(input("Entrez case :")), int(input("up right bottom left : ")))
            elif TRAINING_TYPE == "auto_learning":
                action = dqn_solver.act(state)


            state_next, reward, done, info = env.step(action)
            cumulated_reward += reward
            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next.copy()
            dqn_solver.experience_replay()
            if done:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ",iter: " + str(step))
                break
    env.close()

if __name__ == '__main__':
    solitaire()