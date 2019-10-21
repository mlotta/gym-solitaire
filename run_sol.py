import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

ENV_NAME = 'gym_solitaire:solitaire-v0'

GAMMA = 0.80
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(observation_space, input_shape=(1,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(sum(self.action_space), activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
        	return tuple([random.randrange(a) for a in self.action_space])
        q_values = self.model.predict(state)
        return np.argmax(q_values[0][:self.action_space[0]]), np.argmax(q_values[0][self.action_space[0]:])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = (reward, reward)
            if not terminal:
                prediction = self.model.predict(state_next)[0]
                q_update = ((reward + GAMMA * np.amax(prediction[:self.action_space[0]])), (reward + GAMMA * np.amax(prediction[self.action_space[0]:])))
            q_values = self.model.predict(state)
            q_values[0][action[0]] = q_update[0]
            q_values[0][action[1]] = q_update[1]
            self.model.fit(state, q_values, verbose=0)
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
		while not done:
			step += 1
			env.render()
			#action = (int(input("Entrez case :")), int(input("up right bottom left : ")))
			#action = (random.randrange(0, 33), random.randrange(0, 2))
			action = dqn_solver.act(state)
			
			state_next, reward, done, info = env.step(action)
			dqn_solver.remember(state, action, reward, state_next, done)
			state = state_next
			dqn_solver.experience_replay()
			if done:
				print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate))
				break
	env.close()

if __name__ == '__main__':
	solitaire()
