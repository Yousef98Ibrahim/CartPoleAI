# coding: utf-8
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gym
env = gym.make('CartPole-v0')

brain = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(2, activation='softmax')
])

brain.load_weights('./best.h5')


def test(env, brain):
    done = False
    score = 0
    state = env.reset()
    while not done:
        # env.render()
        state = np.expand_dims(state, axis=0)
        action = brain.predict(state).argmax()
        state, reward, done, _ = env.step(action)
        score += reward
    return score


scores = []
for i in range(100):
    score = test(env, brain)
    scores.append(score)
    print(f'itiration number: {i}, score = {score}', end='\r', flush=True)

print('')
env.close()
print(np.mean(scores))
print(scores)
