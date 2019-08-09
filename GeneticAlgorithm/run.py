import gym
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

GAME = 'CartPole-v0'
POPULATION_SIZE = 50
SAVE_TOP = 0.15
MUTATION_RATE = 0.1


class player:
    def __init__(self, game):
        self.env = gym.make(game)
        self.env._max_episode_steps = 10000
        self.state = self.env.reset()
        self.fittness = 0
        self.render = False
        self.done = False
        self.brain = Sequential([
            Dense(16, activation='relu', input_shape=(4,)),
            Dense(2, activation='softmax')
        ])

    def __lt__(self, player):
        return False

    def step(self):
        if self.render:
            self.env.render()
        p = self.brain.predict(np.expand_dims(self.state, axis=0), steps=1)
        action = np.argmax(p)
        self.state, reward, self.done, _ = self.env.step(action)
        self.fittness += reward

    def run(self):
        while not self.done:
            self.step()
        return self.fittness

    def save(self, name):
        self.brain.save_weights(name)


def select_parents(generation, fittness):
    n = len(generation)
    fittness = np.array(fittness)
    min_fittness = fittness[-int((0.15 * POPULATION_SIZE))]
    fittness[np.abs(fittness) < min_fittness] = 0
    p = np.divide(fittness[-n:], sum(fittness[-n:]))
    return np.random.choice(generation, 2, p=p)


def crossover_(genome1, genome2):
    w1 = genome1.brain.get_weights()
    w2 = genome2.brain.get_weights()

    w = []
    for pair in zip(w1, w2):
        c = np.random.choice([0, 1])
        w.append(pair[c])

    genome = player(GAME)
    genome.brain.set_weights(w)
    return genome


def crossover(generation, no_of_offsprings, fittness):
    offsprings = []
    for _ in range(no_of_offsprings):
        p0, p1 = select_parents(generation, fittness)
        offspring = crossover_(p0, p1)
        offsprings.append(offspring)
    return offsprings


def mutate_(genome, mutation_rate):
    weights = genome.brain.get_weights()
    new_weights = []
    for w in weights:
        mask = np.random.choice([0, 1], w.shape,
                                p=[1-mutation_rate, mutation_rate]
                                ).astype(np.bool)
        r = np.random.randint(-10, 10, size=w.shape)/100
        w[mask] = r[mask]
        new_weights.append(w)
    genome.brain.set_weights(new_weights)
    return genome


def mutate(generation, mutation_rate):
    new_generation = []
    for g in generation:
        new_generation.append(mutate_(g, mutation_rate))
    return new_generation


players = [player(GAME) for _ in range(POPULATION_SIZE)]
best_score = 0
generation_counter = 0
while best_score < 1000:
    print('Calculating fittness...\n')
    scores = []
    c = 0
    for p in players:
        scores.append(p.run())
        print(f'done {c*100//len(players)}%', end='\r', flush=True)
        c += 1
    print('done 100%')

    print('pass best genomes to next generation...')
    scores, players = zip(*sorted(zip(scores, players), reverse=True))

    best = np.argmax(scores)
    players[best].brain.save_weights('best.h5')

    save_top = int(SAVE_TOP * POPULATION_SIZE)
    new_players = list(players[:save_top])

    print('crossover...')
    new_players += crossover(players[save_top:],
                             POPULATION_SIZE - save_top, scores)
    print('mutation...')
    new_players = mutate(new_players, MUTATION_RATE)
    players = new_players
    average_score = np.mean(scores)
    worst_score = np.min(scores)
    best_score = np.max(scores)
    print('\n\n\nGeneration statistics')
    print('_________________________')
    print(f'generation number: {generation_counter}')
    print(f'generation average = {average_score}')
    print(f'generation best = {best_score}')
    print(f'generation worst = {worst_score}')
    print('*****************\n\n')
    # print(players[0].brain.get_weights())
    generation_counter += 1

print(best)
print(players[best].brain.get_weights())
