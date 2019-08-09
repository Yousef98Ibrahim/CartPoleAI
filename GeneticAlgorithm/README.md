# Genetic Algorithm
---

training a keras model to play 'CartPole-v0' from OpenAI gym using genetic algorithms
---

### Requiremets
* keras
* numpy
* gym
---

### usage
* to train the model

> python run.py

train and saves the best performing weights

* testing

> python test.py

loads the saved model and runs it for 100 episodes
---

###About the training process
* population size of 50
* keep the top 15% of the population
* kill the bottom 15%
* crossover the remaining genomes to complete the next generation
* crossover
    * randomly swap the bais or weights
* mutation
    * 10% chance mutate a weight or bais
    * to mutate weight or bias add value in range (-0.1 - 0.1)