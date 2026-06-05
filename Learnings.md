## State and Representation
### State
The agent can only see in the 4 directions of the head, and move in either those directions.
There should be both a graphical map.

### CLI map
Represented as:
- H : snake head
- S: snake body 
- W : wall
- 0 : empty space
- R: red apple
- G: green apple

But not in training to speed up the process.

## Reinforcement Learning
An **agent** as an ability to take actions. These actions are influenced in training by rewards or penalties:
- snake eats red apple: penalty
- snake hits the wall or length = 0 : big penalty
- snake doesn't eat anything: small penalty
- snake eats green apple: reward

### How it works
#### Q function
**Q-value**: expected cumulated reward : Q(s,a) : what's the reward of this **action** based on my current **state** 
Gamma (hyperparameter): defines how patient we want our agent to be: immediate reward vs long-term reward. or: how much future rewards matter or not.
The gamma factor makes rewards fade the further they are.

#### Q-table
A store with a Q-value for every **state** possible. in snake, for every position, there's a different state. but there are as many values for a given state as they are different features: if we have the apple position, and a 10x10 board, they are 100 different values at a certain position. 
So for a 10x10 board with 100 possible apple positions and 4 direction possibilities, there is :

100 x 100 x 4 = 40 000 values in the qtable

> **S**tate **A**ction **R**eward **S**tate **A**ction 

TLDR: I'm in state S1, I choose A1. I end up in S2, with R2. I update A1 based on R2 : was it a good or bad decision to take A1 in S1 ?

We update the table value for every step of the snake, by computing **reward** + **best next reward** x **gamma** (how much we want to make future rewards important).
When the snake dies, we only update with the current reward, because there won't be any more next best steps.

## Features
- qtable: **12 boolean values** : 4 green apple, 4 red apple, 4 danger
- dqns: 16 float values: normalized distance to : 4 snake body, 4 wall, 4 green apple, 4 red apple

## DQNs
### References
- https://medium.com/@jamesnorthfield2001/a-guide-to-deep-q-networks-dqns-806f6f4805f4
- https://www.geeksforgeeks.org/deep-learning/deep-q-learning/
- https://www.youtube.com/watch?v=x83WmvbRa2I !!

### Core ideas
- Experience replay buffer (s1, reward (r), action (a), s2, done)
- periodic minibatches
- add a portion( learning rate) of the MSE between (s1 action q-value) and (s2 highest q-value)
- online vs target network (target is just an older, stable version of the online, mirrored periodically from the online net) -> NOT A GROUND TRUTH

### Process
1. compute Q(s1, a) (the reward for action a in state s1)
2. compute y = (r)(gamma)(maxQ(s2, a))
3. do MSE(Q(s1, a), y)
4. backpropagate the result 

### Hyperparams
- gamma: how distant in the future the reward may be.
  - 0: immediate
  - 0.99: very distant
  - intuition: 0.9 : 10 steps ahead, 0.99: 100 steps ahead
- epsilon: with probability epsilon, choose a random action
  - 1/epsilon: do what the q-table says
  - high epsilon: exploration
  - low epsilon: more exploitation (risk of getting stuck)
  - start between 1 and 0.5
  - end between 0.05 and 0.01 (very important to not stop training before epsilon sits at minimum for around 20% of time for refining)
  - baseline: 
    - start 1.0, 0.9999 decay, 0.005 lowest = 53000 iterations
    - 0.999 decay: 5300 iterations
    - 0.995 decay: ~1000 iterations 
- learning-rate: 
  - doesn't correlate immediately with training iterations
  - some learning rate may fall into a specific local minima: test all learning rates to find the lowest minima and the better model (0.0005 = 4points better results than 0.001 here)
- replay-capacity (dqn only):
  - how many decisions are remembered.
  - 10000 decisions : if a run is ~150 moves, training will retain information from the last 66 runs.

> always have a grid search script for finding the right combination. it takes time but is way less effort than going through params manually, trying to find the right thing.
