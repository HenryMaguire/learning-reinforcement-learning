# Chess

There are roughly 10^43 different valid board states of chess. We can no longer explicitly keep track of the value functino for every single state in a big table. We instead need to estimate the value of a different states, or action-states, with some kind of function approximation. During training, we can't hope to see all 10^43 board states, so we need to train a model that generalises to potentially unseen ones.

## RL Approaches

### Level 1: "Vanilla" Policy Gradient method

### Level 1.5: Trust Region Policy Optimisation (TRPO)
