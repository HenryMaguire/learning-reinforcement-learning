# Simplified Blackjack

![Simplified Blackjack](../static/blackjack.png)

## State space

1. Total value of the player's hand ($H\in [12, 21]$)
    - We ill auto-twist if we get lower than 12
2. The single card we can see in the dealer's hand (ace-10)
3. Whether the player has an ace that can be counted as 11 without busting. (yes/no)

## Action space

1. Stick - Stop receiving cards and terminate
2. Twist - Take another card

## Rewards

Stick:

- +1 if $H > H_{dealer}$
- 0 if $H = H_{dealer}$
- -1 if $H < H_{dealer}$

Twist:

$R=-1$ if $H > 21$ else 0

## Intuition

In this game, trying to compute precise transition probabilities is complicated due to the dynamic state adjustments (turning the useable ace into 1) and the stochastic nature of both the player's and dealer's behaviour.

This means that we should use Model-Free approaches instead, like Monte Carlo, Temporal Difference and TD(Lambda).

## Monte Carlo

Monte Carlo methods are a class of Model-Free algorithms that rely on repeated random sampling to obtain numerical results. They are particularly useful for problems where the state space is large or the transition probabilities are complex to compute. Problems *must* be episodic for MC to work.

1. Episodes
    - Each episode starts with a random initial state and ends when the player sticks or goes bust.
    - Rewards are observed at the end of the episode

2. Policy Evaluation
    - For each state-action pair $(s,a)$ track the average return from all episodes where the agent was in state $s$ and took action $a$
    - Update the Action-Value function $Q(s,a)$ based on the sampled returns
3. Policy Improvement
    - For each state, choose the action $a$ that maximises $Q(s,a)$
