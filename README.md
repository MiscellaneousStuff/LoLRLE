# League of Legends Reinforcement Learning Environment (LoLRLE)

## About

This repo contains code to train an agent to play league of legends in a
distributed manner using the [PPO](https://openai.com/blog/openai-baselines-ppo/)
algorithm. The task is a simple one, kill the enemy player as many times
as you can within one minute. However, the complexity of this task
alone is far more complicated than Chess, Shogi, Go and many other games
due to the state space complexity. Therefore, this introduces an interesting
problem for a machine learning agent to solve as there are many actions
for each timestep as well.

The goal of this project is to produce a bot which is competitive against
human players, one major caveat of this project is that the environment
doesn't contain minions (one of the most important parts of the game)
and that other parts of the game are restricted.

## 