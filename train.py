import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import ReacherWrapper
from ddpg_agent import Agent


def ddpg(env, agent, n_episodes=1000, max_t=10000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset()
        score = 0
        for _ in range(max_t):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_deque)
            ),
            end="",
        )
        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if np.mean(scores_deque) >= 30.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")
            break
    return scores


def main():
    env = ReacherWrapper(file_name="./Reacher")
    state_size = env.observation_size
    action_size = env.action_size
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

    scores = ddpg(env, agent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Scores")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":
    main()
