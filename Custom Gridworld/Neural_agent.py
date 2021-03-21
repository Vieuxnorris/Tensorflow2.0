#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 23:20:21 2021

@author: julien
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from collections import namedtuple
from envs.gridworld import GridworldEnv

class Brain(keras.Model):
    def __init__(self, action_dim=5, input_shape=(1, 8 * 8)):
        """Initialize the Agent's Brain model
        Args:
            action_dim (int): Number of actions
        """
        super(Brain, self).__init__()
        self.dense1 = layers.Dense(32, input_shape=input_shape, activation="relu")
        self.logits = layers.Dense(action_dim)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        logits = self.logits(self.dense1(x))
        return logits

    def process(self, observations):
        # Process batch observations using `call(inputs)` behind-the-scenes
        action_logits = self.predict_on_batch(observations)
        return action_logits


class Agent(object):
    def __init__(self, action_dim=5, input_shape=(1, 8 * 8)):
        """Agent with a neural-network brain powered policy
        Args:
            brain (keras.Model): Neural Network based model
        """
        self.brain = Brain(action_dim, input_shape)
        self.brain.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.policy = self.policy_mlp

    def policy_mlp(self, observations):
        observations = observations.reshape(1, -1)
        action_logits = self.brain.process(observations)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        return action  # tf.squeeze(action, axis=0)

    def get_action(self, observations):
        return self.policy(observations)

    def learn(self, obs, actions, **kwargs):
        self.brain.fit(obs, actions, **kwargs)

Trajectory = namedtuple("Trajectory", ["obs", "actions", "reward"])

def rollout(agent, env, render=False):
    obs, episode_reward, done, step_num = env.reset(), 0.0, False, 0
    
    observations, actions = [], []
    episode_reward = 0.0
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        # Save experience
        observations.append(np.array(obs).reshape(1, -1))
        actions.append(action)
        episode_reward += reward
        obs = next_obs
        step_num += 1
        if render:
            env.render()
    env.close()
    return observations, actions, episode_reward

def gather_elite_xp(trajectories, elitism_criterion):
    batch_obs, batch_actions, batch_reward = zip(*trajectories)
    reward_threshold = np.percentile(batch_reward, elitism_criterion)
    indices = [index for index, value in enumerate(batch_reward) if value >= reward_threshold]
    elite_batch_obs = [batch_obs[i] for i in indices]
    elite_batch_actions = [batch_actions[i] for i in indices]
    unpacked_elite_batch_obs = [item for items in elite_batch_obs for item in items]
    unpacked_elite_batch_actions = [item for items in elite_batch_actions for item in items]
    return np.array(unpacked_elite_batch_obs), np.array(unpacked_elite_batch_actions), reward_threshold

def gen_action_distribution(action_index, action_dim=5):
    action_distribution = np.zeros(action_dim).astype(type(action_index))
    action_distribution[action_index] = 1
    action_distribution = np.expand_dims(action_distribution, 0)
    return action_distribution

if __name__ == "__main__":
    total_trajectory_rollouts = 70
    elitism_criterion = 70
    num_epochs = 100
    mean_rewards = []
    elite_reward_thresholds = []
    
    env = GridworldEnv()
    agent = Agent(env.action_space.n, env.observation_space.shape)
    for i in tqdm(range(num_epochs)):
        trajectories = [Trajectory(*rollout(agent, env)) for _ in range(total_trajectory_rollouts)]
        
        _, _, batch_reward = zip(*trajectories)
        
        elite_obs, elite_actions, elite_threshold = gather_elite_xp(trajectories, elitism_criterion=elitism_criterion)
        
        elite_action_distributions = np.array([gen_action_distribution(a.item()) for a in elite_actions])
        
        elite_obs, elite_action_distributions = elite_obs.astype("float16"), elite_action_distributions.astype("float16")
        
        agent.learn(elite_obs, elite_action_distributions, batch_size=128, epochs=3, verbose=0)
        
        mean_rewards.append(np.mean(batch_reward))
        
        elite_reward_thresholds.append(elite_threshold)
        
        print(f"Episode:{i + 1} elite-reward-threshold:{elite_reward_thresholds[-1]:.2f} reward:{mean_rewards[-1]:.2f}")
    
    plt.plot(mean_rewards, 'r', label='mean_reward')
    plt.plot(elite_reward_thresholds, 'g', label="elites_reward_threshold")
    plt.legend()
    plt.grid()
    plt.show()