"""
The research sibling method uses 2 PPO agents, a 'brother' and a 'sister',
which shares their reward but alternates their randomness.

This file contains the agent for research sibling. This agent is a variant of the original PPO agent.
"""
import numpy as np
import torch
from torch import nn
from agents.ppo_agent import build_network
from torch.distributions import Categorical

class SiblingAgent(nn.Module):
    """
    A single agent in a brother-sister pair, the randomness regulates how it would
    do the research.
    """

    def __init__(self, envs, nodes_counts, args, device):
        super().__init__()
        self.randomness = 0 #initiate randomness
        self.args = args
        self.device = device
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

        input_dim = np.prod(envs.single_observation_space.shape)
        self.critic_nodes = [input_dim] + nodes_counts + [1]
        self.actor_nodes = [input_dim] + nodes_counts + [envs.single_action_space.n]

        self.critic = nn.Sequential(*build_network(self.critic_nodes, 1.0))
        self.actor = nn.Sequential(*build_network(self.actor_nodes, 0.01))



    def get_value(self, x):
        """
        Computes the value of a given state using the critic network.

        Parameters:
        x (torch.Tensor): The input tensor representing the state.

        Returns:
        torch.Tensor: The value of the given state.
        """
        return self.critic(x)

    def get_action_and_value(self, x, randomness, action = None):
        """
        Computes the action to take and its associated value, log probability, and entropy.

        Parameters:
        x (torch.Tensor): The input tensor representing the state.
        action (torch.Tensor, optional): The action to evaluate. If None, a new action will be sampled.
        randomness: the randomness of the action. equals self.randomness in research sibling method

        Returns:
        tuple: A tuple containing the action, its log probability, the entropy of the action distribution, and the value of the state.
        """
        logits = self.actor(x)
        value = self.critic(x)
        probs = Categorical(logits=logits)
        scaled_logits = logits.clone()
        logits = logits

        if action is None:
            #here randomness plays the role in determining the action taken
            for i in range(0, self.args.num_envs):
                scaled_logits[i] = logits[i] / np.tan(randomness[i]*(np.pi/2))
            scaled_probs = torch.softmax(scaled_logits, dim=0)
            action_dist = Categorical(probs=scaled_probs)
            action = action_dist.sample()

        return action, probs.log_prob(action), probs.entropy(), value

    def get_randomness(self, self_reward, total_reward, concentration, CONCENTRATION_THRESHOLD):
        """
        function for the agent to adjust its randomness.
        :param self_reward: reward of this agent
        :param total_reward: total reward of brother + sister
        :param concentration: concentration of the agent, dependent on its recent self_reward
        :return: self.randomness
        """
        if concentration >= CONCENTRATION_THRESHOLD:
            return 0

        else:
            return -0.99 * ((self_reward / total_reward) - 1)





