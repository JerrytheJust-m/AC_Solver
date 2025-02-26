"""
The training loop for the research sibling method, similar to PPO, but containing regulations to randomness.
"""
import argparse
import math
import random
import uuid
import wandb
from collections import deque
from tqdm import tqdm
from os import makedirs
from os.path import join
import numpy as np
import torch
from torch import nn

from agents.training import get_curr_lr


def sibling_training_loop(
        envs,
        args,
        device,
        agent_list,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
):
    #set up global parameters during training
    global_step = 0
    episodic_length = np.array([0] * args.num_envs)
    num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}_sibling-ppo-ffn-nodes_{args.nodes_counts}_{uuid.uuid4()}"
    out_dir = f"sibling_out/{run_name}"
    makedirs(out_dir, exist_ok=True)
    beta = None if args.is_loss_clip else args.beta

    total_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    #initialize the observations and actions of the agents
    for agent in agent_list:
        agent_list[agent]["episode"] = 0
        agent_list[agent]["obs"] = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        agent_list[agent]["actions"] = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        agent_list[agent]["dones"] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        agent_list[agent]["rewards"] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        agent_list[agent]["logprobs"] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        agent_list[agent]["values"] = torch.zeros((args.num_steps, args.num_envs)).to(device)
        agent_list[agent]["episodic_return"] = np.array([0] * args.num_envs)
        agent_list[agent]["episodic_length"] = np.array([0] * args.num_envs)
        agent_list[agent]["returns_queue"] = deque([0], maxlen=100)
        agent_list[agent]["lengths_queue"] = deque([0], maxlen=100)
        agent_list[agent]["round1_complete"] = False  # whether we have already chosen each element of initial_states at least once to initiate rollout

        #move first step
        agent_list[agent]["next_obs"] = torch.Tensor(envs.reset()[0]).to(device)  # get first observation
        agent_list[agent]["next_done"] = torch.zeros(args.num_envs).to(device)  # get first done

    print(f"total number of timesteps: {args.total_timesteps}, updates: {num_updates}")

    #now begins training loop, in each step, all agents move one step
    for update in tqdm(range(1, num_updates + 1), desc="Training Progress", total=num_updates):
        # using different seed for each update to ensure reproducibility of paused-and-resumed runs
        random.seed(args.seed + update)
        np.random.seed(args.seed + update)
        torch.manual_seed(args.seed + update)

        # collecting and recording data
        for step in tqdm(range(0, args.num_steps), leave = False, desc=f"Rollout Phase - {update}"):
            global_step += 1 * args.num_envs
            #do each step separately for each agent.
            for agent in agent_list:
                agent_list[agent]["obs"][step] = agent_list[agent]["next_obs"]
                agent_list[agent]["dones"][step] = agent_list[agent]["next_done"]

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent_list[agent]["agent"].get_action_and_value(
                        agent_list[agent]["next_obs"]
                    )  # shapes: n_envs, n_envs, n_envs, (n_envs, 1)
                    agent_list[agent]["values"][step] = value.flatten()  # num_envs
                agent_list[agent]["actions"][step] = action
                agent_list[agent]["logprobs"][step] = logprob

                #get next step observations
                next_obs, reward, done, truncated, infos = envs.step(action.cpu().numpy()) #on cpu
                reward_float32 = reward.astype(np.float32)
                # TODO: float32 is only to accustom to mps
                agent_list[agent]["rewards"][step] = (
                    torch.tensor(reward_float32).to(device).view(-1)
                )  # r_0 is the reward from taking a_0 in s_0
                agent_list[agent]["episodic_return"] = agent_list[agent]["episodic_return"] + reward
                agent_list[agent]["episodic_length"] = agent_list[agent]["episodic_length"] + 1

                #checking current situations
                _record_info = np.array(
                    [
                        True if done[i] or truncated[i] else False
                        for i in range(args.num_envs)
                    ]
                )

                #log info if some presentations are done or truncated
                if _record_info.any():
                    for i, el in enumerate(_record_info):
                        if done[i]:
                            # if done, add curr_states[i] to 'solved' cases
                            if curr_states[i] in success_record["unsolved"]:
                                success_record["unsolved"].remove(curr_states[i])
                                success_record["solved"].add(curr_states[i])

                            # also if done, record the sequence of actions in ACMoves_hist
                            if curr_states[i] not in ACMoves_hist:
                                # TODO: verify that changing finial_info to actions solves the bug of "final_info"
                                ACMoves_hist[curr_states[i]] = infos["actions"]
                            else:
                                prev_path_length = len(ACMoves_hist[curr_states[i]])
                                new_path_length = len(infos["actions"])
                                if new_path_length < prev_path_length:
                                    ACMoves_hist[curr_states[i]] = infos["actions"]

                        # record+reset episode data, reset ith initial state to the next state in init_states
                        if el:
                            # record and reset episode data
                            agent_list[agent]["returns_queue"].append(agent_list[agent]["episodic_return"][i])
                            agent_list[agent]["lengths_queue"].append(episodic_length[i])
                            agent_list[agent]["episode"] += 1
                            agent_list[agent]["episodic_return"][i], episodic_length[i] = 0, 0

                            # update next_obs to have the next initial state
                            prev_state = curr_states[i]
                            round1_complete = (
                                True
                                if agent_list[agent]["round1_complete"]
                                or (max(states_processed) == len(initial_states) - 1)
                                else False
                            )
                            if not agent_list[agent]["round1_complete"]:
                                curr_states[i] = max(states_processed) + 1
                            else:
                                # TODO: If states-type=all, first choose from all solved presentations then choose from unsolved presentations
                                if len(success_record["solved"]) == 0 or (
                                    success_record["unsolved"]
                                    and random.uniform(0, 1) > args.repeat_solved_prob
                                ):
                                    curr_states[i] = random.choice(
                                        list(success_record["unsolved"])
                                    )
                                else:
                                    curr_states[i] = random.choice(
                                        list(success_record["solved"])
                                    )
                            states_processed.add(curr_states[i])
                            next_obs[i] = initial_states[curr_states[i]]
                            envs.envs[i].reset(options={"starting_state": next_obs[i]})

                agent_list[agent]["next_obs"], agent_list[agent]["next_done"] = torch.Tensor(agent_list[agent]["next_obs"]).to(device), torch.Tensor(
                    done
                ).to(device)


