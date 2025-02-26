import numpy as np
import torch
import random
from torch.optim import Adam
from research_sibling_agent import SiblingAgent
from agents.args import parse_args
from agents.environment import get_env
from research_sibling_train import sibling_training_loop

def train_sibling():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    #TODO: this is for running with mac alternative
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "mps")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)
    print(curr_states)

    agent_list = {}
    agent_list["brother"] = {"agent": SiblingAgent(envs, args.nodes_counts, np.float32(0.2), args, device).to(device)}
    agent_list["sister"] = {"agent": SiblingAgent(envs, args.nodes_counts, np.float32(0.8), args, device).to(device)}
    for agent in agent_list:
         agent_list[agent]["optimizer"] = Adam(agent_list[agent]["agent"].parameters(), lr=args.learning_rate, eps=args.epsilon)

    sibling_training_loop(
        envs,
        args,
        device,
        agent_list,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
    )

    envs.close()


if __name__ == "__main__":
    train_sibling()