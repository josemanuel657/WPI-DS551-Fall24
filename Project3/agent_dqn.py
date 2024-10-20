#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import wandb

from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

from environment import Environment

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


EPSILON_START = 0.9
EPSILON_END = 0.025
NUM_GAMES = 1000000
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / NUM_GAMES
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_MEMORY_SIZE = 10000
START_REPLAY_MEMORY_SIZE = 5000
STEPS_FOR_UPDATE = 5000


wandb_config = {
    "project": "dqn-breakout",
    "config": {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay": EPSILON_DECAY,
        "replay_memory_size": REPLAY_MEMORY_SIZE,
        "start_replay_memory_size": START_REPLAY_MEMORY_SIZE,
        "steps_for_update": STEPS_FOR_UPDATE,
        "num_games": NUM_GAMES
    }
}

# Initialize wandb
wandb.init(
    project=wandb_config["project"],
    config=wandb_config["config"]
)


# Select to use CPU, CUDA, or MPS
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# Define the transition tuple to store state transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):

        self.memory.append(Transition(*args))

    def sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)
    
    


class Agent_DQN(Agent):
    
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

        self.env = env
        self.eps = EPSILON_START
        
        wandb.watch(self.policy_net, log="all")
    
        if args.test_dqn:

            print('loading trained model')
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        


        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device) ## format tensor(BATCH_SIZE, channels, height, width)
        
        if random.random() < self.eps and not test:  # Exploration
            return self.env.get_random_action()
        else:  # Exploitation
            with torch.no_grad():
                q_values = self.policy_net(observation)
                action = q_values.max(1)[1].item()  # Get the index (action) with the highest Q-value
        return action
    
        

    def train(self):
        
        global_steps = 0

        for episode in range(NUM_GAMES):
           
            episode_steps = 0
            episode_reward = 0
            
            total_loss = 0
            total_q_value = 0
            loss_count = 0

            observation = self.env.reset()  
            
            while True:
                action = self.make_action(observation, test=False)
                next_observation, reward, done, truncated, info = self.env.step(action)
                
                if done or truncated: break


                self.replay_buffer.push(observation, action, next_observation, reward)
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                global_steps += 1
                
                if len(self.replay_buffer) >= START_REPLAY_MEMORY_SIZE and len(self.replay_buffer) >= BATCH_SIZE:
                    step_loss, step_q_value = self.update_policy()
                    total_loss += step_loss
                    total_q_value += step_q_value
                    loss_count += 1
                    
                if global_steps % STEPS_FOR_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())  
                    global_steps = 0
                    
            average_loss = total_loss / loss_count if loss_count > 0 else 0
            average_q_value = total_q_value / loss_count if loss_count > 0 else 0
                

            wandb.log({
                "episode_number": episode,
                "epsilon": self.eps,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "average_model_loss": average_loss,
                "average_model_q_value": average_q_value
            })
            
            if episode % 100000 == 0:
                torch.save(self.policy_net.state_dict(), f'model_episode_{episode}.pth')
                
            self.eps = max(EPSILON_END, self.eps - EPSILON_DECAY)


                

    def update_policy(self):

        transitions = self.replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Convert batch data to numpy arrays before converting to tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        state_batch = state_batch.permute(0, 3, 1, 2)  # Permute to (BATCH_SIZE, channels, height, width)

        action_batch = torch.tensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(device).unsqueeze(1)

        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
        next_state_batch = next_state_batch.permute(0, 3, 1, 2)  # Permute to (BATCH_SIZE, channels, height, width)

        # Compute Q-values for current state
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q-values for next state (target)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            
        expected_q_values = reward_batch + (GAMMA * next_q_values)

        # Compute loss (Huber loss)
        huber_loss = F.smooth_l1_loss(q_values, expected_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        huber_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
            
        return (huber_loss.item(), q_values.mean().item())
