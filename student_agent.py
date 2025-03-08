# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.


#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

# student_agent.py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize the model
state_size = 16  # Based on SimpleTaxiEnv observation
action_size = 6  # 6 possible actions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNetwork(state_size, action_size).to(device)
model.load_state_dict(torch.load("taxi_dqn.pth"))
model.eval()

def get_action(obs):
    """
    Returns the best action for the given observation
    obs: tuple of (taxi_row, taxi_col, station coordinates, obstacles, passenger_look, destination_look)
    """
    state = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state)
    return q_values.max(1)[1].item()
