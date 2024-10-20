#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network as per the original DQN paper."""

    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the flattened features after
        # the conv layers
        # Assuming input image size of 84x84
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        size = 84  # Input image size (height and width)
        size = conv2d_size_out(size, kernel_size=8, stride=4)  # After conv1
        size = conv2d_size_out(size, kernel_size=4, stride=2)  # After conv2
        size = conv2d_size_out(size, kernel_size=3, stride=1)  # After conv3

        linear_input_size = size * size * 64  # 64 is the number of filters from conv3

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # Input x has shape (batch_size, in_channels, height, width)
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from convolutional layers
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, flattened_features)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  

        return x  # Returns Q-values for each action
    

class DuelingDQN(nn.Module):

    def __init__(self, in_channels=4, num_actions=4):
        super(DuelingDQN, self).__init__()

        # Convolutional layers (same as before)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the flattened features after the conv layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        size = 84  # Assuming input image size of 84x84
        size = conv2d_size_out(size, 8, 4)  # After conv1
        size = conv2d_size_out(size, 4, 2)  # After conv2
        size = conv2d_size_out(size, 3, 1)  # After conv3

        linear_input_size = size * size * 64  # Flattened size after conv layers

        # Fully connected layer before splitting into state and advantage streams
        self.fc = nn.Linear(linear_input_size, 512)

        # State-value stream (outputs a single value V(s))
        self.value_stream = nn.Linear(512, 1)

        # Advantage stream (outputs one advantage for each action A(s, a))
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)

        # Pass through the shared fully connected layer
        x = F.relu(self.fc(x))

        # Compute the state-value and advantage separately
        value = self.value_stream(x)  # State-value V(s)
        advantage = self.advantage_stream(x)  # Advantage A(s, a)

        # Combine them into the final Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values