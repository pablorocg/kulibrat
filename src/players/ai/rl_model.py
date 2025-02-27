"""
Neural network model for reinforcement learning in Kulibrat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class KulibratNet(nn.Module):
    """
    Neural network that evaluates Kulibrat board positions.
    The network takes a board state as input and outputs:
    1. A policy (probability distribution over possible moves)
    2. A value (estimated advantage of the current player)
    """

    def __init__(
        self,
        board_size: Tuple[int, int] = (4, 3),
        num_filters: int = 32,
        num_res_blocks: int = 3,
    ):
        """
        Initialize the neural network.

        Args:
            board_size: Size of the board (rows, columns)
            num_filters: Number of filters in convolutional layers
            num_res_blocks: Number of residual blocks
        """
        super(KulibratNet, self).__init__()
        self.board_size = board_size
        self.num_filters = num_filters

        # Input channels:
        # 1. Black pieces
        # 2. Red pieces
        # 3. Current player (constant plane of 1's if current player is black, 0's otherwise)
        self.input_channels = 3

        # Initial convolutional layer
        self.conv_initial = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn_initial = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=2,  # 2 filters for policy
            kernel_size=1,
        )
        self.policy_bn = nn.BatchNorm2d(2)

        # Flattened size for policy head
        policy_flat_size = 2 * board_size[0] * board_size[1]

        # Policy output (one for each possible position, plus insert moves)
        # We'll have:
        # - Move type (4 types: INSERT, DIAGONAL, ATTACK, JUMP)
        # - Start position (board_size[0] * board_size[1] possibilities or none)
        # - End position (board_size[0] * board_size[1] possibilities or none)
        # For simplicity, we'll output a probability for each possible action
        # (we can have maximum board_size[0]*board_size[1]*4 different actions)
        num_actions = 46 # board_size[0] * board_size[1] * 4
        self.policy_fc = nn.Linear(policy_flat_size, num_actions)

        # Value head
        self.value_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=1,  # 1 filter for value
            kernel_size=1,
        )
        self.value_bn = nn.BatchNorm2d(1)

        # Flattened size for value head
        value_flat_size = board_size[0] * board_size[1]

        # Value hidden layer and output
        self.value_fc1 = nn.Linear(value_flat_size, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_channels, board_rows, board_cols)

        Returns:
            Tuple of (policy_logits, value)
        """
        # Initial layers
        x = F.relu(self.bn_initial(self.conv_initial(x)))

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output between -1 and 1

        return policy_logits, value


class ResBlock(nn.Module):
    """
    Residual block for the neural network.
    """

    def __init__(self, num_filters: int):
        """
        Initialize a residual block.

        Args:
            num_filters: Number of filters in convolutional layers
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Residual connection
        x = F.relu(x)
        return x


def encode_board(board: np.ndarray, current_player: int) -> torch.Tensor:
    """
    Encode the board state as input for the neural network.

    Args:
        board: NumPy array representing the board
        current_player: Current player (1 for BLACK, -1 for RED)

    Returns:
        Tensor of shape (batch_size=1, channels=3, rows, cols)
    """
    # Board dimensions
    rows, cols = board.shape

    # Create 3 channels:
    # - Channel 0: Black pieces (1 where black pieces are, 0 elsewhere)
    # - Channel 1: Red pieces (1 where red pieces are, 0 elsewhere)
    # - Channel 2: Current player (all 1's if BLACK, all 0's if RED)
    black_channel = (board == 1).astype(np.float32)
    red_channel = (board == -1).astype(np.float32)
    player_channel = (
        np.ones((rows, cols), dtype=np.float32)
        if current_player == 1
        else np.zeros((rows, cols), dtype=np.float32)
    )

    # Stack channels
    encoded = np.stack([black_channel, red_channel, player_channel])

    # Convert to tensor (add batch dimension)
    return torch.FloatTensor(encoded).unsqueeze(0)
