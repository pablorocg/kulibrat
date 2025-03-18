"""
Attention-based neural network model for reinforcement learning in Kulibrat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional


class SelfAttention(nn.Module):
    """Self-attention layer for board state processing"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x, _ = self.multihead_attn(x, x, x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class FeedForward(nn.Module):
    """Feed-forward network following attention layer"""
    
    def __init__(self, embed_dim: int, ff_dim: int = 256, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout1(F.gelu(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        x = self.layer_norm(x + residual)
        return x


class TransformerBlock(nn.Module):
    """Transformer block combining self-attention and feed-forward layers"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block for the neural network.
    """

    def __init__(self, num_filters: int):
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
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Residual connection
        x = F.relu(x)
        return x


def encode_board_for_attention(board: np.ndarray, current_player: int) -> torch.Tensor:
    """
    Encode the board state as input for the attention-based neural network.

    Args:
        board: NumPy array representing the board (4x3 for Kulibrat)
              with 1 for BLACK, -1 for RED, 0 for empty
        current_player: Current player (1 for BLACK, -1 for RED)

    Returns:
        Tensor of shape (batch_size=1, positions=12, features)
    """
    # Board dimensions
    rows, cols = board.shape
    
    # Create standard channels like in the original encode_board function
    black_channel = (board == 1).astype(np.float32)
    red_channel = (board == -1).astype(np.float32)
    empty_channel = (board == 0).astype(np.float32)
    player_channel = np.ones((rows, cols), dtype=np.float32) if current_player == 1 else np.zeros((rows, cols), dtype=np.float32)
    
    # Add additional features useful for attention mechanisms
    
    # Distance from start rows (helps with strategic evaluation)
    black_distance = np.zeros((rows, cols), dtype=np.float32)
    red_distance = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(rows):
        black_distance[r, :] = r / (rows - 1)  # Distance from BLACK's start row (normalized)
        red_distance[r, :] = (rows - 1 - r) / (rows - 1)  # Distance from RED's start row (normalized)
    
    # Feature indicating if a piece can potentially score in the next move
    black_can_score = np.zeros((rows, cols), dtype=np.float32)
    red_can_score = np.zeros((rows, cols), dtype=np.float32)
    
    if rows > 1:  # Safety check
        black_can_score[rows-2, :] = black_channel[rows-2, :]  # BLACK pieces one row away from scoring
        red_can_score[1, :] = red_channel[1, :]  # RED pieces one row away from scoring
    
    # Feature indicating if a piece can be attacked
    black_can_be_attacked = np.zeros((rows, cols), dtype=np.float32)
    red_can_be_attacked = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(1, rows-1):
        for c in range(cols):
            if board[r, c] == 1:  # BLACK piece
                if board[r+1, c] == 0:  # Empty space in front (from RED's perspective)
                    black_can_be_attacked[r, c] = 1
            elif board[r, c] == -1:  # RED piece
                if board[r-1, c] == 0:  # Empty space in front (from BLACK's perspective)
                    red_can_be_attacked[r, c] = 1
    
    # Stack all channels
    features = np.stack([
        black_channel,         # Channel 0: BLACK pieces
        red_channel,           # Channel 1: RED pieces
        empty_channel,         # Channel 2: Empty spaces
        player_channel,        # Channel 3: Current player
        black_distance,        # Channel 4: Distance from BLACK's start row
        red_distance,          # Channel 5: Distance from RED's start row
        black_can_score,       # Channel 6: BLACK pieces that can potentially score
        red_can_score,         # Channel 7: RED pieces that can potentially score
        black_can_be_attacked, # Channel 8: BLACK pieces that can be attacked
        red_can_be_attacked    # Channel 9: RED pieces that can be attacked
    ])
    
    # Number of features per position
    num_features = features.shape[0]
    
    # Reshape to (positions, features)
    features = features.reshape(num_features, -1).T
    
    # Convert to tensor (add batch dimension)
    return torch.FloatTensor(features).unsqueeze(0)


class AttentionKulibratNet(nn.Module):
    """
    Neural network that evaluates Kulibrat board positions using attention mechanisms.
    The network takes a board state as input and outputs:
    1. A policy (probability distribution over possible moves)
    2. A value (estimated advantage of the current player)
    """

    def __init__(
        self,
        board_size: Tuple[int, int] = (4, 3),
        input_features: int = 10,  # Number of features per position in encoded board
        embed_dim: int = 64,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_filters: int = 64,
        num_res_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super(AttentionKulibratNet, self).__init__()
        self.board_size = board_size
        rows, cols = board_size
        self.positions = rows * cols
        
        # Input embedding layer
        self.embedding = nn.Linear(input_features, embed_dim)
        
        # Transformer blocks for attention-based processing
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Reshape parameters for the transition to convolutional layers
        self.reshape_conv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=1
        )
        self.reshape_bn = nn.BatchNorm2d(num_filters)
        
        # Residual blocks for local pattern recognition
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
        policy_flat_size = 2 * rows * cols
        
        # Policy output (51 possible actions from the original implementation)
        num_actions = 51
        self.policy_fc = nn.Linear(policy_flat_size, num_actions)
        
        # Value head
        self.value_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=1,  # 1 filter for value
            kernel_size=1,
        )
        self.value_bn = nn.BatchNorm2d(1)
        
        # Value hidden layers and output
        value_flat_size = rows * cols
        self.value_fc1 = nn.Linear(value_flat_size, 64)
        self.value_fc2 = nn.Linear(64, 32)
        self.value_fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, positions, features)
                from encode_board_for_attention function

        Returns:
            Tuple of (policy_logits, value)
        """
        batch_size = x.size(0)
        rows, cols = self.board_size
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape for convolutional processing 
        # From (batch, positions, embed_dim) to (batch, embed_dim, rows, cols)
        x = x.reshape(batch_size, self.positions, -1)
        x = x.transpose(1, 2).reshape(batch_size, -1, rows, cols)
        
        # Apply initial convolution to transition from attention to conv layers
        x = F.relu(self.reshape_bn(self.reshape_conv(x)))
        
        # Apply residual blocks
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
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))  # Output between -1 and 1
        
        return policy_logits, value.squeeze(-1)


def decode_action(action_idx: int, board: np.ndarray, current_player: int) -> Dict:
    """
    Decode an action index into a Kulibrat move.
    
    Args:
        action_idx: Index of the chosen action
        board: Current board state
        current_player: Current player (1 for BLACK, -1 for RED)
        
    Returns:
        Dictionary with move information
    """
    rows, cols = board.shape
    
    # Define action types based on Kulibrat rules
    ACTION_INSERT = 0
    ACTION_DIAGONAL = 1
    ACTION_ATTACK = 2
    ACTION_JUMP = 3
    
    # For simplicity, we map actions to positions and types
    # The actual mapping would depend on how the model was trained
    action_type = action_idx // (rows * cols)
    position_idx = action_idx % (rows * cols)
    row = position_idx // cols
    col = position_idx % cols
    
    # Build the move dictionary based on action type
    move = {"type": None, "from_row": row, "from_col": col}
    
    if action_type == ACTION_INSERT:
        # INSERT: Place a new piece on the player's starting row
        start_row = 0 if current_player == 1 else rows-1
        move = {
            "type": "INSERT",
            "row": start_row,
            "col": col
        }
    elif action_type == ACTION_DIAGONAL:
        # DIAGONAL: Move diagonally forward
        direction = 1 if current_player == 1 else -1
        move = {
            "type": "DIAGONAL",
            "from_row": row,
            "from_col": col,
            "to_row": row + direction,
            "to_col": col + (1 if col < cols//2 else -1)  # Move right if on left side, left if on right side
        }
    elif action_type == ACTION_ATTACK:
        # ATTACK: Move forward and capture opponent piece
        direction = 1 if current_player == 1 else -1
        move = {
            "type": "ATTACK",
            "from_row": row,
            "from_col": col,
            "to_row": row + direction,
            "to_col": col
        }
    elif action_type == ACTION_JUMP:
        # JUMP: Jump over opponent pieces
        # This is a simplified version - the actual jump logic is more complex
        direction = 1 if current_player == 1 else -1
        move = {
            "type": "JUMP",
            "from_row": row,
            "from_col": col,
            "to_row": row + direction * 2,  # Jump 2 spaces (assuming 1 opponent piece)
            "to_col": col
        }
    
    return move


# Example usage
if __name__ == "__main__":
    # Create a sample board
    board = np.zeros((4, 3), dtype=np.int8)
    
    # Set up some pieces
    # 1 for BLACK, -1 for RED, 0 for empty
    board[0, 1] = 1     # BLACK piece on starting row
    board[2, 0] = 1     # BLACK piece in middle
    board[3, 0] = -1    # RED piece on starting row
    board[1, 2] = -1    # RED piece in middle
    
    # Current player
    current_player = 1  # BLACK
    
    # Encode board for attention
    encoded_board = encode_board_for_attention(board, current_player)
    print(f"Encoded board shape: {encoded_board.shape}")
    
    # Initialize model
    model = AttentionKulibratNet()
    
    # Forward pass
    policy_logits, value = model(encoded_board)
    
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value: {value.item():.4f}")
    
    # Get best action
    best_action = torch.argmax(policy_logits, dim=1).item()
    print(f"Best action: {best_action}")
    
    # Decode the action
    move = decode_action(best_action, board, current_player)
    print(f"Move: {move}")