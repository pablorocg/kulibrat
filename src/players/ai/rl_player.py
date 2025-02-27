"""
Reinforcement Learning player for Kulibrat.
"""

from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_player import AIPlayer
from src.players.ai.rl_strategy import RLStrategy


class RLPlayer(AIPlayer):
    """
    Player that uses a trained neural network to select moves.
    """
    
    def __init__(self, color: PlayerColor, model_path: str, 
                 exploration_rate: float = 0.0, 
                 temperature: float = 0.5,
                 name: str = None):
        """
        Initialize the reinforcement learning player.
        
        Args:
            color: The player's color (BLACK or RED)
            model_path: Path to the trained model file
            exploration_rate: Probability of selecting a random move (0 for tournament play)
            temperature: Temperature for policy sampling (lower for stronger play)
            name: Optional custom name for the player
        """
        strategy = RLStrategy(
            model_path=model_path,
            exploration_rate=exploration_rate,
            temperature=temperature
        )
        
        if name is None:
            name = f"RL {color.name}"
        
        super().__init__(color, strategy, name)
    
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Get a move using the RL strategy.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            The selected move or None if no valid moves
        """
        if game_state.current_player != self.color:
            return None
        
        return self.strategy.select_move(game_state, self.color)