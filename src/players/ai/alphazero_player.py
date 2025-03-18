"""
AlphaZero player for Kulibrat.
"""

from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_player import AIPlayer
from src.players.ai.alphazero_strategy import AlphaZeroStrategy


class AlphaZeroPlayer(AIPlayer):
    """
    Player that uses AlphaZero-style MCTS to select moves.
    """
    
    def __init__(self, color: PlayerColor, model_path: str, 
                 n_simulations: int = 800,
                 c_puct: float = 4.0,
                 exploration_rate: float = 0.0, 
                 temperature: float = 0.5,
                 name: str = None):
        """
        Initialize the AlphaZero player.
        
        Args:
            color: The player's color (BLACK or RED)
            model_path: Path to the trained model file
            n_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for MCTS
            exploration_rate: Probability of selecting a random move (0 for tournament play)
            temperature: Temperature for move selection (lower for stronger play)
            name: Optional custom name for the player
        """
        strategy = AlphaZeroStrategy(
            model_path=model_path,
            n_simulations=n_simulations,
            c_puct=c_puct,
            exploration_rate=exploration_rate,
            temperature=temperature
        )
        self.model_path = model_path
        
        if name is None:
            name = f"AlphaZero {color.name}"
        
        super().__init__(color, strategy, name)
    
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Get a move using the AlphaZero strategy.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            The selected move or None if no valid moves
        """
        if game_state.current_player != self.color:
            return None
        
        return self.strategy.select_move(game_state, self.color)