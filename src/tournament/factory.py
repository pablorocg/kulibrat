"""
Tournament factory for creating players for tournament evaluation.
"""

from typing import Any, Dict
import logging

from src.core.player_color import PlayerColor
from src.players.random_player.random_player import RandomPlayer
from src.players.minimax_player.minimax_player import MinimaxPlayer
from src.players.mcts_player.mcts_player import MCTSPlayer


class AIPlayerFactory:
    """Factory for creating AI players based on configuration."""

    @staticmethod
    def create_player(player_config: Dict[str, Any]) -> Any:
        """
        Create an AI player based on configuration.

        Args:
            player_config: Dictionary with player configuration

        Returns:
            Configured AI player
        """
        player_type = player_config['type']
        name = player_config.get('name', f'{player_type}-default')
        
        # Initialize with BLACK color - will be set properly in the tournament
        color = PlayerColor.BLACK

        try:
            if player_type == 'random':
                return RandomPlayer(color=color, name=name)

            elif player_type == 'minimax':
                return MinimaxPlayer(
                    color=color,
                    name=name,
                    max_depth=player_config.get('depth', 4),
                    use_alpha_beta=player_config.get('use_alpha_beta', True),
                    heuristic=player_config.get('heuristic', 'strategic'),
                    tt_size=player_config.get('tt_size', 1000000)
                )

            elif player_type == 'mcts':
                return MCTSPlayer(
                    color=color,
                    name=name,
                    simulation_time=player_config.get('simulation_time', 1.0),
                    max_iterations=player_config.get('max_iterations', 15000),
                    exploration_weight=player_config.get('exploration_weight', 1.41),
                    num_threads=player_config.get('num_threads', 4)
                )

            else:
                raise ValueError(f"Unknown player type: {player_type}")

        except Exception as e:
            logging.error(f"Error creating player {name}: {e}")
            return None