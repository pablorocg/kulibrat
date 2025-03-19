"""
tournament factory
"""

from typing import Any, Dict
import logging

from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.mcts_strategy import MCTSStrategy
from src.core.player_color import PlayerColor


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

        try:
            if player_type == 'random':
                return SimpleAIPlayer(
                    color=PlayerColor.BLACK,  # Will be set later
                    strategy=RandomStrategy(),
                    name=name
                )

            elif player_type == 'minimax':
                return SimpleAIPlayer(
                    color=PlayerColor.BLACK,  # Will be set later
                    strategy=MinimaxStrategy(
                        max_depth=player_config.get('depth', 4),
                        use_alpha_beta=player_config.get('use_alpha_beta', True),
                        heuristic=player_config.get('heuristic', 'strategic'),
                        tt_size=player_config.get('tt_size', 1000000)
                    ),
                    name=name
                )

            elif player_type == 'mcts':
                return SimpleAIPlayer(
                    color=PlayerColor.BLACK,  # Will be set later
                    strategy=MCTSStrategy(
                        simulation_time=player_config.get('simulation_time', 1.0),
                        max_iterations=player_config.get('max_iterations', 15000)
                    ),
                    name=name
                )

            else:
                raise ValueError(f"Unknown player type: {player_type}")

        except Exception as e:
            logging.error(f"Error creating player {name}: {e}")
            return None