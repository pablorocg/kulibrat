"""
Tournament factory for creating players for tournament evaluation.
"""

from typing import Any, Dict
import logging

from src.core.player_color import PlayerColor
from src.players.random_player.random_player import RandomPlayer
from src.players.minimax_player.minimax_player import MinimaxPlayer
from src.players.mcts_player.mcts_player import MCTSPlayer
from src.players.negascout_player.negascout_player import NegaScoutPlayer


class AIPlayerFactory:
    """Factory for creating AI players based on configuration."""

    # Enhanced logging for AIPlayerFactory.create_player method
    @staticmethod
    def create_player(player_config: Dict[str, Any]) -> Any:
        """
        Create an AI player based on configuration.

        Args:
            player_config: Dictionary with player configuration

        Returns:
            Configured AI player
        """
        logger = logging.getLogger(__name__)
        
        try:
            player_type = player_config['type']
            name = player_config.get('name', f'{player_type}-default')
            
            logger.info(f"Creating player '{name}' of type '{player_type}'")
            logger.debug(f"Player configuration: {player_config}")
            
            # Initialize with BLACK color - will be set properly in the tournament
            color = PlayerColor.BLACK

            if player_type == 'random':
                logger.info(f"Creating RandomPlayer: {name}")
                return RandomPlayer(color=color, name=name)

            elif player_type == 'minimax':
                depth = player_config.get('depth', 4)
                use_alpha_beta = player_config.get('use_alpha_beta', True)
                heuristic = player_config.get('heuristic', 'strategic')
                tt_size = player_config.get('tt_size', 1000000)
                
                logger.info(f"Creating MinimaxPlayer: {name} (depth={depth}, alpha-beta={use_alpha_beta}, heuristic={heuristic})")
                return MinimaxPlayer(
                    color=color,
                    name=name,
                    max_depth=depth,
                    use_alpha_beta=use_alpha_beta,
                    heuristic=heuristic,
                    tt_size=tt_size
                )

            elif player_type == 'mcts':
                simulation_time = player_config.get('simulation_time', 1.0)
                max_iterations = player_config.get('max_iterations', 15000)
                exploration_weight = player_config.get('exploration_weight', 1.41)
                
                logger.info(f"Creating MCTSPlayer: {name} (sim_time={simulation_time}, max_iter={max_iterations})")
                return MCTSPlayer(
                    color=color,
                    name=name,
                    simulation_time=simulation_time,
                    max_iterations=max_iterations,
                    exploration_weight=exploration_weight,  
                )
            
            elif player_type == 'negascout':
                time_limit = player_config.get('time_limit', 1.0)
                depth = player_config.get('depth', 4)
                heuristic = player_config.get('heuristic', 'score_diff')
                tt_size = player_config.get('tt_size', 1000000)
                
                logger.info(f"Creating NegaScoutPlayer: {name} (depth={depth}, heuristic={heuristic})")
                return NegaScoutPlayer(
                    color=color,
                    name=name,
                    time_limit=time_limit,
                    max_depth=depth,
                    heuristic=heuristic,
                    tt_size=tt_size
                )

            else:
                logger.error(f"Unknown player type: {player_type}")
                raise ValueError(f"Unknown player type: {player_type}")

        except Exception as e:
            logger.exception(f"Error creating player {player_config.get('name', 'unknown')}: {e}")
            return None