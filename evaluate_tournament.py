#!/usr/bin/env python3
"""
Advanced Tournament Evaluation for Kulibrat AI Players.

This script runs comprehensive tournaments between different AI strategies,
generating detailed performance metrics and visualizations.
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional

# Kulibrat core imports
from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.move_type import MoveType
from src.core.move import Move

# Player and strategy imports
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.mcts_strategy import MCTSStrategy



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
                        use_alpha_beta=player_config.get('use_alpha_beta', True)
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


class TournamentMatch:
    """Manages a single match between two players."""

    def __init__(
        self, 
        player1_name: str, 
        player1, 
        player2_name: str, 
        player2, 
        target_score: int = 5,
        max_turns: int = 300
    ):
        """
        Initialize tournament match configuration.

        Args:
            player1_name: Name of first player
            player1: First player object
            player2_name: Name of second player
            player2: Second player object
            target_score: Score needed to win
            max_turns: Maximum turns before draw
        """
        self.player1_name = player1_name
        self.player1 = player1
        self.player2_name = player2_name
        self.player2 = player2
        self.target_score = target_score
        self.max_turns = max_turns

        # Match result tracking
        self.match_results = {
            'player1': player1_name,
            'player2': player2_name,
            'winner': None,
            'score_p1': 0,
            'score_p2': 0,
            'turns': 0,
            'total_time': 0,
            'player1_move_types': {m.name: 0 for m in MoveType},
            'player2_move_types': {m.name: 0 for m in MoveType}
        }

    def run_match(self, swap_colors: bool = False) -> Dict[str, Any]:
        """
        Execute a match between two players.

        Args:
            swap_colors: Whether to swap default colors

        Returns:
            Detailed match results
        """
        # Determine player colors
        if not swap_colors:
            black_player = self.player1
            red_player = self.player2
            black_player_name = self.player1_name
            red_player_name = self.player2_name
        else:
            black_player = self.player2
            red_player = self.player1
            black_player_name = self.player2_name
            red_player_name = self.player1_name

        # Set player colors
        black_player.color = PlayerColor.BLACK
        red_player.color = PlayerColor.RED

        # Initialize game state
        game_state = GameState(target_score=self.target_score)

        # Track match progress
        turns = 0
        start_time = datetime.now()

        while not game_state.is_game_over() and turns < self.max_turns:
            # Determine current player
            current_color = game_state.current_player
            current_player = black_player if current_color == PlayerColor.BLACK else red_player
            current_player_name = black_player_name if current_color == PlayerColor.BLACK else red_player_name

            # Get player move
            move = current_player.get_move(game_state)

            if not move:
                # Skip turn if no move possible
                game_state.current_player = game_state.current_player.opposite()
                continue

            # Apply move and track statistics
            if game_state.apply_move(move):
                # Track move types
                move_types_dict = self.match_results['player1_move_types'] if current_player_name == self.player1_name else self.match_results['player2_move_types']
                move_types_dict[move.move_type.name] += 1

                # Switch players and increment turns
                game_state.current_player = game_state.current_player.opposite()
                turns += 1

        # Calculate match duration
        end_time = datetime.now()
        match_duration = (end_time - start_time).total_seconds()

        # Determine winner
        winner = game_state.get_winner()
        if winner == PlayerColor.BLACK:
            winner_name = black_player_name
        elif winner == PlayerColor.RED:
            winner_name = red_player_name
        else:
            winner_name = None

        # Update match results
        self.match_results.update({
            'winner': winner_name,
            'score_p1': game_state.scores[PlayerColor.BLACK] if not swap_colors else game_state.scores[PlayerColor.RED],
            'score_p2': game_state.scores[PlayerColor.RED] if not swap_colors else game_state.scores[PlayerColor.BLACK],
            'turns': turns,
            'total_time': match_duration
        })

        return self.match_results


class TournamentEvaluator:
    """Manages tournament execution and result analysis."""

    def __init__(self, config_path: str):
        """
        Initialize tournament evaluator.

        Args:
            config_path: Path to tournament configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize results storage
        self.tournament_results = []
        self.summary_stats = {}

    def run_tournament(self):
        """Execute the full tournament."""
        # Extract tournament settings
        matches_per_pairing = self.config['tournament'].get('matches_per_pairing', 10)
        target_score = self.config['tournament'].get('target_score', 5)
        max_turns = self.config['tournament'].get('max_turns', 300)

        # Create output directory
        output_dir = self.config['output'].get('results_dir', 'tournament_results')
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp for unique results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_dir, f"tournament_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # Create players
        player_configs = self.config['players']
        players = {}
        for player_config in player_configs:
            player = AIPlayerFactory.create_player(player_config)
            if player:
                players[player_config['name']] = player

        # Run tournament
        player_names = list(players.keys())
        total_matches = len(player_names) * (len(player_names) - 1) * matches_per_pairing

        self.logger.info(f"Starting tournament with {len(players)} players")
        self.logger.info(f"Total matches: {total_matches}")

        # Track progress
        current_match = 0

        # Run matches between all player pairs
        for i, player1_name in enumerate(player_names):
            for j, player2_name in enumerate(player_names):
                if i == j:  # Skip matches with same player
                    continue

                player1 = players[player1_name]
                player2 = players[player2_name]

                # Create match handler
                match_handler = TournamentMatch(
                    player1_name, 
                    player1, 
                    player2_name, 
                    player2,
                    target_score=target_score,
                    max_turns=max_turns
                )

                # Run multiple matches with color swapping
                for match_index in range(matches_per_pairing):
                    # Swap colors every other match
                    swap_colors = match_index % 2 == 1
                    
                    # Run the match
                    match_result = match_handler.run_match(swap_colors)
                    self.tournament_results.append(match_result)

                    # Progress tracking
                    current_match += 1
                    print(f"Progress: {current_match}/{total_matches} matches completed")

        # Save and analyze results
        self._save_results(results_dir)
        self._generate_visualizations(results_dir)

    def _save_results(self, results_dir: str):
        """
        Save tournament results to CSV.

        Args:
            results_dir: Directory to save results
        """
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.tournament_results)

        # Save full results
        full_results_path = os.path.join(results_dir, 'tournament_full_results.csv')
        results_df.to_csv(full_results_path, index=False)
        self.logger.info(f"Full results saved to {full_results_path}")

        # Generate summary statistics
        summary_stats = self._calculate_summary_stats(results_df)
        summary_path = os.path.join(results_dir, 'tournament_summary.csv')
        summary_stats.to_csv(summary_path, index=False)
        self.logger.info(f"Summary statistics saved to {summary_path}")

    def _calculate_summary_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive tournament summary statistics.

        Args:
            results_df: DataFrame with tournament results

        Returns:
            Summary statistics DataFrame
        """
        # Initialize summary DataFrame
        summary_stats = []

        # Unique player names
        players = set(results_df['player1'].unique()) | set(results_df['player2'].unique())

        for player in players:
            # Player's matches as player1
            player1_matches = results_df[results_df['player1'] == player]
            # Player's matches as player2
            player2_matches = results_df[results_df['player2'] == player]

            # Combine matches
            player_matches = pd.concat([
                player1_matches.assign(player_role='player1'),
                player2_matches.assign(player_role='player2')
            ])

            # Calculate statistics
            total_matches = len(player_matches)
            wins = len(player_matches[player_matches['winner'] == player])
            draws = len(player_matches[player_matches['winner'].isna()])
            losses = total_matches - wins - draws

            # Win rate calculation
            win_rate = wins / total_matches * 100 if total_matches > 0 else 0

            # Average points scored and conceded
            points_scored = (
                player1_matches['score_p1'].sum() + 
                player2_matches['score_p2'].sum()
            )
            points_conceded = (
                player1_matches['score_p2'].sum() + 
                player2_matches['score_p1'].sum()
            )

            # Average turns per match
            avg_turns = player_matches['turns'].mean()

            # Move type distribution
            move_types = {}
            for role in ['player1', 'player2']:
                move_type_cols = [
                    f"{role}_move_types" if role == 'player1' else f"{role}_move_types"
                ]
                for move_type_col in move_type_cols:
                    move_type_data = player_matches[move_type_col].apply(pd.Series)
                    for move_type, count in move_type_data.sum().items():
                        if move_type not in move_types:
                            move_types[move_type] = 0
                        move_types[move_type] += count

            # Normalize move type distribution to percentages
            total_moves = sum(move_types.values())
            move_type_dist = {
                k: v / total_moves * 100 for k, v in move_types.items()
            } if total_moves > 0 else {}

            # Compile player summary
            player_summary = {
                'player': player,
                'total_matches': total_matches,
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': win_rate,
                'points_scored': points_scored,
                'points_conceded': points_conceded,
                'point_differential': points_scored - points_conceded,
                'avg_turns': avg_turns,
                **{f'move_type_{k}': v for k, v in move_type_dist.items()}
            }

            summary_stats.append(player_summary)

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_stats)

        # Sort by win rate (descending)
        summary_df = summary_df.sort_values('win_rate', ascending=False)

        return summary_df

    def _generate_visualizations(self, results_dir: str):
        """
        Generate tournament visualizations.

        Args:
            results_dir: Directory to save visualizations
        """
        # Ensure matplotlib doesn't use GUI backend
        plt.switch_backend('Agg')

        # Load results and summary
        results_df = pd.DataFrame(self.tournament_results)
        summary_df = pd.read_csv(os.path.join(results_dir, 'tournament_summary.csv'))

        # 1. Win Rate Bar Chart
        plt.figure(figsize=(12, 6))
        plt.bar(summary_df['player'], summary_df['win_rate'])
        plt.title('Player Win Rates')
        plt.xlabel('Player')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'win_rates.png'))
        plt.close()

        # 2. Move Type Distribution Heatmap
        move_type_cols = [col for col in summary_df.columns if col.startswith('move_type_')]
        if move_type_cols:
            plt.figure(figsize=(12, 8))
            move_type_data = summary_df[move_type_cols].set_index(summary_df['player'])
            move_type_data.columns = [col.replace('move_type_', '') for col in move_type_data.columns]
            
            sns.heatmap(move_type_data, annot=True, cmap='YlGnBu', fmt='.1f')
            plt.title('Move Type Distribution (%)')
            plt.xlabel('Move Type')
            plt.ylabel('Player')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'move_type_distribution.png'))
            plt.close()

        # 3. Pairwise Performance Heatmap
        players = summary_df['player'].tolist()
        pairwise_matrix = np.zeros((len(players), len(players)))

        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if player1 == player2:
                    continue
                
                # Calculate win rate when player1 plays against player2
                matches = results_df[
                    ((results_df['player1'] == player1) & (results_df['player2'] == player2)) |
                    ((results_df['player2'] == player1) & (results_df['player1'] == player2))
                ]
                
                wins = matches[matches['winner'] == player1].shape[0]
                total_matches = matches.shape[0]
                
                win_rate = wins / total_matches * 100 if total_matches > 0 else 0
                pairwise_matrix[i, j] = win_rate

        plt.figure(figsize=(10, 8))
        sns.heatmap(pairwise_matrix, annot=True, cmap='RdYlGn', xticklabels=players, yticklabels=players)
        plt.title('Pairwise Win Rates')
        plt.xlabel('Opponent')
        plt.ylabel('Player')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'pairwise_performance.png'))
        plt.close()

        self.logger.info(f"Visualizations saved in {results_dir}")

def main():
    """Main function to run the tournament."""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Kulibrat AI Tournament")
    parser.add_argument(
        '--config', 
        type=str, 
        default='tournament_config.yaml',
        help='Path to tournament configuration YAML file'
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Run tournament
    try:
        tournament = TournamentEvaluator(args.config)
        tournament.run_tournament()
    except Exception as e:
        logging.error(f"Tournament failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()