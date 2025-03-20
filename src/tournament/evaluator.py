"""
Simplified tournament evaluation core functionality.
"""

import logging
import os
import traceback
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.tournament.factory import AIPlayerFactory
from src.tournament.match import TournamentMatch
from src.core.move_type import MoveType

class TournamentEvaluator:
    """Manages tournament execution with minimal complexity."""

    def __init__(self, config_path: str):
        """
        Initialize tournament evaluator.

        Args:
            config_path: Path to tournament configuration YAML file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler("tournament.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.logger.info(f"Successfully loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
            
        # Initialize results storage
        self.tournament_results = []

    def run_tournament(self):
        """Execute the tournament with robust error handling."""
        # Extract core tournament settings
        matches_per_pairing = self.config['tournament'].get('matches_per_pairing', 3)
        target_score = self.config['tournament'].get('target_score', 5)
        max_turns = self.config['tournament'].get('max_turns', 300)

        # Create simple output directory
        output_dir = self.config['output'].get('results_dir', 'tournament_results')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_dir, f"tournament_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # Create players
        players = {}
        self.logger.info("Creating players...")
        for player_config in self.config['players']:
            self.logger.info(f"Creating player: {player_config['name']} (type: {player_config['type']})")
            player = AIPlayerFactory.create_player(player_config)
            if player:
                players[player_config['name']] = player
                self.logger.info(f"Successfully created player: {player_config['name']}")
            else:
                self.logger.error(f"Failed to create player: {player_config['name']}")

        if len(players) < 2:
            self.logger.error("Not enough players (minimum 2) to run tournament")
            return

        # Run tournament
        player_names = list(players.keys())
        
        # Calculate total matches
        total_matches = len(player_names) * (len(player_names) - 1) * matches_per_pairing * 2
        self.logger.info(f"Starting tournament with {len(players)} players, total matches: {total_matches}")

        # Track progress
        current_match = 0
        progress = tqdm(total=total_matches, desc="Running Tournament")

        try:
            # Run matches between all player pairs
            for i, player1_name in enumerate(player_names):
                for j, player2_name in enumerate(player_names):
                    if i >= j:  # Skip matches with same player or duplicates
                        continue

                    player1 = players[player1_name]
                    player2 = players[player2_name]

                    # Play matches with both color configurations
                    for color_config in range(2):  # 0: player1=BLACK, 1: player1=RED
                        swap_colors = (color_config == 1)
                        
                        # Create match handler
                        match_handler = TournamentMatch(
                            player1_name, 
                            player1, 
                            player2_name, 
                            player2,
                            target_score=target_score,
                            max_turns=max_turns
                        )

                        # Run multiple matches with this color configuration
                        for match_index in range(matches_per_pairing):
                            # Run the match
                            self.logger.info(f"Starting match {current_match+1}/{total_matches}: {player1_name} vs {player2_name} (colors swapped: {swap_colors})")
                            try:
                                match_result = match_handler.run_match(swap_colors)
                                
                                # Ensure move_types are properly copied as dictionaries
                                self._fix_move_types(match_result)
                                
                                # Store and log result
                                self.tournament_results.append(match_result)
                                winner = match_result.get('winner', 'DRAW')
                                score_p1 = match_result.get('score_p1', 0)
                                score_p2 = match_result.get('score_p2', 0)
                                turns = match_result.get('turns', 0)
                                self.logger.info(f"Match completed: Winner={winner}, Score={score_p1}-{score_p2}, Turns={turns}")
                                
                            except Exception as e:
                                self.logger.error(f"Error in match {player1_name} vs {player2_name}: {e}")
                                traceback.print_exc()
                                continue

                            # Progress tracking
                            current_match += 1
                            progress.update(1)

            # Save results
            self._save_results(results_dir)
            
            # Generate the results matrix visualization
            if self.tournament_results:
                self._create_results_matrix(results_dir)
                
            self.logger.info(f"Tournament completed successfully. Results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Tournament failed: {e}")
            traceback.print_exc()
            # Save whatever results we have
            try:
                self._save_results(results_dir + "_incomplete")
                self.logger.info(f"Partial results saved to {results_dir}_incomplete")
            except Exception as save_error:
                self.logger.error(f"Failed to save partial results: {save_error}")
        finally:
            progress.close()

    def _fix_move_types(self, match_result):
        """
        Ensure move_types are properly stored as dictionaries.

        Args:
            match_result: Match result dictionary to fix
        """
        # Initialize move type dictionaries if missing or invalid
        for player_key in ['player1_move_types', 'player2_move_types']:
            if player_key not in match_result or not isinstance(match_result[player_key], dict):
                match_result[player_key] = {m.name: 0 for m in MoveType}
            else:
                # Convert to a plain dict if it's another type of dict-like object
                match_result[player_key] = dict(match_result[player_key])

    def _save_results(self, results_dir: str):
        """
        Save tournament results to CSV.

        Args:
            results_dir: Directory to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Skip if no results
        if not self.tournament_results:
            self.logger.warning("No results to save")
            return
            
        try:
            # Convert results to DataFrame, exclude complex objects
            results_df = pd.DataFrame([
                {k: v for k, v in result.items() if k not in ['all_moves', 'final_state']}
                for result in self.tournament_results
            ])

            # Save full results
            full_results_path = os.path.join(results_dir, 'tournament_full_results.csv')
            results_df.to_csv(full_results_path, index=False)
            self.logger.info(f"Full results saved to {full_results_path}")

            # Generate simple summary statistics
            summary_stats = self._calculate_summary_stats(results_df)
            summary_path = os.path.join(results_dir, 'tournament_summary.csv')
            summary_stats.to_csv(summary_path, index=False)
            self.logger.info(f"Summary statistics saved to {summary_path}")
            
            # Save raw data as YAML for debugging
            raw_results_path = os.path.join(results_dir, 'raw_results.yaml')
            with open(raw_results_path, 'w') as f:
                yaml.dump(self.tournament_results, f)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            traceback.print_exc()

    def _calculate_summary_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic tournament summary statistics.

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
            try:
                # Player's matches as player1
                player1_matches = results_df[results_df['player1'] == player]
                # Player's matches as player2
                player2_matches = results_df[results_df['player2'] == player]

                # Combine matches
                player_matches = pd.concat([
                    player1_matches.assign(player_role='player1'),
                    player2_matches.assign(player_role='player2')
                ])

                # Calculate core statistics
                total_matches = len(player_matches)
                wins = len(player_matches[player_matches['winner'] == player])
                draws = len(player_matches[player_matches['winner'].isna()])
                losses = total_matches - wins - draws

                # Win rate calculation
                win_rate = wins / total_matches * 100 if total_matches > 0 else 0

                # Points scored
                points_scored = (
                    player1_matches['score_p1'].sum() + 
                    player2_matches['score_p2'].sum()
                )
                points_conceded = (
                    player1_matches['score_p2'].sum() + 
                    player2_matches['score_p1'].sum()
                )

                # Create player summary with just the essentials
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
                }

                summary_stats.append(player_summary)
                
            except Exception as e:
                self.logger.error(f"Error calculating stats for player {player}: {e}")
                # Add minimal player entry with error noted
                summary_stats.append({
                    'player': player,
                    'total_matches': 0,
                    'wins': 0,
                    'win_rate': 0,
                    'error': str(e)
                })

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_stats)

        # Sort by win rate (descending) if possible
        if not summary_df.empty and 'win_rate' in summary_df.columns:
            summary_df = summary_df.sort_values('win_rate', ascending=False)

        return summary_df
        
    def _create_results_matrix(self, results_dir: str):
        """
        Create a matrix visualization showing match results with RED players as rows
        and BLACK players as columns.
        
        Args:
            results_dir: Directory to save the visualization
        """
        self.logger.info("Generating results matrix visualization...")
        
        try:
            # Convert results to DataFrame if needed
            if isinstance(self.tournament_results, list) and not isinstance(self.tournament_results, pd.DataFrame):
                results_df = pd.DataFrame([
                    {k: v for k, v in result.items() if k not in ['all_moves', 'final_state']}
                    for result in self.tournament_results
                ])
            else:
                results_df = self.tournament_results
                
            # Get all unique player names
            all_players = sorted(list(set(results_df['player1'].unique()) | set(results_df['player2'].unique())))
            
            # Create empty results matrix for RED (rows) vs BLACK (columns)
            # Format: [wins, total_games]
            matrix_data = np.zeros((len(all_players), len(all_players), 2), dtype=int)
            
            # Fill the matrix with results
            for _, row in results_df.iterrows():
                # Extract players and colors
                player1_name = row['player1']
                player2_name = row['player2']
                player1_color = row['player1_color']
                
                # Determine who was RED and who was BLACK
                if player1_color == 'RED':
                    red_player, black_player = player1_name, player2_name
                    winner = row['winner']
                else:  # player1 is BLACK
                    red_player, black_player = player2_name, player1_name
                    winner = row['winner']
                
                # Find indices in the matrix
                red_idx = all_players.index(red_player)
                black_idx = all_players.index(black_player)
                
                # Update total games count
                matrix_data[red_idx, black_idx, 1] += 1
                
                # Update wins count if there was a winner
                if winner == red_player:
                    matrix_data[red_idx, black_idx, 0] += 1
            
            # Calculate win percentages for display
            win_pct_matrix = np.zeros((len(all_players), len(all_players)))
            for i in range(len(all_players)):
                for j in range(len(all_players)):
                    total = matrix_data[i, j, 1]
                    if total > 0:
                        win_pct_matrix[i, j] = (matrix_data[i, j, 0] / total) * 100
                    else:
                        win_pct_matrix[i, j] = np.nan  # No games played
            
            # Create the visualization
            plt.figure(figsize=(10, 8))
            
            # Create heatmap (masked diagonal to avoid self-play confusion)
            mask = np.eye(len(all_players), dtype=bool)
            plt.imshow(np.ma.array(win_pct_matrix, mask=mask), 
                      cmap='YlOrRd', interpolation='nearest', 
                      vmin=0, vmax=100)
            
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Win Rate (%)')
            
            # Set up axis labels
            plt.xticks(range(len(all_players)), all_players, rotation=45, ha='right')
            plt.yticks(range(len(all_players)), all_players)
            
            # Add title and axis labels
            plt.title('Tournament Results: RED (rows) vs BLACK (columns)', fontsize=14)
            plt.xlabel('Playing as BLACK', fontsize=12)
            plt.ylabel('Playing as RED', fontsize=12)
            
            # Add text annotations to cells
            for i in range(len(all_players)):
                for j in range(len(all_players)):
                    if i != j:  # Skip diagonal elements
                        wins = matrix_data[i, j, 0]
                        total = matrix_data[i, j, 1]
                        if total > 0:
                            text = f"{wins}/{total}\n({win_pct_matrix[i, j]:.1f}%)"
                            # Determine text color based on background
                            text_color = 'white' if win_pct_matrix[i, j] > 50 else 'black'
                            plt.text(j, i, text, ha='center', va='center', color=text_color)
                        else:
                            plt.text(j, i, "N/A", ha='center', va='center', color='gray')
            
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(results_dir, 'results_matrix.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Results matrix visualization saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating results matrix visualization: {e}")
            traceback.print_exc()