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
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.gridspec as gridspec
from tqdm import tqdm

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

        # Match result tracking - initialize move_types as dictionaries
        self.match_results = {
            'player1': player1_name,
            'player2': player2_name,
            'winner': None,
            'score_p1': 0,
            'score_p2': 0,
            'turns': 0,
            'total_time': 0,
            'player1_move_types': {m.name: 0 for m in MoveType},
            'player2_move_types': {m.name: 0 for m in MoveType},
            'player1_color': None,
            'player2_color': None
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
            self.match_results['player1_color'] = PlayerColor.BLACK.name
            self.match_results['player2_color'] = PlayerColor.RED.name
        else:
            black_player = self.player2
            red_player = self.player1
            black_player_name = self.player2_name
            red_player_name = self.player1_name
            self.match_results['player1_color'] = PlayerColor.RED.name
            self.match_results['player2_color'] = PlayerColor.BLACK.name

        # Set player colors
        black_player.color = PlayerColor.BLACK
        red_player.color = PlayerColor.RED

        # Initialize game state
        game_state = GameState(target_score=self.target_score)

        # Track match progress
        turns = 0
        start_time = datetime.now()
        
        # Record all moves for detailed analysis
        all_moves = []

        while not game_state.is_game_over():# and turns < self.max_turns
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
                # Track move types - ensure we're working with a dictionary
                move_types_key = 'player1_move_types' if current_player_name == self.player1_name else 'player2_move_types'
                move_types_dict = self.match_results[move_types_key]
                
                # Explicitly check that we have a dictionary
                if not isinstance(move_types_dict, dict):
                    logging.warning(f"Expected dictionary for {move_types_key}, found {type(move_types_dict)}. Reinitializing.")
                    move_types_dict = {m.name: 0 for m in MoveType}
                    self.match_results[move_types_key] = move_types_dict
                
                # Now safely update the move type count
                move_type_name = move.move_type.name
                if move_type_name in move_types_dict:
                    move_types_dict[move_type_name] += 1
                else:
                    move_types_dict[move_type_name] = 1
                
                # Record the move details
                move_record = {
                    'turn': turns,
                    'player': current_player_name,
                    'color': current_color.name,
                    'move_type': move.move_type.name,
                    'start_pos': move.start_pos,
                    'end_pos': move.end_pos
                }
                all_moves.append(move_record)

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
            'total_time': match_duration,
            'all_moves': all_moves,
            'final_state': {
                'board': game_state.board.copy(),
                'scores': game_state.scores.copy()
            }
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
        
        # Colors for visualizations
        self.player_colors = {}  # Will be set dynamically
        self.cmap_base = 'viridis'

    def run_tournament(self):
        """Execute the full tournament."""
        # Extract tournament settings
        matches_per_pairing = self.config['tournament'].get('matches_per_pairing', 3)
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

        # Set up color scheme for visualizations
        self._initialize_color_scheme(list(players.keys()))

        # Run tournament
        player_names = list(players.keys())
        
        # Calculate total matches: each player plays against every other player in both colors, n matches per configuration
        total_matches = len(player_names) * (len(player_names) - 1) * matches_per_pairing * 2

        self.logger.info(f"Starting tournament with {len(players)} players")
        self.logger.info(f"Total matches: {total_matches}")

        # Track progress
        current_match = 0
        matchup_results = defaultdict(list)  # For tracking head-to-head results

        

        progress = tqdm(total=total_matches)

        try:
            # Run matches between all player pairs, ensuring each player plays both colors
            for i, player1_name in enumerate(player_names):
                for j, player2_name in enumerate(player_names):
                    if i == j:  # Skip matches with same player
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
                            try:
                                match_result = match_handler.run_match(swap_colors)
                                result_copy = match_result.copy()
                                
                                # Ensure move_types are properly copied as dictionaries
                                if 'player1_move_types' in result_copy:
                                    if isinstance(result_copy['player1_move_types'], dict):
                                        result_copy['player1_move_types'] = dict(result_copy['player1_move_types'])
                                    else:
                                        result_copy['player1_move_types'] = {m.name: 0 for m in MoveType}
                                        
                                if 'player2_move_types' in result_copy:
                                    if isinstance(result_copy['player2_move_types'], dict):
                                        result_copy['player2_move_types'] = dict(result_copy['player2_move_types'])
                                    else:
                                        result_copy['player2_move_types'] = {m.name: 0 for m in MoveType}
                                
                                self.tournament_results.append(result_copy)
                                
                                # Track head-to-head results
                                matchup_key = (player1_name, player2_name)
                                matchup_results[matchup_key].append(result_copy)
                            except Exception as e:
                                self.logger.error(f"Error in match {player1_name} vs {player2_name}: {e}")
                                # Don't abort the whole tournament for one failed match
                                traceback.print_exc()
                                continue

                            # Progress tracking
                            current_match += 1
                            progress_percent = (current_match / total_matches) * 100
                            print(f"Progress: {current_match}/{total_matches} matches completed ({progress_percent:.1f}%)")
                            
                            progress.update(1)

            # Save and analyze results
            self._save_interim_results(results_dir, current_match)
            self._save_results(results_dir)
            self._analyze_matchups(matchup_results, results_dir)
            self._generate_visualizations(results_dir)
            
        except Exception as e:
            self.logger.error(f"Tournament failed: {e}")
            traceback.print_exc()
            # Save what we have so far
            try:
                self._save_results(results_dir + "_incomplete")
            except:
                pass
            raise

    def _save_interim_results(self, results_dir: str, match_number: int):
        """
        Save interim tournament results to visualize progress.
        
        Args:
            results_dir: Directory to save results
            match_number: Current match number
        """
        # Only proceed if we have some results
        if not self.tournament_results:
            return
            
        # Convert current results to DataFrame
        interim_results_df = pd.DataFrame([
            {k: v for k, v in result.items() if k not in ['all_moves', 'final_state']}
            for result in self.tournament_results
        ])
        
        # Create interim summary
        interim_summary = self._calculate_summary_stats(interim_results_df)
        
        # Save current progress plot
        plt.figure(figsize=(12, 8))
        
        # Plot win rates so far
        win_rates = interim_summary.sort_values('win_rate', ascending=False)
        sns.barplot(
            x='player', 
            y='win_rate', 
            data=win_rates,
            palette=[self.player_colors.get(player, 'gray') for player in win_rates['player']]
        )
        plt.title(f'Interim Win Rates (After {match_number} Matches)')
        plt.xlabel('Player')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(results_dir, f'interim_win_rates_{match_number}.png'))
        plt.close()

    def _initialize_color_scheme(self, player_names: List[str]):
        """
        Initialize a color scheme for consistent visualization.
        
        Args:
            player_names: List of player names
        """
        # Create a color palette
        num_players = len(player_names)
        cmap = plt.get_cmap(self.cmap_base)
        colors = [cmap(i/num_players) for i in range(num_players)]
        
        # Assign colors to players
        self.player_colors = {
            player: f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            for player, (r, g, b, _) in zip(player_names, colors)
        }

    def _save_results(self, results_dir: str):
        """
        Save tournament results to CSV.

        Args:
            results_dir: Directory to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Convert results to DataFrame, exclude complex objects
        results_df = pd.DataFrame([
            {k: v for k, v in result.items() if k not in ['all_moves', 'final_state']}
            for result in self.tournament_results
        ])

        # Save full results
        full_results_path = os.path.join(results_dir, 'tournament_full_results.csv')
        results_df.to_csv(full_results_path, index=False)
        self.logger.info(f"Full results saved to {full_results_path}")

        # Generate summary statistics
        summary_stats = self._calculate_summary_stats(results_df)
        summary_path = os.path.join(results_dir, 'tournament_summary.csv')
        summary_stats.to_csv(summary_path, index=False)
        self.logger.info(f"Summary statistics saved to {summary_path}")
        
        # Save color matching for future reference
        with open(os.path.join(results_dir, 'player_colors.yaml'), 'w') as f:
            yaml.dump(self.player_colors, f)

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
            
            # Process move types carefully with type checking
            for role, role_col in [('player1', 'player1_move_types'), ('player2', 'player2_move_types')]:
                # Filter matches where this player has this role
                role_matches = player_matches[player_matches['player_role'] == role]
                
                for idx, row in role_matches.iterrows():
                    if role_col in row and isinstance(row[role_col], dict):
                        for move_type, count in row[role_col].items():
                            if move_type not in move_types:
                                move_types[move_type] = 0
                            move_types[move_type] += count

            # Normalize move type distribution to percentages
            total_moves = sum(move_types.values())
            move_type_dist = {
                k: v / total_moves * 100 for k, v in move_types.items()
            } if total_moves > 0 else {}
            
            # Performance by color
            black_matches = player_matches[
                ((player_matches['player1'] == player) & (player_matches['player1_color'] == 'BLACK')) |
                ((player_matches['player2'] == player) & (player_matches['player2_color'] == 'BLACK'))
            ]
            red_matches = player_matches[
                ((player_matches['player1'] == player) & (player_matches['player1_color'] == 'RED')) |
                ((player_matches['player2'] == player) & (player_matches['player2_color'] == 'RED'))
            ]
            
            # Wins by color
            black_wins = len(black_matches[black_matches['winner'] == player])
            black_total = len(black_matches) if len(black_matches) > 0 else 1
            red_wins = len(red_matches[red_matches['winner'] == player])
            red_total = len(red_matches) if len(red_matches) > 0 else 1
            
            black_win_rate = black_wins / black_total * 100 if black_total > 0 else 0
            red_win_rate = red_wins / red_total * 100 if red_total > 0 else 0

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
                'black_win_rate': black_win_rate,
                'red_win_rate': red_win_rate,
                'black_matches': black_total,
                'red_matches': red_total,
                **{f'move_type_{k}': v for k, v in move_type_dist.items()}
            }

            summary_stats.append(player_summary)

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_stats)

        # Sort by win rate (descending)
        if not summary_df.empty and 'win_rate' in summary_df.columns:
            summary_df = summary_df.sort_values('win_rate', ascending=False)

        return summary_df
    
    def _analyze_matchups(self, matchup_results: Dict[tuple, List[Dict]], results_dir: str):
        """
        Analyze head-to-head matchups and save detailed matchup statistics.
        
        Args:
            matchup_results: Dictionary of matchup results
            results_dir: Directory to save results
        """
        # Create a directory for matchup analysis
        matchup_dir = os.path.join(results_dir, 'matchups')
        os.makedirs(matchup_dir, exist_ok=True)
        
        # Prepare a DataFrame for all matchups
        matchup_stats = []
        
        # Analyze each matchup
        for (player1, player2), matches in matchup_results.items():
            # Skip invalid matchups
            if not matches:
                continue
                
            # Create a DataFrame for this matchup with basic stats only
            matchup_df = pd.DataFrame([
                {k: v for k, v in match.items() if k not in ['all_moves', 'final_state', 'player1_move_types', 'player2_move_types']}
                for match in matches
            ])
            
            # Calculate statistics
            total_matches = len(matchup_df)
            player1_wins = len(matchup_df[matchup_df['winner'] == player1])
            player2_wins = len(matchup_df[matchup_df['winner'] == player2])
            draws = total_matches - player1_wins - player2_wins
            
            player1_win_rate = player1_wins / total_matches * 100 if total_matches > 0 else 0
            player2_win_rate = player2_wins / total_matches * 100 if total_matches > 0 else 0
            
            # Points scored
            player1_points = matchup_df['score_p1'].sum()
            player2_points = matchup_df['score_p2'].sum()
            
            # Average turns
            avg_turns = matchup_df['turns'].mean()
            
            # Performance by color for player1
            player1_black_matches = matchup_df[matchup_df['player1_color'] == 'BLACK']
            player1_red_matches = matchup_df[matchup_df['player1_color'] == 'RED']
            
            player1_black_wins = len(player1_black_matches[player1_black_matches['winner'] == player1])
            player1_red_wins = len(player1_red_matches[player1_red_matches['winner'] == player1])
            
            player1_black_win_rate = player1_black_wins / len(player1_black_matches) * 100 if len(player1_black_matches) > 0 else 0
            player1_red_win_rate = player1_red_wins / len(player1_red_matches) * 100 if len(player1_red_matches) > 0 else 0
            
            # Process move types safely
            try:
                # Aggregate move types for both players in this matchup
                move_types_p1 = defaultdict(int)
                move_types_p2 = defaultdict(int)
                
                for match in matches:
                    # Process player1's move types
                    if 'player1_move_types' in match and match.get('player1') == player1 and isinstance(match['player1_move_types'], dict):
                        for move_type, count in match['player1_move_types'].items():
                            move_types_p1[move_type] += count
                    elif 'player2_move_types' in match and match.get('player2') == player1 and isinstance(match['player2_move_types'], dict):
                        for move_type, count in match['player2_move_types'].items():
                            move_types_p1[move_type] += count
                            
                    # Process player2's move types
                    if 'player1_move_types' in match and match.get('player1') == player2 and isinstance(match['player1_move_types'], dict):
                        for move_type, count in match['player1_move_types'].items():
                            move_types_p2[move_type] += count
                    elif 'player2_move_types' in match and match.get('player2') == player2 and isinstance(match['player2_move_types'], dict):
                        for move_type, count in match['player2_move_types'].items():
                            move_types_p2[move_type] += count
            except Exception as e:
                self.logger.error(f"Error processing move types for {player1} vs {player2}: {e}")
                move_types_p1 = defaultdict(int)
                move_types_p2 = defaultdict(int)
            
            # Collect statistics for this matchup
            matchup_stat = {
                'player1': player1,
                'player2': player2,
                'total_matches': total_matches,
                'player1_wins': player1_wins,
                'player2_wins': player2_wins,
                'draws': draws,
                'player1_win_rate': player1_win_rate,
                'player2_win_rate': player2_win_rate,
                'player1_points': player1_points,
                'player2_points': player2_points,
                'avg_turns': avg_turns,
                'player1_black_win_rate': player1_black_win_rate,
                'player1_red_win_rate': player1_red_win_rate,
                'player1_move_types': dict(move_types_p1),
                'player2_move_types': dict(move_types_p2)
            }
            
            matchup_stats.append(matchup_stat)
            
            # Create a visualization for this matchup
            try:
                self._create_matchup_visualization(player1, player2, matchup_stat, move_types_p1, move_types_p2, matchup_dir)
            except Exception as e:
                self.logger.error(f"Error creating visualization for {player1} vs {player2}: {e}")
        
        # Save matchup statistics
        try:
            # Convert to simpler format for CSV
            simplified_matchup_stats = []
            for stat in matchup_stats:
                simple_stat = {k: v for k, v in stat.items() if k not in ['player1_move_types', 'player2_move_types']}
                simplified_matchup_stats.append(simple_stat)
                
            matchup_df = pd.DataFrame(simplified_matchup_stats)
            matchup_df.to_csv(os.path.join(results_dir, 'matchup_statistics.csv'), index=False)
        except Exception as e:
            self.logger.error(f"Error saving matchup statistics: {e}")
    
    def _create_matchup_visualization(self, player1, player2, matchup_stat, move_types_p1, move_types_p2, matchup_dir):
        """
        Create visualization for a specific matchup.
        
        Args:
            player1: First player name
            player2: Second player name
            matchup_stat: Matchup statistics
            move_types_p1: Move types for player1
            move_types_p2: Move types for player2
            matchup_dir: Directory to save visualization
        """
        plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # 1. Win distribution
        ax1 = plt.subplot(gs[0, 0])
        win_data = [matchup_stat['player1_wins'], matchup_stat['player2_wins'], matchup_stat['draws']]
        labels = [f'{player1} wins', f'{player2} wins', 'Draws']
        colors = [self.player_colors.get(player1, 'blue'), 
                  self.player_colors.get(player2, 'orange'), 
                  'gray']
        ax1.pie(win_data, labels=labels, autopct='%1.1f%%', colors=colors)
        ax1.set_title(f'Match Outcomes: {player1} vs {player2}')
        
        # 2. Performance by color
        ax2 = plt.subplot(gs[0, 1])
        color_data = {
            f'{player1} as BLACK': matchup_stat['player1_black_win_rate'],
            f'{player1} as RED': matchup_stat['player1_red_win_rate'],
            f'{player2} as BLACK': 100 - matchup_stat['player1_red_win_rate'],
            f'{player2} as RED': 100 - matchup_stat['player1_black_win_rate']
        }
        bars = ax2.bar(color_data.keys(), color_data.values())
        for i, bar in enumerate(bars):
            if i < 2:  # First player
                bar.set_color(self.player_colors.get(player1, 'blue'))
            else:  # Second player
                bar.set_color(self.player_colors.get(player2, 'orange'))
        ax2.set_title('Win Rate by Color')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Points scored
        ax3 = plt.subplot(gs[1, 0])
        points_data = {
            player1: matchup_stat['player1_points'] / matchup_stat['total_matches'],
            player2: matchup_stat['player2_points'] / matchup_stat['total_matches']
        }
        bars = ax3.bar(points_data.keys(), points_data.values())
        bars[0].set_color(self.player_colors.get(player1, 'blue'))
        bars[1].set_color(self.player_colors.get(player2, 'orange'))
        ax3.set_title('Average Points Scored per Match')
        ax3.set_ylabel('Points')
        
        # 4. Move types distribution
        ax4 = plt.subplot(gs[1, 1])
        
        # Calculate percentages
        total_moves_p1 = sum(move_types_p1.values()) or 1  # Avoid division by zero
        total_moves_p2 = sum(move_types_p2.values()) or 1
        
        move_types_p1_pct = {k: v / total_moves_p1 * 100 for k, v in move_types_p1.items()}
        move_types_p2_pct = {k: v / total_moves_p2 * 100 for k, v in move_types_p2.items()}
        
        # Filter move types that exist in either player's moves
        all_move_types = set(move_types_p1.keys()) | set(move_types_p2.keys())
        move_types = sorted(all_move_types)
        
        # Prepare data
        p1_values = [move_types_p1_pct.get(move, 0) for move in move_types]
        p2_values = [move_types_p2_pct.get(move, 0) for move in move_types]
        
        x = np.arange(len(move_types))
        width = 0.35
        
        ax4.bar(x - width/2, p1_values, width, label=player1, color=self.player_colors.get(player1, 'blue'))
        ax4.bar(x + width/2, p2_values, width, label=player2, color=self.player_colors.get(player2, 'orange'))
        
        ax4.set_title('Move Type Distribution')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(move_types, rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(matchup_dir, f'{player1}_vs_{player2}.png'))
        plt.close()

    def _generate_visualizations(self, results_dir: str):
        """
        Generate tournament visualizations.

        Args:
            results_dir: Directory to save visualizations
        """
        # Ensure matplotlib doesn't use GUI backend
        plt.switch_backend('Agg')

        # Load results and summary
        try:
            # Create simplified results DataFrame without complex objects
            results_df = pd.DataFrame([
                {k: v for k, v in result.items() if k not in ['all_moves', 'final_state', 'player1_move_types', 'player2_move_types']}
                for result in self.tournament_results
            ])
            
            # Load summary
            summary_path = os.path.join(results_dir, 'tournament_summary.csv')
            if os.path.exists(summary_path):
                summary_df = pd.read_csv(summary_path)
            else:
                # Generate summary if file doesn't exist
                summary_df = self._calculate_summary_stats(results_df)
                summary_df.to_csv(summary_path, index=False)
                
        except Exception as e:
            self.logger.error(f"Error loading results for visualization: {e}")
            # Proceed with what we have directly from self.tournament_results
            results_df = pd.DataFrame([
                {k: v for k, v in result.items() if k not in ['all_moves', 'final_state', 'player1_move_types', 'player2_move_types']}
                for result in self.tournament_results
            ])
            summary_df = self._calculate_summary_stats(results_df)

        # Create a directory for visualizations
        viz_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Generate individual visualizations with error handling
        try:
            # 1. Enhanced Win Rate Chart with Wins/Losses/Draws
            self._generate_win_rate_chart(summary_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating win rate chart: {e}")
            
        try:
            # 2. Performance by Color (BLACK vs RED)
            self._generate_color_performance_chart(summary_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating color performance chart: {e}")
            
        try:
            # 3. Enhanced Move Type Distribution Heatmap
            self._generate_move_type_heatmap(summary_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating move type heatmap: {e}")
            
        try:
            # 4. Enhanced Pairwise Performance Heatmap
            self._generate_pairwise_heatmap(results_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating pairwise heatmap: {e}")
            
        try:
            # 5. Points Scored vs Conceded Scatter Plot
            self._generate_points_scatter(summary_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating points scatter: {e}")
            
        try:
            # 6. Game Length Distribution
            self._generate_game_length_distribution(results_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating game length distribution: {e}")
            
        try:
            # 7. Tournament Overview Dashboard
            self._generate_tournament_dashboard(summary_df, results_df, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating tournament dashboard: {e}")

        self.logger.info(f"Visualizations saved in {viz_dir}")

    # [Remaining visualization methods unchanged]
    # Let's include just a sample visualization method to validate the changes:
    
    def _generate_win_rate_chart(self, summary_df: pd.DataFrame, viz_dir: str):
        """
        Generate enhanced win rate chart with wins/losses/draws breakdown.
        
        Args:
            summary_df: Summary statistics DataFrame
            viz_dir: Directory to save visualizations
        """
        # Skip if empty or missing data
        if summary_df.empty or 'win_rate' not in summary_df.columns:
            self.logger.warning("Missing data for win rate chart")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Sort players by win rate
        sorted_df = summary_df.sort_values('win_rate', ascending=False)
        players = sorted_df['player'].tolist()
        
        # Set up positions for the bars
        x = np.arange(len(players))
        width = 0.25
        
        # Create stacked bar chart for match outcomes
        ax1 = plt.subplot(111)
        
        # Plot wins, losses, and draws as stacked bars
        wins = sorted_df['wins'].fillna(0).astype(int)
        losses = sorted_df['losses'].fillna(0).astype(int)
        draws = sorted_df['draws'].fillna(0).astype(int)
        
        ax1.bar(x, wins, width, label='Wins', color='forestgreen')
        ax1.bar(x, losses, width, bottom=wins, label='Losses', color='firebrick')
        ax1.bar(x, draws, width, bottom=wins + losses, label='Draws', color='goldenrod')
        
        # Add total matches as a line
        ax2 = ax1.twinx()
        ax2.plot(x, sorted_df['win_rate'], 'o-', color='blue', linewidth=2, markersize=8, label='Win Rate (%)')
        
        # Enhance appearance
        ax1.set_xlabel('Player', fontsize=12)
        ax1.set_ylabel('Number of Matches', fontsize=12)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        
        ax1.set_title('Player Performance Overview', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(players, rotation=45, ha='right')
        
        # Set up grid
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Display both legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Add value labels on the bars
        for i, (w, l, d) in enumerate(zip(wins, losses, draws)):
            win_rate = sorted_df['win_rate'].iloc[i]
            # Win rate label
            ax2.annotate(f'{win_rate:.1f}%', 
                        xy=(i, win_rate),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontweight='bold')
            
            # Stack counts
            total = w + l + d
            ax1.annotate(f'{w}', 
                        xy=(i, w/2),
                        ha='center',
                        va='center',
                        color='white',
                        fontweight='bold')
                        
            ax1.annotate(f'{l}', 
                        xy=(i, w + l/2),
                        ha='center',
                        va='center',
                        color='white',
                        fontweight='bold')
                        
            ax1.annotate(f'{d}', 
                        xy=(i, w + l + d/2),
                        ha='center',
                        va='center',
                        color='black',
                        fontweight='bold')
                        
            # Total matches at the top
            ax1.annotate(f'Total: {total}', 
                        xy=(i, total),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center',
                        va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'player_performance.png'), dpi=300)
        plt.close()

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
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()