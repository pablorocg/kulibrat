#!/usr/bin/env python3
"""
Improved Tournament Runner for Kulibrat AI Evaluation.
This script addresses potential issues with color bias and ensures fair evaluation.
"""

import os
import sys
import time
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to be able to import from src directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.game_rules import GameRules
from src.core.turn_manager import TurnManager
from src.players.player_factory import PlayerFactory
from src.players.player import Player


class ImprovedTournament:
    """
    Improved tournament system with robust evaluation, logging, and color balancing.
    """
    
    def __init__(self, config_path, verbose=False):
        """Initialize the tournament with the given configuration."""
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler("improved_tournament.log"),
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
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("tournament_results", f"improved_tournament_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize results storage
        self.tournament_results = []
        
        # Player cache (to avoid recreating players)
        self.player_cache = {}
        
    def create_player(self, player_config, color):
        """Create a player with specified configuration and color."""
        player_key = f"{player_config['name']}_{color.name}"
        
        # Return cached player if available
        if player_key in self.player_cache:
            player = self.player_cache[player_key]
            player.color = color  # Update color
            return player
        
        # Create new player
        try:
            player_type = player_config['type']
            name = player_config.get('name', f'{player_type}-default')
            
            self.logger.info(f"Creating player '{name}' of type '{player_type}' with color {color.name}")
            
            # Initialize parameters from configuration
            params = {k: v for k, v in player_config.items() if k not in ['type', 'name']}
            
            # Create player using factory
            player = PlayerFactory.create_player(player_type, color, None, **params)
            
            # Cache the player
            self.player_cache[player_key] = player
            return player
            
        except Exception as e:
            self.logger.error(f"Error creating player {player_config.get('name', 'unknown')}: {e}")
            raise
    
    def run_match(self, player1_config, player2_config, black_player_idx, red_player_idx, match_index):
        """
        Run a single match between two players with specified colors.
        
        Args:
            player1_config: Configuration for first player
            player2_config: Configuration for second player
            black_player_idx: Index of player to be BLACK (0=player1, 1=player2)
            red_player_idx: Index of player to be RED (0=player1, 1=player2)
            match_index: Match index for logging
            
        Returns:
            Dictionary with match results
        """
        # Determine player colors
        if black_player_idx == 0:
            black_player_config = player1_config
            red_player_config = player2_config
            black_player_name = player1_config['name']
            red_player_name = player2_config['name']
        else:
            black_player_config = player2_config
            red_player_config = player1_config
            black_player_name = player2_config['name']
            red_player_name = player1_config['name']
            
        # Create players with their colors
        black_player = self.create_player(black_player_config, PlayerColor.BLACK)
        red_player = self.create_player(red_player_config, PlayerColor.RED)
        
        # Create game components
        target_score = self.config['tournament'].get('target_score', 5)
        max_turns = self.config['tournament'].get('max_turns', 300)
        game_state = GameState(target_score=target_score)
        rules_engine = GameRules()
        turn_manager = TurnManager(rules_engine)
        
        # Initialize players
        if hasattr(black_player, 'setup'):
            black_player.setup(game_state)
        if hasattr(red_player, 'setup'):
            red_player.setup(game_state)
            
        # Result tracking
        match_result = {
            "black_player": black_player_name,
            "red_player": red_player_name,
            "winner": None,
            "black_score": 0,
            "red_score": 0,
            "turns": 0,
            "total_time": 0,
            "black_move_types": defaultdict(int),
            "red_move_types": defaultdict(int),
            "match_index": match_index
        }
        
        # Log match start
        self.logger.info(f"Starting match {match_index}: {black_player_name} (BLACK) vs {red_player_name} (RED)")
        
        # Main game loop
        start_time = time.time()
        current_player_color = PlayerColor.BLACK
        turns = 0
        
        while not game_state.is_game_over() and turns < max_turns:
            # Get current player
            current_player = black_player if current_player_color == PlayerColor.BLACK else red_player
            
            try:
                # Get player move with timeout protection
                move_start = time.time()
                move = current_player.get_move(game_state)
                move_time = time.time() - move_start
                
                # Check for timeout
                if move_time > 10:  # 10 second timeout
                    self.logger.warning(f"Player {current_player.name} took {move_time:.2f}s for move, exceeding normal time")
                
                # Track move types
                if move:
                    if current_player_color == PlayerColor.BLACK:
                        match_result["black_move_types"][move.move_type.name] += 1
                    else:
                        match_result["red_move_types"][move.move_type.name] += 1
                
                # Process turn
                if move:
                    # Apply move
                    new_state = turn_manager.process_turn(game_state, current_player, move)
                    game_state = new_state
                    
                    # Notify players about the move
                    if hasattr(black_player, 'notify_move'):
                        black_player.notify_move(move, game_state)
                    if hasattr(red_player, 'notify_move'):
                        red_player.notify_move(move, game_state)
                
                # Update current player - very important to use the state's current player
                current_player_color = game_state.current_player
                turns += 1
                
            except Exception as e:
                self.logger.error(f"Error during move by {current_player.name}: {e}")
                import traceback
                traceback.print_exc()
                # On error, switch players to avoid getting stuck
                current_player_color = current_player_color.opposite()
                
        # Calculate match duration
        match_duration = time.time() - start_time
        
        # Determine winner and get final scores
        winner = game_state.get_winner()
        
        if winner == PlayerColor.BLACK:
            winner_name = black_player_name
        elif winner == PlayerColor.RED:
            winner_name = red_player_name
        else:
            winner_name = "DRAW"
            
        # Update match results
        match_result.update({
            "winner": winner_name,
            "black_score": game_state.scores[PlayerColor.BLACK],
            "red_score": game_state.scores[PlayerColor.RED],
            "turns": turns,
            "total_time": match_duration
        })
        
        # Notify players of game over
        if hasattr(black_player, 'game_over'):
            black_player.game_over(game_state)
        if hasattr(red_player, 'game_over'):
            red_player.game_over(game_state)
            
        # Log match result
        self.logger.info(f"Match {match_index} completed: Winner={winner_name}, " +
                        f"Score={match_result['black_score']}-{match_result['red_score']}, " +
                        f"Turns={turns}")
        
        return match_result
    
    def run_tournament(self):
        """Execute the full tournament with all player combinations."""
        # Extract tournament settings
        matches_per_pairing = self.config['tournament'].get('matches_per_pairing', 5)
        
        # Get player configurations
        player_configs = self.config['players']
        
        if len(player_configs) < 2:
            self.logger.error("Not enough players (minimum 2) to run tournament")
            return
            
        # Calculate total matches
        total_pairings = len(player_configs) * (len(player_configs) - 1) // 2
        total_matches = total_pairings * matches_per_pairing * 2  # Each pairing plays as both colors
        
        self.logger.info(f"Starting tournament with {len(player_configs)} players")
        self.logger.info(f"Total pairings: {total_pairings}, total matches: {total_matches}")
        
        # Progress tracking
        match_index = 0
        progress = tqdm(total=total_matches, desc="Running Tournament")
        
        try:
            # Run matches between all player pairs
            for i, player1_config in enumerate(player_configs):
                for j, player2_config in enumerate(player_configs):
                    # Skip self-play for now
                    if i >= j:
                        continue
                        
                    player1_name = player1_config['name']
                    player2_name = player2_config['name']
                    
                    self.logger.info(f"Starting matches between {player1_name} and {player2_name}")
                    
                    # Run matches alternating colors
                    for _ in range(matches_per_pairing):
                        # Player 1 as BLACK, Player 2 as RED
                        match_index += 1
                        match_result = self.run_match(
                            player1_config, player2_config, 
                            black_player_idx=0, red_player_idx=1,
                            match_index=match_index
                        )
                        self.tournament_results.append(match_result)
                        progress.update(1)
                        
                        # Player 1 as RED, Player 2 as BLACK
                        match_index += 1
                        match_result = self.run_match(
                            player1_config, player2_config, 
                            black_player_idx=1, red_player_idx=0,
                            match_index=match_index
                        )
                        self.tournament_results.append(match_result)
                        progress.update(1)
            
            # Save results
            self._save_results()
            
            # Generate visualizations
            self._create_visualizations()
            
        except Exception as e:
            self.logger.error(f"Tournament failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to save partial results
            self._save_results(suffix="_incomplete")
        finally:
            progress.close()
            
    def _save_results(self, suffix=""):
        """Save tournament results to CSV files."""
        if not self.tournament_results:
            self.logger.warning("No results to save")
            return
            
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.tournament_results)
            
            # Convert move type dictionaries to strings for CSV storage
            results_df['black_move_types'] = results_df['black_move_types'].apply(str)
            results_df['red_move_types'] = results_df['red_move_types'].apply(str)
            
            # Save full results
            results_path = os.path.join(self.results_dir, f'tournament_full_results{suffix}.csv')
            results_df.to_csv(results_path, index=False)
            self.logger.info(f"Full results saved to {results_path}")
            
            # Generate summary statistics
            summary_stats = self._calculate_summary_stats(results_df)
            summary_path = os.path.join(self.results_dir, f'tournament_summary{suffix}.csv')
            summary_stats.to_csv(summary_path, index=False)
            self.logger.info(f"Summary statistics saved to {summary_path}")
            
            # Save configuration for reference
            config_path = os.path.join(self.results_dir, f'tournament_config{suffix}.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
                
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()
            
    def _calculate_summary_stats(self, results_df):
        """Calculate tournament summary statistics by player."""
        # Get unique player names
        all_players = list(set(results_df['black_player'].unique()) | 
                          set(results_df['red_player'].unique()))
        
        # Initialize stats
        player_stats = []
        
        for player in all_players:
            # Matches as BLACK
            black_matches = results_df[results_df['black_player'] == player]
            black_wins = len(black_matches[black_matches['winner'] == player])
            black_points = black_matches['black_score'].sum()
            
            # Matches as RED
            red_matches = results_df[results_df['red_player'] == player]
            red_wins = len(red_matches[red_matches['winner'] == player])
            red_points = red_matches['red_score'].sum()
            
            # Total stats
            total_matches = len(black_matches) + len(red_matches)
            total_wins = black_wins + red_wins
            draws = len(black_matches[black_matches['winner'] == "DRAW"]) + \
                   len(red_matches[red_matches['winner'] == "DRAW"])
            
            losses = total_matches - total_wins - draws
            
            # Points conceded
            points_conceded = black_matches['red_score'].sum() + red_matches['black_score'].sum()
            
            # Win rate calculation
            win_rate = (total_wins / total_matches * 100) if total_matches > 0 else 0
            
            # Separately track stats by color to identify color bias
            black_win_rate = (black_wins / len(black_matches) * 100) if len(black_matches) > 0 else 0
            red_win_rate = (red_wins / len(red_matches) * 100) if len(red_matches) > 0 else 0
            
            # Create stat record
            stat = {
                'player': player,
                'total_matches': total_matches,
                'wins': total_wins,
                'losses': losses,
                'draws': draws,
                'win_rate': win_rate,
                'black_win_rate': black_win_rate,
                'red_win_rate': red_win_rate,
                'points_scored': black_points + red_points,
                'points_conceded': points_conceded,
                'point_differential': (black_points + red_points) - points_conceded
            }
            player_stats.append(stat)
            
        # Convert to DataFrame and sort by win rate
        summary_df = pd.DataFrame(player_stats)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('win_rate', ascending=False)
            
        return summary_df
    
    def _create_visualizations(self):
        """Create visualizations of tournament results."""
        if not self.tournament_results:
            self.logger.warning("No results to visualize")
            return
            
        try:
            # Convert results to DataFrame if needed
            results_df = pd.DataFrame(self.tournament_results)
            
            # Create results matrix visualization (RED vs BLACK)
            self._create_results_matrix(results_df)
            
            # Create player performance by color visualization
            self._create_color_bias_chart(results_df)
            
            # Create win rate comparison chart
            self._create_win_rate_chart(results_df)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_results_matrix(self, results_df):
        """Create a matrix visualization showing win rates with RED players as rows and BLACK as columns."""
        # Get all unique player names
        all_players = sorted(list(set(results_df['black_player'].unique()) | 
                                 set(results_df['red_player'].unique())))
        
        # Initialize matrix for win counts and total games
        matrix_data = np.zeros((len(all_players), len(all_players), 2), dtype=int)
        
        # Fill matrix with results
        for _, row in results_df.iterrows():
            red_player = row['red_player']
            black_player = row['black_player']
            winner = row['winner']
            
            # Find indices
            red_idx = all_players.index(red_player)
            black_idx = all_players.index(black_player)
            
            # Increment total games counter
            matrix_data[red_idx, black_idx, 1] += 1
            
            # Increment win counter if RED won
            if winner == red_player:
                matrix_data[red_idx, black_idx, 0] += 1
        
        # Calculate win percentages
        win_pct_matrix = np.zeros((len(all_players), len(all_players)))
        for i in range(len(all_players)):
            for j in range(len(all_players)):
                total = matrix_data[i, j, 1]
                if total > 0:
                    win_pct_matrix[i, j] = (matrix_data[i, j, 0] / total) * 100
                else:
                    win_pct_matrix[i, j] = np.nan  # No games played
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Create heatmap (mask diagonal to avoid self-play confusion)
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
        
        # Save visualization
        output_path = os.path.join(self.results_dir, 'results_matrix.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Results matrix saved to {output_path}")

    def _create_color_bias_chart(self, results_df):
        """Create a chart showing win rates by player color to identify color bias."""
        # Calculate win rates by player and color
        player_color_stats = []
        
        # Get unique player names
        all_players = sorted(list(set(results_df['black_player'].unique()) | 
                                 set(results_df['red_player'].unique())))
        
        for player in all_players:
            # Matches as BLACK
            black_matches = results_df[results_df['black_player'] == player]
            black_total = len(black_matches)
            black_wins = len(black_matches[black_matches['winner'] == player])
            black_win_rate = (black_wins / black_total * 100) if black_total > 0 else 0
            
            # Matches as RED
            red_matches = results_df[results_df['red_player'] == player]
            red_total = len(red_matches)
            red_wins = len(red_matches[red_matches['winner'] == player])
            red_win_rate = (red_wins / red_total * 100) if red_total > 0 else 0
            
            player_color_stats.append({
                'player': player,
                'black_win_rate': black_win_rate,
                'red_win_rate': red_win_rate
            })
        
        # Convert to DataFrame
        color_stats_df = pd.DataFrame(player_color_stats)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Set width and positions for bars
        bar_width = 0.35
        x = np.arange(len(all_players))
        
        # Create bars
        plt.bar(x - bar_width/2, color_stats_df['black_win_rate'], bar_width, 
                label='Playing as BLACK', color='gray')
        plt.bar(x + bar_width/2, color_stats_df['red_win_rate'], bar_width,
                label='Playing as RED', color='red', alpha=0.7)
        
        # Add details
        plt.xlabel('Player')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rates by Player Color')
        plt.xticks(x, all_players, rotation=45, ha='right')
        plt.legend()
        
        # Add text labels on bars
        for i, player in enumerate(all_players):
            plt.text(i - bar_width/2, color_stats_df.iloc[i]['black_win_rate'] + 2, 
                     f"{color_stats_df.iloc[i]['black_win_rate']:.1f}%", 
                     ha='center', va='bottom')
            plt.text(i + bar_width/2, color_stats_df.iloc[i]['red_win_rate'] + 2, 
                     f"{color_stats_df.iloc[i]['red_win_rate']:.1f}%", 
                     ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.results_dir, 'color_bias_chart.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Color bias chart saved to {output_path}")
        
    def _create_win_rate_chart(self, results_df):
        """Create a chart comparing overall win rates."""
        # Calculate statistics
        summary_stats = self._calculate_summary_stats(results_df)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Plot win rates
        players = summary_stats['player']
        win_rates = summary_stats['win_rate']
        
        # Create bars
        bars = plt.bar(range(len(players)), win_rates, color='blue', alpha=0.7)
        
        # Add details
        plt.xlabel('Player')
        plt.ylabel('Overall Win Rate (%)')
        plt.title('Player Performance Comparison')
        plt.xticks(range(len(players)), players, rotation=45, ha='right')
        
        # Add text labels on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{win_rates.iloc[i]:.1f}%", ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.results_dir, 'win_rate_chart.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Win rate chart saved to {output_path}")


def main():
    """Command-line entry point for the improved tournament."""
    parser = argparse.ArgumentParser(description="Improved Kulibrat AI Tournament Runner")
    parser.add_argument(
        '--config', 
        type=str, 
        default='tournament_config.yaml',
        help='Path to tournament configuration YAML file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Check if config file exists
        if not os.path.exists(args.config):
            print(f"Configuration file not found: {args.config}")
            return 1
            
        # Create and run tournament
        tournament = ImprovedTournament(args.config, args.verbose)
        tournament.run_tournament()
        
        print(f"Tournament completed successfully. Results saved to {tournament.results_dir}")
        return 0
        
    except Exception as e:
        print(f"Tournament execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
