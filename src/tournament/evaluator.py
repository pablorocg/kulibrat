"""
Tournament evaluation core functionality.
"""

from collections import defaultdict
import logging
import os
import traceback
from datetime import datetime
from typing import List, Dict   

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm

from src.players.ai_player_factory import AIPlayerFactory
from src.tournament.tournament_match import TournamentMatch
from src.core.move_type import MoveType

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

        

       

        

    