"""
Tournament evaluation core functionality.
"""

from collections import defaultdict
import logging
import os
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm

from src.tournament.factory import AIPlayerFactory
from src.tournament.match import TournamentMatch
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

        progress = tqdm(total=total_matches, desc="Running Tournament")

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
                            progress.update(1)

                            # Save interim results every 10% of matches
                            if current_match % max(1, total_matches // 10) == 0:
                                self._save_interim_results(results_dir, current_match)

            # Save and analyze results
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
        finally:
            progress.close()

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

    def _create_matchup_visualization(self, player1: str, player2: str, matchup_stat: Dict, 
                                     move_types_p1: Dict, move_types_p2: Dict, matchup_dir: str):
        """
        Create visualizations for a specific player matchup.
        
        Args:
            player1: First player name
            player2: Second player name
            matchup_stat: Dictionary with matchup statistics
            move_types_p1: Dictionary of move types for player 1
            move_types_p2: Dictionary of move types for player 2
            matchup_dir: Directory to save matchup visualizations
        """
        # Set up figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Matchup Analysis: {player1} vs {player2}", fontsize=16)
        
        # 1. Win Distribution - Top Left
        labels = [f"{player1} Wins", f"{player2} Wins", "Draws"]
        sizes = [matchup_stat['player1_wins'], matchup_stat['player2_wins'], matchup_stat['draws']]
        colors = [self.player_colors.get(player1, 'blue'), 
                 self.player_colors.get(player2, 'red'), 
                 'lightgray']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                     shadow=False, startangle=90)
        axes[0, 0].set_title("Win Distribution")
        
        # 2. Performance by Color - Top Right
        colors = ['black', 'red']
        player1_rates = [matchup_stat['player1_black_win_rate'], matchup_stat['player1_red_win_rate']]
        player2_rates = [100 - matchup_stat['player1_black_win_rate'], 100 - matchup_stat['player1_red_win_rate']]
        
        width = 0.35
        x = np.arange(len(colors))
        
        axes[0, 1].bar(x - width/2, player1_rates, width, label=player1, 
                     color=self.player_colors.get(player1, 'blue'))
        axes[0, 1].bar(x + width/2, player2_rates, width, label=player2, 
                     color=self.player_colors.get(player2, 'red'))
        
        axes[0, 1].set_xlabel('Player Color')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].set_title('Performance by Color')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(colors)
        axes[0, 1].legend()
        
        # 3. Move Type Distribution - Bottom Left
        move_types = sorted(set(move_types_p1.keys()) | set(move_types_p2.keys()))
        p1_values = [move_types_p1.get(mt, 0) for mt in move_types]
        p2_values = [move_types_p2.get(mt, 0) for mt in move_types]
        
        # Normalize to percentages
        p1_total = sum(p1_values)
        p2_total = sum(p2_values)
        p1_pct = [v/p1_total*100 if p1_total > 0 else 0 for v in p1_values]
        p2_pct = [v/p2_total*100 if p2_total > 0 else 0 for v in p2_values]
        
        x = np.arange(len(move_types))
        
        axes[1, 0].bar(x - width/2, p1_pct, width, label=player1, 
                     color=self.player_colors.get(player1, 'blue'))
        axes[1, 0].bar(x + width/2, p2_pct, width, label=player2, 
                     color=self.player_colors.get(player2, 'red'))
        
        axes[1, 0].set_xlabel('Move Types')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_title('Move Type Distribution')
        axes[1, 0].set_xticks(x)
        # Format move type labels
        labels = [mt.replace('MOVE_TYPES[', '').replace(']', '') if mt.startswith('MOVE_TYPES') else mt 
                for mt in move_types]
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 0].legend()
        
        # 4. Points Scored - Bottom Right
        labels = ['Points Scored', 'Points Conceded', 'Point Differential']
        p1_points = [matchup_stat['player1_points'], 
                    matchup_stat['player2_points'], 
                    matchup_stat['player1_points'] - matchup_stat['player2_points']]
        p2_points = [matchup_stat['player2_points'], 
                    matchup_stat['player1_points'], 
                    matchup_stat['player2_points'] - matchup_stat['player1_points']]
        
        x = np.arange(len(labels))
        
        axes[1, 1].bar(x - width/2, p1_points, width, label=player1, 
                     color=self.player_colors.get(player1, 'blue'))
        axes[1, 1].bar(x + width/2, p2_points, width, label=player2, 
                     color=self.player_colors.get(player2, 'red'))
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Points')
        axes[1, 1].set_title('Point Distribution')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].legend()
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        # Save the visualization
        output_path = os.path.join(matchup_dir, f"{player1}_vs_{player2}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

    def _generate_visualizations(self, results_dir: str):
        """
        Generate tournament visualizations.

        Args:
            results_dir: Directory to save visualizations
        """
        # Ensure matplotlib doesn't use GUI backend
        plt.switch_backend('Agg')
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

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

        # Generate various tournament visualizations
        try:
            # Visualization 1: Overall Win Rate
            self._create_win_rate_visualization(summary_df, viz_dir)
            
            # Visualization 2: Win Ratio Matrix
            self._create_win_matrix_visualization(results_df, viz_dir)
            
            # Visualization 3: Performance by Color
            self._create_color_performance_visualization(summary_df, viz_dir)
            
            # Visualization 4: Move Type Distribution
            self._create_move_type_visualization(summary_df, viz_dir)
            
            # Visualization 5: Points Scored vs Conceded
            self._create_points_visualization(summary_df, viz_dir)
            
            # Visualization 6: Tournament Overview Dashboard
            self._create_tournament_dashboard(summary_df, results_df, viz_dir)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            traceback.print_exc()

    def _create_win_rate_visualization(self, summary_df: pd.DataFrame, viz_dir: str):
        """
        Create win rate visualization.
        
        Args:
            summary_df: DataFrame with tournament summary statistics
            viz_dir: Directory to save visualizations
        """
        plt.figure(figsize=(12, 8))
        
        # Sort by win rate
        df = summary_df.sort_values('win_rate', ascending=False)
        
        # Create bar plot with custom colors
        colors = [self.player_colors.get(player, 'blue') for player in df['player']]
        
        # Main bars
        ax = sns.barplot(x='player', y='win_rate', data=df, palette=colors)
        
        # Add value labels on top of bars
        for i, value in enumerate(df['win_rate']):
            ax.text(i, value + 1, f"{value:.1f}%", ha='center', va='bottom', fontweight='bold')
        
        # Add wins/total as text inside bars
        for i, (wins, total) in enumerate(zip(df['wins'], df['total_matches'])):
            ax.text(i, df['win_rate'].iloc[i]/2, f"{wins}/{total}", 
                   ha='center', va='center', color='white', fontweight='bold')
        
        plt.title('Tournament Win Rates by Player', fontsize=16)
        plt.xlabel('Player', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.ylim(0, max(df['win_rate']) * 1.1)  # Add some space for labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(viz_dir, 'win_rates.png'), dpi=300)
        plt.close()

    def _create_win_matrix_visualization(self, results_df: pd.DataFrame, viz_dir: str):
        """
        Create a win matrix heatmap showing head-to-head performance.
        
        Args:
            results_df: DataFrame with tournament results
            viz_dir: Directory to save visualizations
        """
        # Get unique players
        all_players = sorted(list(set(results_df['player1'].unique()) | set(results_df['player2'].unique())))
        
        # Create empty win matrix
        win_matrix = pd.DataFrame(
            index=all_players,
            columns=all_players,
            data=np.zeros((len(all_players), len(all_players)))
        )
        games_matrix = pd.DataFrame(
            index=all_players,
            columns=all_players,
            data=np.zeros((len(all_players), len(all_players)))
        )
        
        # Fill matrices with data
        for _, row in results_df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            winner = row['winner']
            
            # Count games
            games_matrix.loc[player1, player2] += 1
            games_matrix.loc[player2, player1] += 1
            
            # Count wins
            if winner == player1:
                win_matrix.loc[player1, player2] += 1
            elif winner == player2:
                win_matrix.loc[player2, player1] += 1
        
        # Calculate win percentages
        win_pct_matrix = win_matrix / games_matrix * 100
        win_pct_matrix = win_pct_matrix.fillna(0)
        
        # Create win percentage matrix visualization
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.eye(len(all_players), dtype=bool)  # Mask for diagonal
        
        # Create heatmap with rounded values
        heatmap = sns.heatmap(
            win_pct_matrix, 
            annot=True, 
            fmt='.1f', 
            cmap='YlGnBu',
            linewidths=1, 
            linecolor='gray',
            mask=mask,
            cbar_kws={'label': 'Win Percentage (%)'}
        )
        
        # Customize labels
        plt.title('Head-to-Head Win Percentage Matrix', fontsize=16)
        plt.xlabel('Opponent', fontsize=14)
        plt.ylabel('Player', fontsize=14)
        
        # Add game counts as text annotations in each cell
        for i, player1 in enumerate(all_players):
            for j, player2 in enumerate(all_players):
                if i != j:  # Skip diagonal
                    games = int(games_matrix.loc[player1, player2])
                    wins = int(win_matrix.loc[player1, player2])
                    plt.text(
                        j + 0.5, i + 0.8, 
                        f"W:{wins}/G:{games//2}", 
                        ha='center', va='center', 
                        fontsize=9, 
                        color='black' if win_pct_matrix.loc[player1, player2] < 70 else 'white',
                        alpha=0.7
                    )
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'win_matrix.png'), dpi=300)
        plt.close()
        
        # Save win matrix as CSV
        win_pct_matrix.to_csv(os.path.join(viz_dir, 'win_percentage_matrix.csv'))
        games_matrix.to_csv(os.path.join(viz_dir, 'games_played_matrix.csv'))

    def _create_color_performance_visualization(self, summary_df: pd.DataFrame, viz_dir: str):
        """
        Create visualization showing player performance by color.
        
        Args:
            summary_df: DataFrame with tournament summary statistics
            viz_dir: Directory to save visualizations
        """
        plt.figure(figsize=(14, 8))
        
        # Sort by overall win rate
        df = summary_df.sort_values('win_rate', ascending=False)
        
        # Prepare data for grouped bar chart
        x = np.arange(len(df))
        width = 0.35
        
        # Create grouped bar chart
        ax = plt.subplot(111)
        black_bars = ax.bar(x - width/2, df['black_win_rate'], width, label='As BLACK', color='#404040')
        red_bars = ax.bar(x + width/2, df['red_win_rate'], width, label='As RED', color='#B03060')
        
        # Add value labels on top of bars
        for i, bar in enumerate(black_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f"{height:.1f}%", ha='center', va='bottom', fontsize=9)
            
        for i, bar in enumerate(red_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f"{height:.1f}%", ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        plt.title('Win Rate by Player Color', fontsize=16)
        plt.xlabel('Player', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.xticks(x, df['player'], rotation=45, ha='right')
        plt.ylim(0, max(max(df['black_win_rate']), max(df['red_win_rate'])) * 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add average line
        overall_avg = df['win_rate'].mean()
        plt.axhline(y=overall_avg, color='green', linestyle='--', alpha=0.8, 
                   label=f'Overall Average ({overall_avg:.1f}%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'color_performance.png'), dpi=300)
        plt.close()

    def _create_move_type_visualization(self, summary_df: pd.DataFrame, viz_dir: str):
        """
        Create visualization showing move type distribution by player.
        
        Args:
            summary_df: DataFrame with tournament summary statistics
            viz_dir: Directory to save visualizations
        """
        # Get move type columns
        move_type_cols = [col for col in summary_df.columns if col.startswith('move_type_')]
        
        if not move_type_cols:
            self.logger.warning("No move type data found in summary stats")
            return
        
        # Prepare data for stacked bar chart
        df = summary_df.sort_values('win_rate', ascending=False)
        move_data = df[['player'] + move_type_cols].set_index('player')
        
        # Clean up column names for better display
        move_data.columns = [col.replace('move_type_', '') for col in move_data.columns]
        
        # Create stacked bar chart
        plt.figure(figsize=(14, 8))
        
        ax = move_data.plot(
            kind='bar',
            stacked=True, 
            figsize=(14, 8),
            colormap='tab10'
        )
        
        # Customize plot
        plt.title('Move Type Distribution by Player', fontsize=16)
        plt.xlabel('Player', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend(title='Move Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add win rate text annotation above each bar
        for i, player in enumerate(move_data.index):
            win_rate = df[df['player'] == player]['win_rate'].values[0]
            ax.text(i, 101, f"Win Rate: {win_rate:.1f}%", ha='center', fontsize=9, 
                   color='black', fontweight='bold', rotation=0)
        
        plt.ylim(0, 110)  # Add space for the win rate text
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'move_type_distribution.png'), dpi=300)
        plt.close()

    def _create_points_visualization(self, summary_df: pd.DataFrame, viz_dir: str):
        """
        Create visualization showing points scored vs conceded by player.
        
        Args:
            summary_df: DataFrame with tournament summary statistics
            viz_dir: Directory to save visualizations
        """
        plt.figure(figsize=(12, 10))
        
        # Sort by win rate
        df = summary_df.sort_values('win_rate', ascending=False)
        
        # Calculate point averages
        matches_per_player = df['total_matches']
        df['avg_points_scored'] = df['points_scored'] / matches_per_player
        df['avg_points_conceded'] = df['points_conceded'] / matches_per_player
        
        # Create scatter plot
        scatter = plt.scatter(
            df['avg_points_conceded'], 
            df['avg_points_scored'],
            s=df['win_rate'] * 10,  # Size proportional to win rate
            c=[self.player_colors.get(p, 'blue') for p in df['player']],
            alpha=0.7
        )
        
        # Add player labels
        for i, player in enumerate(df['player']):
            plt.annotate(
                player,
                (df['avg_points_conceded'].iloc[i], df['avg_points_scored'].iloc[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        # Add reference line (y=x)
        max_val = max(df['avg_points_scored'].max(), df['avg_points_conceded'].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal Points Line')
        
        # Customize plot
        plt.title('Points Scored vs. Conceded per Match', fontsize=16)
        plt.xlabel('Average Points Conceded per Match', fontsize=12)
        plt.ylabel('Average Points Scored per Match', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add win rate legend
        win_rates = [20, 40, 60, 80]
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=np.sqrt(wr * 10), label=f'{wr}% Win Rate')
            for wr in win_rates
        ]
        plt.legend(handles=legend_elements, title="Win Rate Legend", 
                  loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'points_analysis.png'), dpi=300)
        plt.close()

    def _create_tournament_dashboard(self, summary_df: pd.DataFrame, results_df: pd.DataFrame, viz_dir: str):
        """
        Create comprehensive tournament dashboard with key statistics.
        
        Args:
            summary_df: DataFrame with tournament summary statistics
            results_df: DataFrame with tournament results
            viz_dir: Directory to save visualizations
        """
        # Create large figure with subplots
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle('Tournament Analysis Dashboard', fontsize=20, y=0.98)
        
        # Grid setup: 2 rows, 2 columns
        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)
        
        # 1. Win Rate Bar Chart (Top Left)
        ax1 = fig.add_subplot(grid[0, 0])
        df_sorted = summary_df.sort_values('win_rate', ascending=False)
        colors = [self.player_colors.get(player, 'blue') for player in df_sorted['player']]
        sns.barplot(x='player', y='win_rate', data=df_sorted, palette=colors, ax=ax1)
        
        # Add value labels
        for i, value in enumerate(df_sorted['win_rate']):
            ax1.text(i, value + 1, f"{value:.1f}%", ha='center', va='bottom', fontsize=9)
        
        ax1.set_title('Player Win Rates', fontsize=14)
        ax1.set_xlabel('Player', fontsize=10)
        ax1.set_ylabel('Win Rate (%)', fontsize=10)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Win Matrix Heatmap (Top Right)
        ax2 = fig.add_subplot(grid[0, 1])
        
        # Create win matrix
        all_players = sorted(list(set(results_df['player1'].unique()) | set(results_df['player2'].unique())))
        win_matrix = pd.DataFrame(index=all_players, columns=all_players, data=np.zeros((len(all_players), len(all_players))))
        games_matrix = pd.DataFrame(index=all_players, columns=all_players, data=np.zeros((len(all_players), len(all_players))))
        
        # Fill matrices
        for _, row in results_df.iterrows():
            p1, p2, winner = row['player1'], row['player2'], row['winner']
            games_matrix.loc[p1, p2] += 1
            games_matrix.loc[p2, p1] += 1
            if winner == p1:
                win_matrix.loc[p1, p2] += 1
            elif winner == p2:
                win_matrix.loc[p2, p1] += 1
        
        # Win percentage
        win_pct_matrix = win_matrix / games_matrix * 100
        win_pct_matrix = win_pct_matrix.fillna(0)
        
        # Create heatmap
        mask = np.eye(len(all_players), dtype=bool)
        sns.heatmap(
            win_pct_matrix, 
            annot=True, 
            fmt='.1f', 
            cmap='YlGnBu',
            linewidths=0.5, 
            mask=mask,
            ax=ax2,
            cbar_kws={'label': 'Win %'}
        )
        
        ax2.set_title('Head-to-Head Win Percentage', fontsize=14)
        ax2.set_xlabel('Opponent', fontsize=10)
        ax2.set_ylabel('Player', fontsize=10)
        
        # 3. Performance by Color (Bottom Left)
        ax3 = fig.add_subplot(grid[1, 0])
        
        # Prepare data
        x = np.arange(len(df_sorted))
        width = 0.35
        
        # Create bars
        black_bars = ax3.bar(x - width/2, df_sorted['black_win_rate'], width, label='As BLACK', color='#404040')
        red_bars = ax3.bar(x + width/2, df_sorted['red_win_rate'], width, label='As RED', color='#B03060')
        
        ax3.set_title('Win Rate by Player Color', fontsize=14)
        ax3.set_xlabel('Player', fontsize=10)
        ax3.set_ylabel('Win Rate (%)', fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(df_sorted['player'], rotation=45, ha='right')
        ax3.legend()
        
        # 4. Turn Distribution Violin Plot (Bottom Right)
        ax4 = fig.add_subplot(grid[1, 1])
        
        # Create violin plot
        sns.violinplot(
            x='winner', 
            y='turns', 
            data=results_df[results_df['winner'].notna()],
            palette=[self.player_colors.get(p, 'blue') for p in sorted(results_df['winner'].unique())],
            ax=ax4
        )
        
        ax4.set_title('Turn Distribution by Winner', fontsize=14)
        ax4.set_xlabel('Winner', fontsize=10)
        ax4.set_ylabel('Number of Turns', fontsize=10)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add tournament metadata
        metadata_text = (
            f"Total Players: {len(summary_df)}\n"
            f"Total Matches: {len(results_df)}\n"
            f"Avg. Turns per Match: {results_df['turns'].mean():.1f}\n"
            f"Draws: {len(results_df[results_df['winner'].isna()])}\n"
            f"Tournament Date: {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        plt.figtext(0.5, 0.01, metadata_text, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig(os.path.join(viz_dir, 'tournament_dashboard.png'), dpi=300)
        plt.close()