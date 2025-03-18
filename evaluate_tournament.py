#!/usr/bin/env python3
"""
Run a tournament between different AI players for Kulibrat and evaluate performance metrics.
This script pits various AI strategies against each other in a round-robin tournament
to generate comparative performance data.
"""

import argparse
import os
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.move_type import MoveType
from src.core.move import Move
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.mcts_strategy import MCTSStrategy
from src.players.ai.rl_player import RLPlayer
from src.players.ai.alphazero_player import AlphaZeroPlayer


class TournamentMatch:
    """Class to handle and record a match between two players."""
    
    def __init__(
        self, 
        player1_name: str, 
        player1, 
        player2_name: str, 
        player2, 
        target_score: int = 5,
        max_turns: int = 300,
    ):
        """
        Initialize a tournament match.
        
        Args:
            player1_name: Name of player 1
            player1: Player 1 object
            player2_name: Name of player 2
            player2: Player 2 object
            target_score: Score needed to win
            max_turns: Maximum turns before declaring a draw
        """
        self.player1_name = player1_name
        self.player1 = player1
        self.player2_name = player2_name
        self.player2 = player2
        self.target_score = target_score
        self.max_turns = max_turns
        
        # Match statistics to track
        self.winner = None
        self.num_turns = 0
        self.match_duration = 0
        self.final_score = (0, 0)  # (player1_score, player2_score)
        self.moves_used = {
            player1_name: {
                'INSERT': 0,
                'DIAGONAL': 0,
                'ATTACK': 0,
                'JUMP': 0
            },
            player2_name: {
                'INSERT': 0,
                'DIAGONAL': 0,
                'ATTACK': 0,
                'JUMP': 0
            }
        }
        
    def run_match(self, swap_colors: bool = False) -> Dict[str, Any]:
        """
        Run a match between two players.
        
        Args:
            swap_colors: Whether to swap the default colors (player1=BLACK, player2=RED)
            
        Returns:
            Dictionary with match statistics
        """
        # Set up players with appropriate colors
        if not swap_colors:
            black_player = self.player1
            red_player = self.player2
            black_player_name = self.player1_name
            red_player_name = self.player2_name
            black_player.color = PlayerColor.BLACK
            red_player.color = PlayerColor.RED
        else:
            black_player = self.player2
            red_player = self.player1
            black_player_name = self.player2_name
            red_player_name = self.player1_name
            black_player.color = PlayerColor.BLACK
            red_player.color = PlayerColor.RED
        
        # Initialize game state
        game_state = GameState(target_score=self.target_score)
        
        # Reset match statistics
        self.winner = None
        self.num_turns = 0
        self.moves_used = {
            self.player1_name: {
                'INSERT': 0,
                'DIAGONAL': 0,
                'ATTACK': 0,
                'JUMP': 0
            },
            self.player2_name: {
                'INSERT': 0,
                'DIAGONAL': 0,
                'ATTACK': 0,
                'JUMP': 0
            }
        }
        
        # Start timer
        start_time = time.time()
        
        # Play until game is over or max turns reached
        while not game_state.is_game_over() and self.num_turns < self.max_turns:
            # Get current player
            current_color = game_state.current_player
            if current_color == PlayerColor.BLACK:
                current_player = black_player
                current_player_name = black_player_name
            else:
                current_player = red_player
                current_player_name = red_player_name
            
            # Get move from the current player
            move = current_player.get_move(game_state)
            
            if not move:
                # Player couldn't make a move, skip turn
                game_state.current_player = game_state.current_player.opposite()
                continue
            
            # Record move type
            player_name = self.player1_name if (
                (current_player == self.player1 and not swap_colors) or 
                (current_player == self.player2 and swap_colors)
            ) else self.player2_name
            
            self.moves_used[player_name][move.move_type.name] += 1
            
            # Apply move
            success = game_state.apply_move(move)
            
            if success:
                # Switch to the other player
                game_state.current_player = game_state.current_player.opposite()
                self.num_turns += 1
        
        # End timer
        end_time = time.time()
        self.match_duration = end_time - start_time
        
        # Determine final score and winner
        self.final_score = (
            game_state.scores[PlayerColor.BLACK if not swap_colors else PlayerColor.RED],
            game_state.scores[PlayerColor.RED if not swap_colors else PlayerColor.BLACK]
        )
        
        # Determine winner based on score or game state
        if game_state.is_game_over():
            winner_color = game_state.get_winner()
            
            if winner_color == PlayerColor.BLACK:
                self.winner = black_player_name
            elif winner_color == PlayerColor.RED:
                self.winner = red_player_name
            else:
                self.winner = None  # Draw
        else:
            # Draw due to max turns
            self.winner = None
        
        # Map winner back to player1 or player2
        winner_result = None
        if self.winner:
            winner_result = 1 if self.winner == self.player1_name else 2 if self.winner == self.player2_name else 0
        
        # Compile match statistics
        match_stats = {
            'player1': self.player1_name,
            'player2': self.player2_name,
            'winner': winner_result,  # 1, 2, or None (draw)
            'score': self.final_score,
            'turns': self.num_turns,
            'duration': self.match_duration,
            'moves_used': self.moves_used,
            'p1_as_black': not swap_colors
        }
        
        return match_stats


class KulibratTournament:
    """Class to run a tournament between different AI players."""
    
    def __init__(
        self,
        players: Dict[str, Any],
        matches_per_pairing: int = 10,
        target_score: int = 5,
        max_turns_per_game: int = 300,
        results_dir: str = "tournament_results"
    ):
        """
        Initialize the tournament.
        
        Args:
            players: Dictionary mapping player names to player objects
            matches_per_pairing: Number of matches for each player pair
            target_score: Score needed to win each match
            max_turns_per_game: Maximum turns before declaring a draw
            results_dir: Directory to save results
        """
        self.players = players
        self.matches_per_pairing = matches_per_pairing
        self.target_score = target_score
        self.max_turns_per_game = max_turns_per_game
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize result tracking
        self.results = []
        self.pairwise_results = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0}))
        self.player_stats = defaultdict(lambda: {
            'matches': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'points_scored': 0,
            'points_conceded': 0,
            'avg_turns_per_win': [],
            'avg_duration_per_game': [],
            'move_types': defaultdict(int)
        })
        
    def run_tournament(self):
        """Run the full tournament with all player pairings."""
        player_names = list(self.players.keys())
        total_pairings = len(player_names) * (len(player_names) - 1) // 2
        total_matches = total_pairings * self.matches_per_pairing
        
        print(f"Starting tournament with {len(player_names)} players")
        print(f"Total pairings: {total_pairings}")
        print(f"Matches per pairing: {self.matches_per_pairing}")
        print(f"Total matches: {total_matches}")
        print(f"Target score: {self.target_score}")
        print(f"Max turns per game: {self.max_turns_per_game}")
        print()
        
        progress_bar = tqdm(total=total_matches, desc="Tournament Progress")
        
        # Run matches for all pairs of players
        for i, player1_name in enumerate(player_names):
            for j, player2_name in enumerate(player_names):
                if i >= j:  # Skip same player and redundant pairings
                    continue
                
                player1 = self.players[player1_name]
                player2 = self.players[player2_name]
                
                # Create match handler
                match_handler = TournamentMatch(
                    player1_name, 
                    player1, 
                    player2_name, 
                    player2,
                    target_score=self.target_score,
                    max_turns=self.max_turns_per_game
                )
                
                # Run multiple matches for this pairing
                for match_idx in range(self.matches_per_pairing):
                    # Swap colors every other match
                    swap_colors = match_idx % 2 == 1
                    
                    # Run the match
                    match_result = match_handler.run_match(swap_colors=swap_colors)
                    self.results.append(match_result)
                    
                    # Update pairwise results
                    if match_result['winner'] == 1:
                        self.pairwise_results[player1_name][player2_name]['wins'] += 1
                        self.pairwise_results[player2_name][player1_name]['losses'] += 1
                    elif match_result['winner'] == 2:
                        self.pairwise_results[player1_name][player2_name]['losses'] += 1
                        self.pairwise_results[player2_name][player1_name]['wins'] += 1
                    else:
                        self.pairwise_results[player1_name][player2_name]['draws'] += 1
                        self.pairwise_results[player2_name][player1_name]['draws'] += 1
                    
                    # Update player stats
                    for player_idx, player_name in enumerate([player1_name, player2_name]):
                        other_idx = 1 - player_idx
                        other_name = [player1_name, player2_name][other_idx]
                        
                        # Basic stats
                        self.player_stats[player_name]['matches'] += 1
                        
                        if match_result['winner'] == player_idx + 1:
                            self.player_stats[player_name]['wins'] += 1
                            self.player_stats[player_name]['avg_turns_per_win'].append(match_result['turns'])
                        elif match_result['winner'] == other_idx + 1:
                            self.player_stats[player_name]['losses'] += 1
                        else:
                            self.player_stats[player_name]['draws'] += 1
                        
                        # Points scored/conceded
                        if player_idx == 0:
                            self.player_stats[player_name]['points_scored'] += match_result['score'][0]
                            self.player_stats[player_name]['points_conceded'] += match_result['score'][1]
                        else:
                            self.player_stats[player_name]['points_scored'] += match_result['score'][1]
                            self.player_stats[player_name]['points_conceded'] += match_result['score'][0]
                        
                        # Duration
                        self.player_stats[player_name]['avg_duration_per_game'].append(match_result['duration'])
                        
                        # Move types
                        for move_type, count in match_result['moves_used'][player_name].items():
                            self.player_stats[player_name]['move_types'][move_type] += count
                    
                    # Update progress
                    progress_bar.update(1)
        
        progress_bar.close()
        
        # Compute final statistics
        self._compute_final_stats()
        
        # Generate reports
        self._generate_reports()
        
    def _compute_final_stats(self):
        """Compute final statistics based on match results."""
        # For each player, compute averages and derived metrics
        for player_name, stats in self.player_stats.items():
            # Win rate
            stats['win_rate'] = (stats['wins'] / stats['matches']) * 100 if stats['matches'] > 0 else 0
            
            # Average turns per win
            stats['avg_turns_per_win'] = np.mean(stats['avg_turns_per_win']) if stats['avg_turns_per_win'] else 0
            
            # Average duration per game
            stats['avg_duration_per_game'] = np.mean(stats['avg_duration_per_game']) if stats['avg_duration_per_game'] else 0
            
            # Point differential
            stats['point_differential'] = stats['points_scored'] - stats['points_conceded']
            
            # Compute move distribution
            total_moves = sum(stats['move_types'].values())
            if total_moves > 0:
                stats['move_distribution'] = {
                    move_type: (count / total_moves) * 100 
                    for move_type, count in stats['move_types'].items()
                }
            else:
                stats['move_distribution'] = {move_type: 0 for move_type in stats['move_types']}
            
            # Tournament score (3 for win, 1 for draw)
            stats['tournament_score'] = 3 * stats['wins'] + stats['draws']
        
    def _generate_reports(self):
        """Generate tournament reports and visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.results_dir, f"tournament_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Save raw results as CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(report_dir, "match_results.csv"), index=False)
        
        # 2. Generate player rankings
        rankings = []
        for player_name, stats in self.player_stats.items():
            rankings.append({
                'Player': player_name,
                'Matches': stats['matches'],
                'Wins': stats['wins'],
                'Losses': stats['losses'],
                'Draws': stats['draws'],
                'Win Rate (%)': stats['win_rate'],
                'Points Scored': stats['points_scored'],
                'Points Conceded': stats['points_conceded'],
                'Point Differential': stats['point_differential'],
                'Tournament Score': stats['tournament_score'],
                'Avg Turns per Win': stats['avg_turns_per_win'],
                'Avg Duration (s)': stats['avg_duration_per_game']
            })
        
        # Sort rankings by tournament score
        rankings_df = pd.DataFrame(rankings).sort_values(
            by=['Tournament Score', 'Win Rate (%)', 'Point Differential', 'Points Scored'],
            ascending=False
        )
        rankings_df.to_csv(os.path.join(report_dir, "player_rankings.csv"), index=False)
        
        # 3. Create win rate matrix
        player_names = list(self.players.keys())
        win_matrix = np.zeros((len(player_names), len(player_names)))
        
        for i, player1 in enumerate(player_names):
            for j, player2 in enumerate(player_names):
                if i != j:
                    pair_results = self.pairwise_results[player1][player2]
                    total_matches = pair_results['wins'] + pair_results['losses'] + pair_results['draws']
                    if total_matches > 0:
                        win_rate = (pair_results['wins'] / total_matches) * 100
                    else:
                        win_rate = 0
                    win_matrix[i, j] = win_rate
        
        # Create and save win rate heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            win_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            xticklabels=player_names,
            yticklabels=player_names,
            vmin=0,
            vmax=100
        )
        plt.title('Win Rate Matrix (%)')
        plt.xlabel('Opponent')
        plt.ylabel('Player')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "win_rate_matrix.png"), dpi=300)
        plt.close()
        
        # 4. Create win rate bar chart
        plt.figure(figsize=(12, 6))
        win_rates = [stats['win_rate'] for stats in self.player_stats.values()]
        bar_colors = plt.cm.viridis(np.linspace(0, 1, len(player_names)))
        
        bars = plt.bar(player_names, win_rates, color=bar_colors)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)  # 50% reference line
        
        plt.title('Win Rate by Player')
        plt.xlabel('Player')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add win rate labels
        for bar, win_rate in zip(bars, win_rates):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{win_rate:.1f}%',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "win_rate_chart.png"), dpi=300)
        plt.close()
        
        # 5. Create move type distribution chart
        plt.figure(figsize=(14, 8))
        
        move_types = ['INSERT', 'DIAGONAL', 'ATTACK', 'JUMP']
        move_data = []
        
        for player_name in player_names:
            player_move_dist = self.player_stats[player_name]['move_distribution']
            move_data.append([player_move_dist.get(move_type, 0) for move_type in move_types])
        
        move_data = np.array(move_data)
        
        # Create stacked bar chart
        bottom = np.zeros(len(player_names))
        
        for i, move_type in enumerate(move_types):
            plt.bar(
                player_names, 
                move_data[:, i], 
                bottom=bottom, 
                label=move_type
            )
            bottom += move_data[:, i]
        
        plt.title('Move Type Distribution by Player')
        plt.xlabel('Player')
        plt.ylabel('Percentage of Moves (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "move_distribution.png"), dpi=300)
        plt.close()
        
        # 6. Create tournament report text file
        with open(os.path.join(report_dir, "tournament_report.txt"), 'w') as f:
            f.write("=====================================================\n")
            f.write("             KULIBRAT TOURNAMENT REPORT              \n")
            f.write("=====================================================\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Players: {', '.join(player_names)}\n")
            f.write(f"Matches per pairing: {self.matches_per_pairing}\n")
            f.write(f"Target score: {self.target_score}\n")
            f.write(f"Max turns per game: {self.max_turns_per_game}\n\n")
            
            f.write("FINAL RANKINGS\n")
            f.write("-------------\n")
            f.write(rankings_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("PAIRWISE RESULTS\n")
            f.write("---------------\n")
            for player1 in player_names:
                for player2 in player_names:
                    if player1 != player2:
                        pair_results = self.pairwise_results[player1][player2]
                        total = pair_results['wins'] + pair_results['losses'] + pair_results['draws']
                        win_rate = (pair_results['wins'] / total) * 100 if total > 0 else 0
                        
                        f.write(f"{player1} vs {player2}: {pair_results['wins']}-{pair_results['losses']}-{pair_results['draws']}")
                        f.write(f" (Win rate: {win_rate:.1f}%)\n")
                
                f.write("\n")
            
            f.write("PLAYER STATISTICS\n")
            f.write("----------------\n")
            for player_name, stats in self.player_stats.items():
                f.write(f"{player_name}:\n")
                f.write(f"  Matches: {stats['matches']}\n")
                f.write(f"  Record: {stats['wins']}-{stats['losses']}-{stats['draws']}\n")
                f.write(f"  Win Rate: {stats['win_rate']:.1f}%\n")
                f.write(f"  Points: Scored={stats['points_scored']}, Conceded={stats['points_conceded']}\n")
                f.write(f"  Avg Turns per Win: {stats['avg_turns_per_win']:.1f}\n")
                f.write(f"  Avg Game Duration: {stats['avg_duration_per_game']:.2f}s\n")
                f.write(f"  Move Distribution:\n")
                
                for move_type, percentage in stats['move_distribution'].items():
                    f.write(f"    {move_type}: {percentage:.1f}%\n")
                
                f.write("\n")
        
        print(f"Tournament reports saved to: {report_dir}")
        
        return report_dir


def create_player(player_config: Dict[str, Any]):
    """
    Create a player based on configuration.
    
    Args:
        player_config: Dictionary with player configuration
        
    Returns:
        Player object
    """
    player_type = player_config['type']
    
    if player_type == 'random':
        return SimpleAIPlayer(
            color=PlayerColor.BLACK,  # Will be set later
            strategy=RandomStrategy(),
            name=player_config.get('name', 'Random')
        )
    
    elif player_type == 'minimax':
        depth = player_config.get('depth', 6)
        use_ab = player_config.get('use_alpha_beta', True)
        return SimpleAIPlayer(
            color=PlayerColor.BLACK,  # Will be set later
            strategy=MinimaxStrategy(max_depth=depth, use_alpha_beta=use_ab),
            name=player_config.get('name', f'Minimax(d={depth})')
        )
    
    elif player_type == 'mcts':
        sim_time = player_config.get('sim_time', 1.5)
        max_iter = player_config.get('max_iter', 30000)
        return SimpleAIPlayer(
            color=PlayerColor.BLACK,  # Will be set later
            strategy=MCTSStrategy(simulation_time=sim_time, max_iterations=max_iter),
            name=player_config.get('name', f'MCTS(t={sim_time})')
        )
    
    elif player_type == 'rl':
        model_path = player_config.get('model_path', 'models/rl_model.pth')
        temperature = player_config.get('temperature', 0.5)
        return RLPlayer(
            color=PlayerColor.BLACK,  # Will be set later
            model_path=model_path,
            exploration_rate=0,
            temperature=temperature,
            name=player_config.get('name', f'RL(t={temperature})')
        )
    
    elif player_type == 'alphazero':
        model_path = player_config.get('model_path', 'models/alphazero_model_best.pt')
        simulations = player_config.get('simulations', 800)
        temperature = player_config.get('temperature', 0.1)
        return AlphaZeroPlayer(
            color=PlayerColor.BLACK,  # Will be set later
            model_path=model_path,
            n_simulations=simulations,
            exploration_rate=0,
            temperature=temperature,
            name=player_config.get('name', f'AlphaZero(s={simulations})')
        )
    
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def main():
    """Main function to run the tournament evaluation."""
    parser = argparse.ArgumentParser(description="Run a tournament between different AI players for Kulibrat")
    
    parser.add_argument(
        "--matches-per-pairing",
        type=int,
        default=10,
        help="Number of matches for each player pairing (default: 10)"
    )
    
    parser.add_argument(
        "--target-score",
        type=int,
        default=5,
        help="Target score to win each game (default: 5)"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=300,
        help="Maximum turns before declaring a draw (default: 300)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="tournament_results",
        help="Directory to save tournament results (default: tournament_results)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file specifying players (optional)"
    )
    
    # Add player-specific arguments if no config file
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include random player in tournament"
    )
    
    parser.add_argument(
        "--include-minimax",
        action="store_true",
        help="Include minimax player in tournament"
    )
    
    parser.add_argument(
        "--minimax-depths",
        type=int,
        nargs="+",
        default=[4, 6],
        help="Depths for minimax search (default: [4, 6])"
    )
    
    parser.add_argument(
        "--include-mcts",
        action="store_true",
        help="Include MCTS player in tournament"
    )
    
    parser.add_argument(
        "--mcts-times",
        type=float,
        nargs="+",
        default=[0.5, 1.5],
        help="Simulation times for MCTS in seconds (default: [0.5, 1.5])"
    )
    
    parser.add_argument(
        "--include-rl",
        action="store_true",
        help="Include RL player in tournament"
    )
    
    parser.add_argument(
        "--rl-models",
        type=str,
        nargs="+",
        default=["models/rl_model.pth"],
        help="Paths to RL model files (default: ['models/rl_model.pth'])"
    )
    
    parser.add_argument(
        "--include-alphazero",
        action="store_true",
        help="Include AlphaZero player in tournament"
    )
    
    parser.add_argument(
        "--az-models",
        type=str,
        nargs="+",
        default=["models/alphazero_model_best.pt"],
        help="Paths to AlphaZero model files (default: ['models/alphazero_model_best.pt'])"
    )
    
    parser.add_argument(
        "--az-simulations",
        type=int,
        nargs="+",
        default=[200, 800],
        help="Simulation counts for AlphaZero (default: [200, 800])"
    )
    
    args = parser.parse_args()
    
    # Initialize players
    players = {}
    
    # If config file is provided, load players from it
    if args.config:
        import json
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
            if 'players' in config:
                for player_config in config['players']:
                    player_name = player_config.get('name')
                    if not player_name:
                        # Generate name based on type
                        player_type = player_config.get('type')
                        if player_type == 'random':
                            player_name = 'Random'
                        elif player_type == 'minimax':
                            depth = player_config.get('depth', 6)
                            player_name = f'Minimax-{depth}'
                        elif player_type == 'mcts':
                            sim_time = player_config.get('sim_time', 1.5)
                            player_name = f'MCTS-{sim_time}'
                        elif player_type == 'rl':
                            model_name = os.path.basename(player_config.get('model_path', 'model'))
                            player_name = f'RL-{model_name}'
                        elif player_type == 'alphazero':
                            simulations = player_config.get('simulations', 800)
                            player_name = f'AlphaZero-{simulations}'
                        else:
                            player_name = f'Player-{len(players) + 1}'
                    
                    # Check for duplicate names
                    if player_name in players:
                        i = 2
                        while f"{player_name}-{i}" in players:
                            i += 1
                        player_name = f"{player_name}-{i}"
                    
                    # Add player
                    player_config['name'] = player_name
                    players[player_name] = create_player(player_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using command line arguments instead.")
    
    # If no players loaded from config, use command line arguments
    if not players:
        # Add random player if requested
        if args.include_random:
            players["Random"] = create_player({'type': 'random'})
        
        # Add minimax players if requested
        if args.include_minimax:
            for depth in args.minimax_depths:
                players[f"Minimax-{depth}"] = create_player({
                    'type': 'minimax',
                    'depth': depth,
                    'use_alpha_beta': True
                })
        
        # Add MCTS players if requested
        if args.include_mcts:
            for sim_time in args.mcts_times:
                players[f"MCTS-{sim_time}"] = create_player({
                    'type': 'mcts',
                    'sim_time': sim_time,
                    'max_iter': 30000
                })
        
        # Add RL players if requested
        if args.include_rl:
            for model_path in args.rl_models:
                if os.path.exists(model_path):
                    model_name = os.path.basename(model_path).split('.')[0]
                    players[f"RL-{model_name}"] = create_player({
                        'type': 'rl',
                        'model_path': model_path
                    })
                else:
                    print(f"Warning: RL model not found at {model_path}, skipping")
        
        # Add AlphaZero players if requested
        if args.include_alphazero:
            for model_path in args.az_models:
                if os.path.exists(model_path):
                    model_name = os.path.basename(model_path).split('.')[0]
                    for simulations in args.az_simulations:
                        players[f"AlphaZero-{model_name}-{simulations}"] = create_player({
                            'type': 'alphazero',
                            'model_path': model_path,
                            'simulations': simulations
                        })
                else:
                    print(f"Warning: AlphaZero model not found at {model_path}, skipping")
    
    # Check if we have enough players
    if len(players) < 2:
        print("Error: At least 2 players are required for a tournament")
        print("Use --include-random, --include-minimax, --include-mcts, --include-rl, or --include-alphazero options")
        return
    
    print(f"Tournament will include {len(players)} players:")
    for name in players.keys():
        print(f"  - {name}")
    print()
    
    # Create and run tournament
    tournament = KulibratTournament(
        players=players,
        matches_per_pairing=args.matches_per_pairing,
        target_score=args.target_score,
        max_turns_per_game=args.max_turns,
        results_dir=args.results_dir
    )
    
    # Run the tournament
    tournament.run_tournament()


if __name__ == "__main__":
    main()