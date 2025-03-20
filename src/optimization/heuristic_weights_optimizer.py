#!/usr/bin/env python3
"""
Heuristic Weights Optimization for Kulibrat AI Minimax Player.
"""

import random
import numpy as np
import logging
import os
import sys
import contextlib
import io

import deap.base
import deap.creator
import deap.tools
import deap.algorithms

from src.tournament.match import TournamentMatch
from src.players.minimax_player.minimax_player import MinimaxPlayer
from src.core.player_color import PlayerColor
from src.core.move import MoveType

class HeuristicWeightsOptimizer:
    def __init__(
        self, 
        config=None,
        num_weights=5,  # Default to 5 weights for normalized heuristic
        population_size=50,
        generations=15,
        crossover_prob=0.7,
        mutation_prob=0.2
    ):
        """
        Initialize Heuristic Weights Optimizer.
        
        Args:
            config: Configuration dictionary
            num_weights: Number of weights to optimize
            population_size: Size of the genetic population
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
        """
        # Default configuration
        self.config = {
            'population_size': population_size,
            'generations': generations,
            'crossover_prob': crossover_prob,
            'mutation_prob': mutation_prob,
            'weight_ranges': [(0.5, 2.0)] * num_weights,
            'num_opponents': 2,
            'matches_per_pairing': 10,
            'max_depth': 3,  # Limit depth for faster evaluation
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Silence logging
        self._silence_logging()
        
        # Prepare opponents
        self._prepare_opponents()
        
        # Setup genetic algorithm framework
        self._setup_genetic_algorithm()

    def _silence_logging(self):
        """Suppress logging output during optimization."""
        for logger_name in ['__main__', 'src', 'deap']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
        
        # Redirect stdout and stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def _prepare_opponents(self):
        """
        Prepare a set of opponent players to test against.
        """
        from src.players.random_player.random_player import RandomPlayer
        from src.players.mcts_player.mcts_player import MCTSPlayer

        # Define opponents with different strategies
        self.opponents = [
            # Random player
            RandomPlayer(color=PlayerColor.RED, name="Random"),
            
            # MCTS player
            MCTSPlayer(
                color=PlayerColor.RED, 
                name="MCTS", 
                simulation_time=1.0,
                max_iterations=5000
            ),
            
            # Alternative Minimax player
            MinimaxPlayer(
                color=PlayerColor.RED,
                name="Minimax-Alt",
                max_depth=self.config['max_depth'],
                use_alpha_beta=True,
                heuristic='score_diff'
            )
        ][:self.config['num_opponents']]

    def _setup_genetic_algorithm(self):
        """
        Setup the DEAP genetic algorithm framework.
        """
        # Define fitness and individual
        if not hasattr(deap.creator, "FitnessMax"):
            deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        if not hasattr(deap.creator, "Individual"):
            deap.creator.create("Individual", list, fitness=deap.creator.FitnessMax)
        
        # Toolbox for genetic operations
        self.toolbox = deap.base.Toolbox()
        
        # Get weight ranges from config
        weight_ranges = self.config['weight_ranges']
        
        # Attribute generators for each weight
        for i, (min_val, max_val) in enumerate(weight_ranges):
            self.toolbox.register(
                f"attr_weight_{i}", 
                random.uniform, 
                min_val, 
                max_val
            )
        
        # Individual and population creation
        self.toolbox.register(
            "individual", 
            deap.tools.initCycle, 
            deap.creator.Individual,
            tuple(
                self.toolbox.attrs[f"attr_weight_{i}"] 
                for i in range(len(weight_ranges))
            ),
            n=1
        )
        
        self.toolbox.register(
            "population", 
            deap.tools.initRepeat, 
            list, 
            self.toolbox.individual
        )
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_weights)
        self.toolbox.register("mate", deap.tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=3)

    def _create_player_with_modified_heuristic(self, weights):
        """
        Create a Minimax Player with a custom heuristic using the provided weights.
        
        Args:
            weights: List of weights to use in the heuristic
        
        Returns:
            Configured MinimaxPlayer
        """
        def custom_heuristic(state, player_color):
            """
            Customizable heuristic function with dynamic weights.
            This is a simplified version of the normalized_heuristic.
            """
            # Weights from the genetic algorithm
            W1, W2, W3, W4, W5 = weights
            
            # Terminal state handling
            if state.is_game_over():
                winner = state.get_winner()
                if winner == player_color:
                    return 1000.0  # Win
                elif winner is None:
                    return 0.0  # Draw
                else:
                    return -1000.0  # Loss
            
            opponent_color = player_color.opposite()
            
            # Score difference component
            score_diff = state.scores[player_color] - state.scores[opponent_color]
            score_component = W1 * score_diff
            
            # Piece progression
            progress_score = 0
            for row in range(state.BOARD_ROWS):
                for col in range(state.BOARD_COLS):
                    if state.board[row, col] == player_color.value:
                        # Calculate progress based on row and player color
                        progress = (
                            row if player_color == PlayerColor.BLACK 
                            else (state.BOARD_ROWS - 1 - row)
                        )
                        progress_score += progress
            
            progress_component = W2 * progress_score
            
            # Mobility calculation
            current_player = state.current_player
            
            state.current_player = player_color
            player_moves = state.get_valid_moves()
            
            state.current_player = opponent_color
            opponent_moves = state.get_valid_moves()
            
            state.current_player = current_player
            
            # Mobility ratio
            mobility_ratio = 0
            if len(player_moves) + len(opponent_moves) > 0:
                mobility_ratio = (len(player_moves) - len(opponent_moves)) / (len(player_moves) + len(opponent_moves))
            
            mobility_component = W3 * mobility_ratio
            
            # Jump opportunities
            jump_moves = sum(1 for m in player_moves if m.move_type == MoveType.JUMP)
            jump_component = W4 * jump_moves
            
            # Blocking score
            blocking_score = 0
            for row in range(state.BOARD_ROWS):
                for col in range(state.BOARD_COLS):
                    if state.board[row, col] == opponent_color.value:
                        # Check if piece is blocked
                        forward_row = row + opponent_color.direction
                        if 0 <= forward_row < state.BOARD_ROWS and state.board[forward_row, col] != 0:
                            blocking_score += 1
            
            blocking_component = W5 * blocking_score
            
            # Combine components
            evaluation = (
                score_component + 
                progress_component + 
                mobility_component + 
                jump_component + 
                blocking_component
            )
            
            return evaluation
        
        # Create player with custom heuristic
        return MinimaxPlayer(
            color=PlayerColor.BLACK,
            name="Genetic-Optim-Minimax",
            max_depth=self.config['max_depth'],
            use_alpha_beta=True,
            heuristic=custom_heuristic
        )

    def _evaluate_weights(self, weights):
        """
        Evaluate fitness of a set of weights by running tournament matches.
        
        Args:
            weights: List of weights to evaluate
        
        Returns:
            Tuple with fitness score (win rate)
        """
        total_wins = 0
        total_matches = len(self.opponents) * self.config['matches_per_pairing'] * 2
        
        # Create AI player with these weights
        test_ai = self._create_player_with_modified_heuristic(weights)
        
        # Run matches against each opponent
        for opponent in self.opponents:
            for _ in range(self.config['matches_per_pairing']):
                # Alternate colors for fairness
                match_results = []
                for swap_colors in [False, True]:
                    # Redirect stdout to suppress prints
                    with contextlib.redirect_stdout(io.StringIO()):
                        # Create match handler
                        match_handler = TournamentMatch(
                            "Genetic-Optim-Minimax", 
                            test_ai, 
                            opponent.name, 
                            opponent,
                            target_score=5,
                            max_turns=100  # Reduced for faster evaluation
                        )
                        
                        # Run the match
                        match_result = match_handler.run_match(swap_colors)
                        match_results.append(match_result)
                
                # Check wins
                for result in match_results:
                    if result['winner'] == "Genetic-Optim-Minimax":
                        total_wins += 1
        
        # Calculate win rate as fitness
        win_rate = total_wins / total_matches
        return (win_rate,)

    def optimize(self):
        """
        Run the genetic algorithm optimization.
        
        Returns:
            Dictionary with optimization results
        """
        # Redirect and silence output
        try:
            # Initial population
            population = self.toolbox.population(n=self.config['population_size'])
            
            # Track statistics
            stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Create logbook for tracking
            logbook = deap.tools.Logbook()
            logbook.header = "gen", "evals", "avg", "max"
            
            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Main evolutionary loop
            for gen in range(self.config['generations']):
                # Select and clone individuals
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.config['crossover_prob']:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Apply mutation
                for mutant in offspring:
                    if random.random() < self.config['mutation_prob']:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate new individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                population[:] = offspring
                
                # Record statistics
                record = stats.compile(population)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                
                # Optional: print generation stats
                print(f"Generation {gen}: Avg fitness = {record['avg']:.4f}, Max fitness = {record['max']:.4f}")
            
            # Get best individual
            best_ind = deap.tools.selBest(population, k=1)[0]
            
            # Save results
            self._save_results(best_ind, logbook)
            
            # Return results dictionary
            return {
                'best_weights': list(best_ind),
                'best_fitness': best_ind.fitness.values[0],
                'generations': self.config['generations'],
                'population_size': self.config['population_size']
            }
        
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

    def _save_results(self, best_ind, logbook):
        """
        Save optimization results to files.
        
        Args:
            best_ind: Best individual (weights)
            logbook: Evolutionary algorithm logbook
        """
        # Create results directory
        results_dir = 'optimization_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Timestamp for unique filenames
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best weights
        import numpy as np
        np.savetxt(
            os.path.join(results_dir, f'best_heuristic_weights_{timestamp}.txt'), 
            best_ind
        )
        
        # Save evolution progress
        import pandas as pd
        results_df = pd.DataFrame(
            logbook, 
            columns=['gen', 'evals', 'avg', 'max']
        )
        results_df.to_csv(
            os.path.join(results_dir, f'heuristic_weights_progress_{timestamp}.csv'), 
            index=False
        )
        
        # Create summary file
        with open(
            os.path.join(results_dir, f'heuristic_weights_summary_{timestamp}.txt'), 
            'w'
        ) as f:
            f.write("===== HEURISTIC WEIGHTS OPTIMIZATION =====\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Best Weights: {list(best_ind)}\n")
            f.write(f"Best Fitness: {best_ind.fitness.values[0]:.4f}\n")
            f.write(f"Generations: {self.config['generations']}\n")
            f.write(f"Population Size: {self.config['population_size']}\n")
            f.write(f"Crossover Probability: {self.config['crossover_prob']}\n")
            f.write(f"Mutation Probability: {self.config['mutation_prob']}\n")
            f.write("\n===== GENERATION PROGRESS =====\n")
            
            # Write generation progress to the summary file
            for record in logbook:
                f.write(
                    f"Gen {record['gen']}: "
                    f"Evals={record['evals']}, "
                    f"Avg Fitness={record['avg']:.4f}, "
                    f"Max Fitness={record['max']:.4f}\n"
                )

def main():
    """
    Run heuristic weights optimization from command line.
    
    This provides a way to directly run the optimization 
    from the terminal with configurable parameters.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Kulibrat Heuristic Weights Optimizer")
    
    parser.add_argument(
        '--population', 
        type=int, 
        default=50, 
        help='Population size for genetic algorithm'
    )
    parser.add_argument(
        '--generations', 
        type=int, 
        default=15, 
        help='Number of generations to evolve'
    )
    parser.add_argument(
        '--crossover', 
        type=float, 
        default=0.7, 
        help='Crossover probability'
    )
    parser.add_argument(
        '--mutation', 
        type=float, 
        default=0.2, 
        help='Mutation probability'
    )
    parser.add_argument(
        '--matches', 
        type=int, 
        default=10, 
        help='Matches per opponent pairing'
    )
    parser.add_argument(
        '--depth', 
        type=int, 
        default=3, 
        help='Maximum search depth for minimax'
    )
    
    args = parser.parse_args()
    
    # Configuration dictionary
    config = {
        'population_size': args.population,
        'generations': args.generations,
        'crossover_prob': args.crossover,
        'mutation_prob': args.mutation,
        'matches_per_pairing': args.matches,
        'max_depth': args.depth,
        'weight_ranges': [(0.5, 2.0)] * 5  # 5 weights with range 0.5 to 2.0
    }
    
    # Seed for reproducibility
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Create and run optimizer
        optimizer = HeuristicWeightsOptimizer(config)
        results = optimizer.optimize()
        
        # Print results
        if results:
            print("\n===== OPTIMIZATION COMPLETE =====")
            print(f"Best Weights: {results['best_weights']}")
            print(f"Best Fitness (Win Rate): {results['best_fitness']:.4f}")
            print(f"Generations: {results['generations']}")
            print(f"Population Size: {results['population_size']}")
            print("=" * 40)
        
        return 0
    
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())