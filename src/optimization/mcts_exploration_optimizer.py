#!/usr/bin/env python3
"""
Optimization of MCTS Exploration Weight for Kulibrat AI.
"""

import random
import numpy as np
import logging
import os
import sys
import contextlib
import io
import pandas as pd

import deap.base
import deap.creator
import deap.tools
import deap.algorithms

from src.tournament.match import TournamentMatch
from src.players.mcts_player.mcts_player import MCTSPlayer
from src.core.player_color import PlayerColor
from src.tournament.runner import TournamentRunner

class MCTSExplorationOptimizer:
    def __init__(
        self, 
        config=None,
        population_size=30,
        generations=15,
        exploration_range=(0.5, 3.0),
        matches_per_pairing=10,
        max_iterations=15000,
        simulation_time=2.5
    ):
        """
        Initialize MCTS Exploration Weight Optimizer.
        
        Args:
            config: Configuration dictionary
            population_size: Number of exploration weights to test per generation
            generations: Number of generations to evolve
            exploration_range: Range for exploration weight
            matches_per_pairing: Matches to play for each weight configuration
            max_iterations: Maximum MCTS iterations per move
            simulation_time: Simulation time for MCTS
        """
        # Default configuration
        self.config = {
            'population_size': population_size,
            'generations': generations,
            'exploration_range': exploration_range,
            'matches_per_pairing': matches_per_pairing,
            'max_iterations': max_iterations,
            'simulation_time': simulation_time
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
        """Suppress logging output during optimization, but less aggressively."""
        # Reduce logging level instead of completely silencing
        logging.getLogger('src').setLevel(logging.WARNING)
        logging.getLogger('deap').setLevel(logging.WARNING)
        
        # Optional: Redirect only error output
        # sys.stderr = open('optimization_errors.log', 'w')

    def _prepare_opponents(self):
        """
        Prepare a set of opponent players to test against.
        """
        from src.players.random_player.random_player import RandomPlayer
        from src.players.minimax_player.minimax_player import MinimaxPlayer

        # Define opponents with different strategies
        self.opponents = [
            # Random player
            RandomPlayer(color=PlayerColor.RED, name="Random"),
            
            # Minimax player
            MinimaxPlayer(
                color=PlayerColor.RED, 
                name="Minimax",
                max_depth=3,
                use_alpha_beta=True
            )
        ]

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
        
        # Get exploration weight range from config
        min_val, max_val = self.config['exploration_range']
        
        # Attribute generator for exploration weight
        self.toolbox.register(
            "attr_weight", 
            random.uniform, 
            min_val, 
            max_val
        )
        
        # Individual and population creation
        self.toolbox.register(
            "individual", 
            deap.tools.initRepeat, 
            deap.creator.Individual, 
            self.toolbox.attr_weight, 
            n=1
        )
        
        self.toolbox.register(
            "population", 
            deap.tools.initRepeat, 
            list, 
            self.toolbox.individual
        )
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_exploration_weight)
        self.toolbox.register("mate", deap.tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", deap.tools.selTournament, tournsize=3)

    def _create_mcts_player(self, exploration_weight):
        """
        Create an MCTS player with a specific exploration weight.
        
        Args:
            exploration_weight: Exploration weight to use
        
        Returns:
            Configured MCTSPlayer
        """
        return MCTSPlayer(
            color=PlayerColor.BLACK,
            name=f"MCTS-{exploration_weight:.2f}",
            simulation_time=self.config['simulation_time'],
            max_iterations=self.config['max_iterations'],
            exploration_weight=exploration_weight
        )

    def _evaluate_exploration_weight(self, individual):
        """
        Evaluate the fitness of an exploration weight.
        
        Args:
            individual: List containing the exploration weight
        
        Returns:
            Tuple with fitness score (win rate)
        """
        exploration_weight = individual[0]
        total_wins = 0
        total_matches = len(self.opponents) * self.config['matches_per_pairing'] * 2
        
        # Prepare results list for manual DataFrame creation
        match_results = []
        
        # Create MCTS player with this exploration weight
        test_ai = self._create_mcts_player(exploration_weight)
        
        print(f"\n--- Evaluating Exploration Weight: {exploration_weight:.4f} ---")
        
        # Run matches against each opponent
        for opponent in self.opponents:
            opponent_wins = 0
            for match_num in range(self.config['matches_per_pairing']):
                # Alternate colors for fairness
                for swap_colors in [False, True]:
                    try:
                        # Create match handler
                        match_handler = TournamentMatch(
                            "MCTS-Exploration", 
                            test_ai, 
                            opponent.name, 
                            opponent,
                            target_score=5,
                            max_turns=100  # Reduced for faster evaluation
                        )
                        
                        # Run the match
                        match_result = match_handler.run_match(swap_colors)
                        
                        # Add match result with required columns
                        result_dict = {
                            'player1': "MCTS-Exploration",
                            'player2': opponent.name,
                            'winner': match_result.get('winner'),
                            'score_p1': match_result.get('score_p1', 0),
                            'score_p2': match_result.get('score_p2', 0),
                            'turns': match_result.get('turns', 0),
                            'player1_color': match_result.get('player1_color', ''),
                            'player2_color': match_result.get('player2_color', '')
                        }
                        match_results.append(result_dict)
                        
                        # Check for wins and print details
                        if result_dict['winner'] == "MCTS-Exploration":
                            total_wins += 1
                            opponent_wins += 1
                            print(f"  Win against {opponent.name} (Swap: {swap_colors})")
                        else:
                            print(f"  Loss against {opponent.name} (Swap: {swap_colors})")
                    
                    except Exception as e:
                        print(f"Error in match against {opponent.name}: {e}")
            
            # Print summary for this opponent
            print(f"  Wins against {opponent.name}: {opponent_wins}/{self.config['matches_per_pairing']*2}")
        
        # Calculate win rate as fitness
        win_rate = total_wins / total_matches if total_matches > 0 else 0
        
        print(f"Total Wins: {total_wins}/{total_matches}")
        print(f"Win Rate: {win_rate:.4f}")
        
        # Optional: Create DataFrame to verify column structure
        try:
            match_df = pd.DataFrame(match_results)
            match_df.to_csv(f'mcts_matches_weight_{exploration_weight:.4f}.csv', index=False)
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
        
        return (win_rate,)

    def optimize(self):
        """
        Run the genetic algorithm optimization.
        
        Returns:
            Dictionary with optimization results
        """
        # Seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
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
            print("\n===== STARTING MCTS EXPLORATION WEIGHT OPTIMIZATION =====")
            print(f"Population Size: {self.config['population_size']}")
            print(f"Exploration Range: {self.config['exploration_range']}")
            print(f"Matches per Pairing: {self.config['matches_per_pairing']}")
            print(f"Generations: {self.config['generations']}\n")
            
            for gen in range(self.config['generations']):
                print(f"\n--- Generation {gen}/{self.config['generations']-1} ---")
                
                # Select and clone individuals
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:  # Crossover probability
                        print(f"  Crossover: {child1[0]:.4f} ↔ {child2[0]:.4f}")
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Apply mutation
                for mutant in offspring:
                    if random.random() < 0.2:  # Mutation probability
                        old_weight = mutant[0]
                        self.toolbox.mutate(mutant)
                        print(f"  Mutation: {old_weight:.4f} → {mutant[0]:.4f}")
                        del mutant.fitness.values
                
                # Evaluate new individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                print(f"  Evaluating {len(invalid_ind)} new individuals...")
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Replace population
                population[:] = offspring
                
                # Record statistics
                record = stats.compile(population)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                
                # Print generation stats
                print(f"  Gen {gen} Stats:")
                print(f"    Avg Fitness: {record['avg']:.4f}")
                print(f"    Max Fitness: {record['max']:.4f}")
                print(f"    Std Dev: {record['std']:.4f}")
                
                # Display best individuals in this generation
                best_inds = deap.tools.selBest(population, k=3)
                print("  Top 3 Individuals:")
                for i, ind in enumerate(best_inds, 1):
                    print(f"    {i}. Weight: {ind[0]:.4f}, Fitness: {ind.fitness.values[0]:.4f}")
            
            # Get best individuals
            best_ind = deap.tools.selBest(population, k=1)[0]
            top_3_inds = deap.tools.selBest(population, k=3)
            
            # Save results
            self._save_results(best_ind, logbook)
            
            # Final summary
            print("\n===== OPTIMIZATION COMPLETE =====")
            print("Top 3 Exploration Weights:")
            for i, ind in enumerate(top_3_inds, 1):
                print(f"  {i}. Weight: {ind[0]:.4f}, Fitness: {ind.fitness.values[0]:.4f}")
            
            # Additional statistics from the final population
            final_weights = [ind[0] for ind in population]
            final_fitness = [ind.fitness.values[0] for ind in population]
            
            print("\nFinal Population Statistics:")
            print(f"  Mean Exploration Weight: {np.mean(final_weights):.4f}")
            print(f"  Std Dev Exploration Weight: {np.std(final_weights):.4f}")
            print(f"  Mean Fitness: {np.mean(final_fitness):.4f}")
            print(f"  Std Dev Fitness: {np.std(final_fitness):.4f}")
            
            # Return results dictionary with more information
            return {
                'best_exploration_weight': best_ind[0],
                'fitness': best_ind.fitness.values[0],
                'generations': self.config['generations'],
                'population_size': self.config['population_size'],
                'top_3_weights': [ind[0] for ind in top_3_inds],
                'top_3_fitness': [ind.fitness.values[0] for ind in top_3_inds],
                'mean_weight': np.mean(final_weights),
                'std_weight': np.std(final_weights),
                'mean_fitness': np.mean(final_fitness),
                'std_fitness': np.std(final_fitness)
            }
        
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_results(self, best_ind, logbook):
        """
        Save optimization results to files.
        
        Args:
            best_ind: Best individual (exploration weight)
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
            os.path.join(results_dir, f'best_mcts_exploration_weight_{timestamp}.txt'), 
            best_ind
        )
        
        # Save evolution progress
        import pandas as pd
        results_df = pd.DataFrame(
            logbook, 
            columns=['gen', 'evals', 'avg', 'max']
        )
        results_df.to_csv(
            os.path.join(results_dir, f'mcts_exploration_progress_{timestamp}.csv'), 
            index=False
        )
        
        # Create summary file
        with open(
            os.path.join(results_dir, f'mcts_exploration_summary_{timestamp}.txt'), 
            'w'
        ) as f:
            f.write("===== MCTS EXPLORATION WEIGHT OPTIMIZATION =====\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Best Exploration Weight: {best_ind[0]:.4f}\n")
            f.write(f"Best Fitness: {best_ind.fitness.values[0]:.4f}\n")
            f.write(f"Generations: {self.config['generations']}\n")
            f.write(f"Population Size: {self.config['population_size']}\n")
            f.write(f"Exploration Range: {self.config['exploration_range']}\n")
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
    Run MCTS exploration weight optimization from command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Kulibrat MCTS Exploration Weight Optimizer")
    
    parser.add_argument(
        '--population', 
        type=int, 
        default=30, 
        help='Population size for genetic algorithm'
    )
    parser.add_argument(
        '--generations', 
        type=int, 
        default=15, 
        help='Number of generations to evolve'
    )
    parser.add_argument(
        '--min-weight', 
        type=float, 
        default=0.5, 
        help='Minimum exploration weight'
    )
    parser.add_argument(
        '--max-weight', 
        type=float, 
        default=3.0, 
        help='Maximum exploration weight'
    )
    parser.add_argument(
        '--matches', 
        type=int, 
        default=10, 
        help='Matches per opponent pairing'
    )
    parser.add_argument(
        '--max-iterations', 
        type=int, 
        default=15000, 
        help='Maximum MCTS iterations per move'
    )
    parser.add_argument(
        '--simulation-time', 
        type=float, 
        default=2.5, 
        help='MCTS simulation time per move'
    )
    
    args = parser.parse_args()
    
    # Configuration dictionary
    config = {
        'population_size': args.population,
        'generations': args.generations,
        'exploration_range': (args.min_weight, args.max_weight),
        'matches_per_pairing': args.matches,
        'max_iterations': args.max_iterations,
        'simulation_time': args.simulation_time
    }
    
    # Seed for reproducibility
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Restore standard output
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        # Create and run optimizer
        optimizer = MCTSExplorationOptimizer(config)
        results = optimizer.optimize()
        
        # Print results
        if results:
            print("\n===== OPTIMIZATION COMPLETE =====")
            print(f"Best Exploration Weight: {results['best_exploration_weight']:.4f}")
            print(f"Best Fitness (Win Rate): {results['fitness']:.4f}")
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