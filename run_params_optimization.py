#!/usr/bin/env python3
"""
Genetic Algorithm Optimizer for Kulibrat AI Heuristic Weights
with progress tracking and silenced logging.
"""

import random
import numpy as np
import pandas as pd
import deap.base
import deap.creator
import deap.tools
import deap.algorithms
import logging
import sys
import os
from tqdm import tqdm
from datetime import datetime
import io
import contextlib

# Kulibrat imports
from src.players.minimax_player.minimax_player import MinimaxPlayer
from src.tournament.match import TournamentMatch
from src.core.player_color import PlayerColor
from src.core.move_type import MoveType

# Suppress all logging
for logger_name in ['__main__', 'src', 'pygame', 'deap']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)

# Disable pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Create a silenced MinimaxPlayer class that doesn't print anything
class SilentMinimaxPlayer(MinimaxPlayer):
    """A MinimaxPlayer that doesn't print anything to stdout."""
    
    def get_move(self, game_state):
        """Override to silence outputs."""
        # Redirect stdout to suppress prints
        with contextlib.redirect_stdout(io.StringIO()):
            # Call the parent method
            return super().get_move(game_state)
            
    def get_stats(self):
        """Override to silence outputs."""
        with contextlib.redirect_stdout(io.StringIO()):
            return super().get_stats()

class HeuristicWeightOptimizer:
    def __init__(
        self, 
        population_size=20,      
        generations=10,          
        crossover_prob=0.7, 
        mutation_prob=0.2,
        num_opponents=2,         
        matches_per_pairing=2    
    ):
        """
        Initialize the genetic algorithm optimizer with reduced parameters.
        """
        # Genetic algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_opponents = num_opponents
        self.matches_per_pairing = matches_per_pairing
        
        # Status tracking
        self.start_time = datetime.now()
        
        # Prepare opponents for testing
        self._prepare_opponents()
        
        # Setup DEAP framework
        self._setup_genetic_algorithm()
        
    def _prepare_opponents(self):
        """
        Prepare a reduced list of opponent players to test against.
        """
        from src.players.random_player.random_player import RandomPlayer
        
        # Define a smaller set of opponents with different strategies
        self.opponents = [
            # Random player
            RandomPlayer(color=PlayerColor.RED, name="Random"),
            
            # Silent Minimax player
            SilentMinimaxPlayer(
                color=PlayerColor.RED,
                name="Minimax-Score",
                max_depth=3,
                use_alpha_beta=True, 
                heuristic='score_diff'
            )
        ][:self.num_opponents]
    
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
        
        # Attribute generator: weights between 0.5 and 2.0
        self.toolbox.register("attr_float", random.uniform, 0.5, 2.0)
        
        # Individual and population creation
        self.toolbox.register(
            "individual", 
            deap.tools.initRepeat, 
            deap.creator.Individual, 
            self.toolbox.attr_float, 
            n=5
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
    
    def _evaluate_weights(self, weights):
        """
        Evaluate the fitness of a set of weights by running tournament matches.
        
        Args:
            weights: List of 5 weights to evaluate
        
        Returns:
            Tuple containing the fitness score (win rate)
        """
        total_wins = 0
        total_matches = len(self.opponents) * self.matches_per_pairing * 2  # *2 for both colors
        
        # Create AI player once with our silent wrapper
        test_ai = self._create_player_with_modified_heuristic(weights)
        
        for opponent in self.opponents:
            for _ in range(self.matches_per_pairing):
                # Alternate colors for fairness
                match_results = []
                for swap_colors in [False, True]:
                    # Redirect stdout during match execution to capture all prints
                    with contextlib.redirect_stdout(io.StringIO()):
                        # Create match handler
                        match_handler = TournamentMatch(
                            "Genetic-Optimized", 
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
                    if result['winner'] == "Genetic-Optimized":
                        total_wins += 1
        
        # Calculate win rate as fitness
        win_rate = total_wins / total_matches
        return (win_rate,)
    
    def _create_player_with_modified_heuristic(self, weights):
        """
        Create a SilentMinimaxPlayer with modified weights for the heuristic.
        """
        def modified_advanced_minimax_optim(state, player_color):
            # Use the provided weights
            W1, W2, W3, W4, W5 = weights
            
            # Check for terminal state
            if state.is_game_over():
                winner = state.get_winner()
                if winner == player_color:
                    return 1000.0  # Win
                elif winner is None:
                    return 0.0  # Draw
                else:
                    return -1000.0  # Loss

            opponent_color = player_color.opposite()
            
            # Score difference
            score_diff = state.scores[player_color] - state.scores[opponent_color]
            score_component = W1 * score_diff
            
            # Progress score
            progress_score = 0.0
            player_start_row = 0 if player_color == PlayerColor.BLACK else state.BOARD_ROWS - 1
            
            for row in range(state.BOARD_ROWS):
                for col in range(state.BOARD_COLS):
                    if state.board[row, col] == player_color.value:
                        if player_color == PlayerColor.BLACK:
                            progress = row
                        else:
                            progress = state.BOARD_ROWS - 1 - row
                        
                        if (player_color == PlayerColor.BLACK and row >= state.BOARD_ROWS // 2) or \
                           (player_color == PlayerColor.RED and row < state.BOARD_ROWS // 2):
                            progress *= 1.2
                        
                        if row == (0 if player_color == PlayerColor.RED else state.BOARD_ROWS - 1):
                            progress *= 2.5
                            
                        progress_score += progress
                    
                    # Subtract opponent progress
                    elif state.board[row, col] == opponent_color.value:
                        if opponent_color == PlayerColor.BLACK:
                            opponent_progress = row
                        else:
                            opponent_progress = state.BOARD_ROWS - 1 - row
                            
                        progress_score -= opponent_progress * 0.7
            
            progress_component = W2 * progress_score
            
            # Blocking score
            blocking_score = 0.0
            
            for row in range(state.BOARD_ROWS):
                for col in range(state.BOARD_COLS):
                    if state.board[row, col] == opponent_color.value:
                        # Check if piece is blocked
                        forward_row = row + opponent_color.direction
                        if 0 <= forward_row < state.BOARD_ROWS and state.board[forward_row, col] != 0:
                            blocking_score += 1.2
                        
                        # Check diagonal blockage
                        for diagonal_col in [col-1, col+1]:
                            if 0 <= diagonal_col < state.BOARD_COLS:
                                diagonal_row = row + opponent_color.direction
                                if 0 <= diagonal_row < state.BOARD_ROWS and state.board[diagonal_row, diagonal_col] != 0:
                                    blocking_score += 0.6
            
            blocking_component = W3 * blocking_score
            
            # Mobility calculation
            current_player = state.current_player
            
            state.current_player = player_color
            player_moves = state.get_valid_moves()
            
            state.current_player = opponent_color
            opponent_moves = state.get_valid_moves()
            
            state.current_player = current_player
            
            # Track jump and attack opportunities
            jump_moves = sum(1 for m in player_moves if m.move_type == MoveType.JUMP)
            attack_moves = sum(1 for m in player_moves if m.move_type == MoveType.ATTACK)
            
            # Mobility ratio
            mobility_ratio = 0
            if len(player_moves) + len(opponent_moves) > 0:
                mobility_ratio = (len(player_moves) - len(opponent_moves)) / (len(player_moves) + len(opponent_moves))
            
            mobility_component = W4 * mobility_ratio * 10
            
            # Jump opportunity component
            jump_component = W5 * jump_moves
            
            # Combine components
            evaluation = (
                score_component + 
                progress_component + 
                blocking_component + 
                mobility_component + 
                jump_component
            )
            
            return evaluation
        
        # Return a SilentMinimaxPlayer with the modified heuristic
        return SilentMinimaxPlayer(
            color=PlayerColor.BLACK, 
            name="Genetic-Optimized",
            max_depth=3,  # Reduced depth for faster evaluation
            use_alpha_beta=True, 
            heuristic=modified_advanced_minimax_optim
        )
    
    def _estimate_total_matches(self):
        """Calculate total matches to be played for progress tracking."""
        matches_per_eval = len(self.opponents) * self.matches_per_pairing * 2  # *2 for both colors
        initial_evals = self.population_size
        re_evals_per_gen = int(self.population_size * 0.8)
        total_evals = initial_evals + re_evals_per_gen * (self.generations - 1)
        total_matches = total_evals * matches_per_eval
        return total_matches, matches_per_eval
    
    def optimize(self):
        """
        Run the genetic algorithm optimization with progress tracking.
        
        Returns:
            Best individual (weight configuration) and its fitness
        """
        # Show optimization settings
        total_matches, matches_per_eval = self._estimate_total_matches()
        print("\n===== KULIBRAT HEURISTIC WEIGHT OPTIMIZER =====")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Opponents: {self.num_opponents}")
        print(f"Matches per pairing: {self.matches_per_pairing}")
        print(f"Estimated total matches: {total_matches}")
        print("=" * 49 + "\n")
        
        # Set up progress tracking
        match_counter = 0
        match_progress = tqdm(
            total=total_matches, 
            desc="Total progress", 
            unit="matches",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Initial population
        print("Generating initial population...")
        population = self.toolbox.population(n=self.population_size)
        
        # Tracking statistics
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Logging and tracking
        logbook = deap.tools.Logbook()
        logbook.header = "gen", "evals", "avg", "max"
        
        # Create a fitness evaluation wrapper to update progress
        def evaluate_with_progress(ind):
            nonlocal match_counter
            result = self.toolbox.evaluate(ind)
            match_counter += matches_per_eval
            match_progress.update(matches_per_eval)
            return result
        
        # Initial evaluation with progress tracking
        print("Evaluating initial population...")
        fitnesses = list(map(evaluate_with_progress, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Record initial statistics
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        
        # Print generation info
        gen_info = logbook[0]
        print(f"Generation 0: Avg fitness = {gen_info['avg']:.4f}, Max fitness = {gen_info['max']:.4f}")
        
        # Main evolutionary loop
        for gen in range(1, self.generations):
            gen_start = datetime.now()
            print(f"\nGeneration {gen}/{self.generations-1} started...")
            
            # Select individuals for next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            
            # Create progress bar specifically for this generation
            gen_progress = tqdm(
                total=len(invalid_ind), 
                desc=f"Gen {gen}", 
                unit="ind",
                position=1,
                leave=False
            )
            
            # Evaluate individuals with progress tracking
            for i, ind in enumerate(invalid_ind):
                ind.fitness.values = evaluate_with_progress(ind)
                gen_progress.update(1)
            
            gen_progress.close()
            
            # Replace population
            population[:] = offspring
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            
            # Print generation stats and time
            gen_info = logbook[gen]
            gen_time = datetime.now() - gen_start
            print(f"Generation {gen}: Avg fitness = {gen_info['avg']:.4f}, Max fitness = {gen_info['max']:.4f}")
            print(f"Generation time: {gen_time}")
            
            # Checkpoint: Save current best weights to a file
            best_in_gen = deap.tools.selBest(population, k=1)[0]
            np.savetxt(f'checkpoint_gen_{gen}_weights.txt', best_in_gen)
            
        match_progress.close()
        
        # Get best individual
        best_ind = deap.tools.selBest(population, k=1)[0]
        
        # Save results in multiple formats
        self._save_results(best_ind, logbook)
        
        # Show final results
        total_time = datetime.now() - self.start_time
        print("\n===== OPTIMIZATION COMPLETE =====")
        print(f"Total time: {total_time}")
        print(f"Total matches played: {match_counter}")
        print(f"Best weights: {[round(w, 4) for w in best_ind]}")
        print(f"Best fitness (win rate): {best_ind.fitness.values[0]:.4f}")
        print("=" * 35)
        
        return best_ind, best_ind.fitness.values[0]
    
    def _save_results(self, best_ind, logbook):
        """Save optimization results in multiple formats."""
        try:
            # Create results directory if it doesn't exist
            os.makedirs("optimization_results", exist_ok=True)
            
            # Save timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save best weights to text file
            np.savetxt(f'optimization_results/best_weights_{timestamp}.txt', best_ind)
            
            # Save evolution progress to CSV
            results_df = pd.DataFrame(
                logbook, 
                columns=['gen', 'evals', 'avg', 'max']
            )
            results_df.to_csv(f'optimization_results/evolution_progress_{timestamp}.csv', index=False)
            
            # Save all information to a summary file
            with open(f'optimization_results/optimization_summary_{timestamp}.txt', 'w') as f:
                # Write parameters
                f.write("===== OPTIMIZATION PARAMETERS =====\n")
                f.write(f"Start time: {self.start_time}\n")
                f.write(f"End time: {datetime.now()}\n")
                f.write(f"Population size: {self.population_size}\n")
                f.write(f"Generations: {self.generations}\n")
                f.write(f"Crossover probability: {self.crossover_prob}\n")
                f.write(f"Mutation probability: {self.mutation_prob}\n")
                f.write(f"Number of opponents: {self.num_opponents}\n")
                f.write(f"Matches per pairing: {self.matches_per_pairing}\n\n")
                
                # Write results
                f.write("===== OPTIMIZATION RESULTS =====\n")
                f.write(f"Best weights: {[round(w, 4) for w in best_ind]}\n")
                f.write(f"Best fitness (win rate): {best_ind.fitness.values[0]:.4f}\n\n")
                
                # Write evolution progress
                f.write("===== EVOLUTION PROGRESS =====\n")
                for gen_data in logbook:
                    f.write(f"Gen {gen_data['gen']}: ")
                    f.write(f"Evals={gen_data['evals']}, ")
                    f.write(f"Avg={gen_data['avg']:.4f}, ")
                    f.write(f"Max={gen_data['max']:.4f}\n")
        
            print(f"All results saved to 'optimization_results' directory with timestamp {timestamp}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Disable all print output from libraries
        with contextlib.redirect_stdout(io.StringIO()):
            # Redirect only during initialization to suppress initial logging
            pass
            
        # Create and run optimizer with reduced parameters
        optimizer = HeuristicWeightOptimizer(
            population_size=200,       # Reduced from 50
            generations=10,           # Reduced from 20
            crossover_prob=0.7, 
            mutation_prob=0.2,
            num_opponents=2,          # Reduced from 3
            matches_per_pairing=2     # Reduced from 3
        )
        
        # Run optimization
        best_weights, best_fitness = optimizer.optimize()
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()