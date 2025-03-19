#!/usr/bin/env python3
"""
Genetic Algorithm Optimizer for Kulibrat AI Heuristic Weights

This script uses genetic algorithms to optimize the weights of the 
advanced_minimax_optim heuristic function for better tournament performance.

Dependencies:
- DEAP (Distributed Evolutionary Algorithms in Python)
- numpy
- pandas
"""

import random
import numpy as np
import pandas as pd
import deap.base
import deap.creator as creator
import deap.tools as tools
import deap.algorithms as algorithms

# Kulibrat imports
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.tournament.match import TournamentMatch
from src.core.player_color import PlayerColor
from src.players.ai.simple_ai_player import SimpleAIPlayer

class HeuristicWeightOptimizer:
    def __init__(
        self, 
        population_size=50, 
        generations=20, 
        crossover_prob=0.7, 
        mutation_prob=0.2,
        num_opponents=3,
        matches_per_pairing=3
    ):
        """
        Initialize the genetic algorithm optimizer.
        
        Args:
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover between individuals
            mutation_prob: Probability of mutation
            num_opponents: Number of different opponents to test against
            matches_per_pairing: Number of matches against each opponent
        """
        # Genetic algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_opponents = num_opponents
        self.matches_per_pairing = matches_per_pairing
        
        # Prepare opponents for testing
        self._prepare_opponents()
        
        # Setup DEAP framework
        self._setup_genetic_algorithm()
        
    def _prepare_opponents(self):
        """
        Prepare a list of opponent strategies to test against.
        """
        from src.players.ai.random_strategy import RandomStrategy
        from src.players.ai.minimax_strategy import MinimaxStrategy
        from src.players.ai.mcts_strategy import MCTSStrategy
        from src.players.ai.simple_ai_player import SimpleAIPlayer
        
        # Define a variety of opponents with different strategies
        self.opponents = [
            SimpleAIPlayer(PlayerColor.RED, RandomStrategy(), "Random"),
            SimpleAIPlayer(PlayerColor.RED, 
                MinimaxStrategy(
                    max_depth=4, 
                    use_alpha_beta=True, 
                    heuristic='score_diff'
                ), 
                "Minimax-Score"
            ),
            SimpleAIPlayer(PlayerColor.RED, 
                MinimaxStrategy(
                    max_depth=6, 
                    use_alpha_beta=True, 
                    heuristic='advanced'
                ), 
                "Minimax-Advanced"
            ),
            SimpleAIPlayer(PlayerColor.RED, 
                MCTSStrategy(
                    simulation_time=1.0, 
                    max_iterations=10000
                ), 
                "MCTS"
            )
        ][:self.num_opponents]
    
    def _setup_genetic_algorithm(self):
        """
        Setup the DEAP genetic algorithm framework.
        """
        # Define fitness and individual
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMax)
        
        # Toolbox for genetic operations
        self.toolbox = deap.base.Toolbox()
        
        # Attribute generator: weights between 0.5 and 2.0
        self.toolbox.register("attr_float", random.uniform, 0.5, 2.0)
        
        # Individual and population creation
        # We'll optimize 5 weights used in advanced_minimax_optim
        # W1: Score difference weight
        # W2: Progress weight
        # W3: Opponent blocking weight
        # W4: Mobility weight
        # W5: Jump opportunity weight
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
        total_matches = len(self.opponents) * self.matches_per_pairing
        
        for opponent in self.opponents:
            for _ in range(self.matches_per_pairing):
                # Create a new Minimax strategy with the current weights
                modified_heuristic = self._create_modified_heuristic(weights)
                
                # Create AI players
                test_ai = SimpleAIPlayer(
                    PlayerColor.BLACK, 
                    modified_heuristic, 
                    "Genetic-Optimized"
                )
                
                # Alternate colors for fairness
                match_results = []
                for swap_colors in [False, True]:
                    # Create match handler
                    match_handler = TournamentMatch(
                        "Genetic-Optimized", 
                        test_ai, 
                        opponent.name, 
                        opponent,
                        target_score=5,
                        max_turns=100
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
    
    def _create_modified_heuristic(self, weights):
        """
        Create a modified Minimax strategy with custom weights.
        
        Args:
            weights: List of 5 weights to use in the heuristic
        
        Returns:
            MinimaxStrategy with modified weights
        """
        def modified_advanced_minimax_optim(state, player_color):
            # Use the provided weights
            W1, W2, W3, W4, W5 = weights
            
            # This is a replica of the original advanced_minimax_optim 
            # with the weights replaced by the input weights
            
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
            
            # Progress score (similar to original implementation)
            progress_score = 0.0
            player_start_row = 0 if player_color == PlayerColor.BLACK else state.BOARD_ROWS - 1
            
            # Existing progress calculation from the original heuristic
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
            
            # Blocking score and other components remain similar
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
            
            # Mobility calculation (similar to original)
            current_player = state.current_player
            
            state.current_player = player_color
            player_moves = state.get_valid_moves()
            
            state.current_player = opponent_color
            opponent_moves = state.get_valid_moves()
            
            state.current_player = current_player
            
            # Track jump and attack opportunities
            jump_moves = sum(1 for m in player_moves if m.move_type == state.MOVE_TYPES['JUMP'])
            attack_moves = sum(1 for m in player_moves if m.move_type == state.MOVE_TYPES['ATTACK'])
            
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
        
        # Return a MinimaxStrategy with the modified heuristic
        return MinimaxStrategy(
            max_depth=6, 
            use_alpha_beta=True, 
            heuristic=modified_advanced_minimax_optim
        )
    
    def optimize(self):
        """
        Run the genetic algorithm optimization.
        
        Returns:
            Best individual (weight configuration) and its fitness
        """
        # Initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Tracking statistics
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Logging and tracking
        logbook = deap.tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        
        # Initial evaluation
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Main evolutionary loop
        for gen in range(self.generations):
            # Select individuals for next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
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
            print(logbook.stream)
        
        # Get best individual
        best_ind = tools.selBest(population, k=1)[0]
        
        # Save results
        results_df = pd.DataFrame(
            logbook, 
            columns=['gen', 'evals', 'std', 'min', 'avg', 'max']
        )
        results_df.to_csv('genetic_optimization_results.csv', index=False)
        
        print("\nBest individual:")
        print(f"Weights: {best_ind}")
        print(f"Fitness (Win Rate): {best_ind.fitness.values[0]}")
        
        return best_ind, best_ind.fitness.values[0]

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run optimizer
    optimizer = HeuristicWeightOptimizer(
        population_size=50,
        generations=20,
        crossover_prob=0.7,
        mutation_prob=0.2
    )
    
    # Run optimization
    best_weights, best_fitness = optimizer.optimize()
    
    # Optional: save best weights for future use
    np.savetxt('best_heuristic_weights.txt', best_weights)