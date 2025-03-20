#!/usr/bin/env python3
"""
Main entry point for running optimization experiments for Kulibrat AI.
"""

import argparse
import sys
from typing import List

from src.optimization.mcts_exploration_optimizer import MCTSExplorationOptimizer
from src.optimization.heuristic_weights_optimizer import HeuristicWeightsOptimizer

def main():
    """
    Run optimization experiments based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Kulibrat AI Optimization Experiments")
    parser.add_argument(
        "--mode", 
        choices=["mcts", "heuristic", "both"], 
        default="both", 
        help="Optimization mode"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="optimization_config.yaml", 
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run selected optimization
    try:
        if args.mode in ["mcts", "both"]:
            mcts_optimizer = MCTSExplorationOptimizer(config.get('mcts', {}))
            mcts_results = mcts_optimizer.optimize()
            print("MCTS Exploration Weight Optimization Results:", mcts_results)
        
        if args.mode in ["heuristic", "both"]:
            heuristic_optimizer = HeuristicWeightsOptimizer(config.get('heuristic', {}))
            heuristic_results = heuristic_optimizer.optimize()
            print("Heuristic Weights Optimization Results:", heuristic_results)
        
        return 0
    
    except Exception as e:
        print(f"Optimization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())