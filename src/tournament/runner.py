"""
Tournament Runner for Kulibrat AI.
"""

import argparse
import logging
import sys
import traceback
from typing import Optional, List, Dict, Any

from src.tournament.config import TournamentConfig
from src.tournament.evaluator import TournamentEvaluator


class TournamentRunner:
    """
    Manages the execution of AI strategy tournaments.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        verbose: bool = False
    ):
        """
        Initialize tournament runner.
        
        Args:
            config_path: Path to tournament configuration file
            verbose: Enable verbose logging
        """
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        # Store configuration path
        self.config_path = config_path
        
        # Tournament results storage
        self.results: List[Dict[str, Any]] = []
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the tournament.
        
        Returns:
            List of tournament match results
        """
        try:
            # Create tournament evaluator with the config path
            if not self.config_path:
                raise ValueError("Configuration path not provided")
                
            tournament = TournamentEvaluator(self.config_path)
            
            # Run tournament
            tournament.run_tournament()
            
            # Store and return results
            self.results = tournament.tournament_results
            return self.results
        
        except Exception as e:
            self.logger.error(f"Tournament execution failed: {e}")
            if self.verbose:
                traceback.print_exc()
            raise
    
    @classmethod
    def cli(cls):
        """
        Command-line interface for running tournaments.
        """
        # Setup argument parser
        parser = argparse.ArgumentParser(description="Kulibrat AI Tournament Runner")
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
        
        # Parse arguments
        args = parser.parse_args()
        
        try:
            # Create and run tournament
            runner = cls(args.config, args.verbose)
            results = runner.run()
            
            # Optional: Add more CLI-specific result handling here
            print(f"Tournament completed. Total matches: {len(results)}")
            
            return 0
        except Exception:
            return 1


def main():
    """
    Entry point for tournament runner.
    """
    sys.exit(TournamentRunner.cli())


if __name__ == "__main__":
    main()