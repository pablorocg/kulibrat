# Enhanced logging for run_tournament.py
#!/usr/bin/env python3
"""
Entry point for running Kulibrat AI tournaments.
"""

import sys
import logging
from src.tournament.runner import TournamentRunner


def main():
    """
    Run the tournament using the TournamentRunner.
    """
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("tournament.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Kulibrat tournament")
    try:
        exit_code = TournamentRunner.cli()
        logger.info(f"Tournament completed with exit code: {exit_code}")
        return exit_code
    except Exception as e:
        logger.exception(f"Tournament failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())