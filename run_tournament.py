#!/usr/bin/env python3
"""
Entry point for running Kulibrat AI tournaments.
"""

import sys
from src.tournament.runner import TournamentRunner


def main():
    """
    Run the tournament using the TournamentRunner.
    """
    sys.exit(TournamentRunner.cli())


if __name__ == "__main__":
    main()