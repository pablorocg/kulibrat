#!/usr/bin/env python3
"""
Advanced Tournament Evaluation for Kulibrat AI Players.

This script runs comprehensive tournaments between different AI strategies,
generating detailed performance metrics and visualizations.
"""

import os
import sys
import yaml
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Kulibrat core imports
from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.move_type import MoveType
from src.core.move import Move

# Player and strategy imports
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.mcts_strategy import MCTSStrategy










def main():
    """Main function to run the tournament."""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Kulibrat AI Tournament")
    parser.add_argument(
        '--config', 
        type=str, 
        default='tournament_config.yaml',
        help='Path to tournament configuration YAML file'
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Run tournament
    try:
        tournament = TournamentEvaluator(args.config)
        tournament.run_tournament()
    except Exception as e:
        logging.error(f"Tournament failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()