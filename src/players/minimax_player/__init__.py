"""
Minimax Players package for Kulibrat game.
"""

from src.players.minimax_player.minimax_player import MinimaxPlayer
from src.players.minimax_player.heuristics import HeuristicRegistry

__all__ = ['MinimaxPlayer', 'HeuristicRegistry']