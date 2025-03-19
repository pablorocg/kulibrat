"""
Tournament module for evaluating Kulibrat AI strategies.
"""

from src.tournament.evaluator import TournamentEvaluator
from src.tournament.runner import run_tournament

__all__ = ["TournamentEvaluator", "run_tournament"]