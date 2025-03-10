"""
Minimax strategy implementation for Kulibrat AI.
"""

import random
import time
from typing import Optional, Tuple

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy


class MinimaxStrategy(AIStrategy):
    """AI strategy using the minimax algorithm with optional alpha-beta pruning."""

    def __init__(self, max_depth: int = 5, use_alpha_beta: bool = True):
        """
        Initialize the minimax strategy.

        Args:
            max_depth: Maximum search depth
            use_alpha_beta: Whether to use alpha-beta pruning
        """
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_evaluated = 0

    def select_move(
        self, game_state: GameState, player_color: PlayerColor
    ) -> Optional[Move]:
        """
        Select the best move using minimax.

        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move

        Returns:
            The best move according to minimax
        """
        start_time = time.time()
        self.nodes_evaluated = 0

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None

        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]

        best_score = float("-inf")
        best_moves = []

        for move in valid_moves:
            # Apply the move to a copy of the state
            next_state = game_state.copy()
            next_state.apply_move(move)

            # Run minimax
            if self.use_alpha_beta:
                score, _ = self._alpha_beta(
                    state=next_state,
                    depth=self.max_depth - 1,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    current_player=player_color.opposite(),
                    maximizing_player=player_color,
                )
            else:
                score, _ = self._minimax(
                    state=next_state,
                    depth=self.max_depth - 1,
                    current_player=player_color.opposite(),
                    maximizing_player=player_color,
                )

            # Track best moves
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        # Choose randomly from equally good moves for variety
        selected_move = random.choice(best_moves)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Minimax evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s")
        print(f"Selected move: {selected_move} with score: {best_score}")

        return selected_move

    def _minimax(
        self,
        state: GameState,
        depth: int,
        current_player: PlayerColor,
        maximizing_player: PlayerColor,
    ) -> Tuple[float, Optional[Move]]:
        """
        Minimax algorithm without alpha-beta pruning.

        Args:
            state: Current game state
            depth: Remaining search depth
            current_player: Player making the move at this level
            maximizing_player: The AI player we're finding the best move for

        Returns:
            Tuple of (score, move)
        """
        self.nodes_evaluated += 1

        # Terminal conditions
        if state.is_game_over() or depth == 0:
            return self._evaluate(state, maximizing_player), None

        valid_moves = state.get_valid_moves()

        # No valid moves means the other player gets an extra turn
        if not valid_moves:
            return self._minimax(
                state, depth - 1, current_player.opposite(), maximizing_player
            )

        is_maximizing = current_player == maximizing_player
        best_score = float("-inf") if is_maximizing else float("inf")
        best_move = None

        for move in valid_moves:
            # Apply the move to a copy of the state
            next_state = state.copy()
            next_state.apply_move(move)

            # Recursively evaluate
            score, _ = self._minimax(
                next_state, depth - 1, next_state.current_player, maximizing_player
            )

            # Update best score
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

    def _alpha_beta(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        current_player: PlayerColor,
        maximizing_player: PlayerColor,
    ) -> Tuple[float, Optional[Move]]:
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            current_player: Player making the move at this level
            maximizing_player: The AI player we're finding the best move for

        Returns:
            Tuple of (score, move)
        """
        self.nodes_evaluated += 1

        # Terminal conditions
        if state.is_game_over() or depth == 0:
            return self._evaluate(state, maximizing_player), None

        valid_moves = state.get_valid_moves()

        # No valid moves means the other player gets an extra turn
        if not valid_moves:
            return self._alpha_beta(
                state,
                depth - 1,
                alpha,
                beta,
                current_player.opposite(),
                maximizing_player,
            )

        is_maximizing = current_player == maximizing_player
        best_move = None

        if is_maximizing:
            best_score = float("-inf")

            for move in valid_moves:
                # Apply the move to a copy of the state
                next_state = state.copy()
                next_state.apply_move(move)

                # Recursively evaluate
                score, _ = self._alpha_beta(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    next_state.current_player,
                    maximizing_player,
                )

                # Update best score
                if score > best_score:
                    best_score = score
                    best_move = move

                # Update alpha
                alpha = max(alpha, best_score)

                # Alpha-beta pruning
                if beta <= alpha:
                    break

            return best_score, best_move
        else:
            best_score = float("inf")

            for move in valid_moves:
                # Apply the move to a copy of the state
                next_state = state.copy()
                next_state.apply_move(move)

                # Recursively evaluate
                score, _ = self._alpha_beta(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    next_state.current_player,
                    maximizing_player,
                )

                # Update best score
                if score < best_score:
                    best_score = score
                    best_move = move

                # Update beta
                beta = min(beta, best_score)

                # Alpha-beta pruning
                if beta <= alpha:
                    break

            return best_score, best_move

    # def _evaluate(self, state: GameState, player_color: PlayerColor) -> float:
    #     """
    #     Heuristic: Evaluate the state for the given player.

    #     Args:
    #         state: Game state to evaluate
    #         player_color: Player to evaluate the state for

    #     Returns:
    #         Score representing how good the state is for the player
    #     """
    #     # Check if the game is over
    #     if state.is_game_over():
    #         winner = state.get_winner()
    #         if winner == player_color:
    #             return 1000.0  # Win
    #         elif winner is None:
    #             return 0.0  # Draw
    #         else:
    #             return -1000.0  # Loss

    #     opponent_color = player_color.opposite()

    #     # Calculate score based on piece positions and scores
    #     score = 0.0

    #     # Score difference
    #     score += 100.0 * (state.scores[player_color] - state.scores[opponent_color])

    #     # Piece advantage
    #     pieces_on_board_player = (
    #         4 - state.pieces_off_board[player_color] + state.scores[player_color]
    #     )
    #     pieces_on_board_opponent = (
    #         4 - state.pieces_off_board[opponent_color] + state.scores[opponent_color]
    #     )
    #     score += 10.0 * (pieces_on_board_player - pieces_on_board_opponent)

    #     # Piece advancement (pieces closer to scoring)
    #     player_advancement = 0
    #     opponent_advancement = 0

    #     # Calculate advancement for both players
    #     for row in range(4):
    #         for col in range(3):
    #             piece = state.board[row, col]
    #             if piece == player_color.value:
    #                 # For BLACK, advancement is measured by how far down the board they are
    #                 # For RED, advancement is measured by how far up the board they are
    #                 if player_color == PlayerColor.BLACK:
    #                     player_advancement += row  # 0 at top, 3 at bottom
    #                 else:
    #                     player_advancement += 3 - row  # 3 at top, 0 at bottom
    #             elif piece == opponent_color.value:
    #                 if opponent_color == PlayerColor.BLACK:
    #                     opponent_advancement += row
    #                 else:
    #                     opponent_advancement += 3 - row

    #     score += 5.0 * (player_advancement - opponent_advancement)

    #     # Number of valid moves (mobility)
    #     current_player = state.current_player
    #     state.current_player = player_color
    #     player_moves = len(state.get_valid_moves())
    #     state.current_player = opponent_color
    #     opponent_moves = len(state.get_valid_moves())
    #     state.current_player = current_player  # Restore original current player

    #     score += 2.0 * (player_moves - opponent_moves)

    #     return score
    def _evaluate(self, state: GameState, player_color: PlayerColor) -> float:
        """
        Heuristic: Evaluate the state for the given player.
        Args:
            state: Game state to evaluate
            player_color: Player to evaluate the state for
        Returns:
            Score representing how good the state is for the player
        """
        # Determinar el color del oponente
        opponent_color = player_color.opposite()
        
        # Dimensiones del tablero
        ROWS, COLS = 4, 3
        
        # Filas iniciales según el color del jugador
        player_start = 0 if player_color == PlayerColor.BLACK else ROWS - 1
        opponent_start = ROWS - 1 if player_color == PlayerColor.BLACK else 0
        
        # Dirección de movimiento (1 para BLACK hacia abajo, -1 para RED hacia arriba)
        direction = 1 if player_color == PlayerColor.BLACK else -1
        
        # 1. Diferencia de puntuación (factor más importante)
        score_diff = state.scores[player_color] - state.scores[opponent_color]
        evaluation = score_diff * 10.0  # Alta ponderación a la diferencia de puntuación
        
        # Variables para el análisis
        player_pieces = []
        opponent_pieces = []
        player_count = 0
        opponent_count = 0
        advancement_score = 0
        scoring_positions = 0
        
        # Analizar el tablero
        for row in range(ROWS):
            for col in range(COLS):
                piece = state.board[row, col]
                
                if piece == player_color:
                    player_pieces.append((row, col))
                    player_count += 1
                    
                    # Progresión de la pieza (valor por distancia desde su inicio)
                    progress = abs(row - player_start)
                    advancement_score += progress * 1.5
                    
                    # Posiciones a punto de anotar (en la fila inicial del oponente)
                    if row == opponent_start:
                        scoring_positions += 1
                        advancement_score += 2.0
                    
                    # Bonus por control de columna central (mayor flexibilidad)
                    if col == 1:
                        advancement_score += 0.7
                
                elif piece == opponent_color:
                    opponent_pieces.append((row, col))
                    opponent_count += 1
                    
                    # Penalizar piezas del oponente cerca de anotar
                    if row == player_start:
                        advancement_score -= 2.0
        
        # Añadir puntuación por avance y posicionamiento
        evaluation += advancement_score
        evaluation += scoring_positions * 3.0  # Alta ponderación a posiciones de anotación
        
        # 2. Ventaja en número de piezas
        piece_advantage = player_count - opponent_count
        evaluation += piece_advantage * 0.8
        
        # 3. Movilidad y situaciones especiales
        try:
            player_moves = state.get_valid_moves()
            
            # Cambiar temporalmente al jugador oponente para evaluar sus movimientos
            temp_state = state.copy()
            temp_state.current_player = opponent_color
            opponent_moves = temp_state.get_valid_moves()
            
            # Diferencia de movilidad
            mobility_diff = len(player_moves) - len(opponent_moves)
            evaluation += mobility_diff * 0.5
            
            # Situaciones de bloqueo (muy importantes)
            if len(player_moves) == 0 and len(opponent_moves) > 0:
                evaluation -= 8.0  # Gran penalización por estar bloqueado
            elif len(opponent_moves) == 0 and len(player_moves) > 0:
                evaluation += 8.0  # Gran bonus por bloquear al oponente
        except:
            # Si hay errores al obtener movimientos, continuamos sin esta parte
            pass
        
        # 4. Analizar oportunidades específicas
        attack_opportunities = 0
        jump_opportunities = 0
        scoring_jump_opportunities = 0
        
        # Contar oportunidades potenciales de ataque
        for p_row, p_col in player_pieces:
            next_row = p_row + direction
            if 0 <= next_row < ROWS:
                if state.board[next_row, p_col] == opponent_color:
                    attack_opportunities += 1
        
        # Contar oportunidades potenciales de salto
        for p_row, p_col in player_pieces:
            for jump_len in range(1, 4):  # Puede saltar sobre 1-3 piezas
                # Verificar que hay una línea de piezas del oponente
                line_valid = True
                for i in range(1, jump_len + 1):
                    check_row = p_row + direction * i
                    if not (0 <= check_row < ROWS) or state.board[check_row, p_col] != opponent_color:
                        line_valid = False
                        break
                
                if line_valid:
                    landing_row = p_row + direction * (jump_len + 1)
                    # Verificar si el aterrizaje está fuera del tablero (puntuación) o en casilla vacía
                    if landing_row < 0 or landing_row >= ROWS:
                        scoring_jump_opportunities += 1
                    elif state.board[landing_row, p_col] is None:
                        jump_opportunities += 1
        
        # Añadir puntuación por oportunidades tácticas
        evaluation += attack_opportunities * 1.2
        evaluation += jump_opportunities * 1.0
        evaluation += scoring_jump_opportunities * 2.5  # Mayor valor a saltos que puntúan
        
        # 5. Detectar patrones estratégicos para ciclos ganadores
        player_on_opponent_side = sum(1 for r, c in player_pieces if (r - opponent_start) * direction >= 0)
        opponent_on_player_side = sum(1 for r, c in opponent_pieces if (r - player_start) * direction <= 0)
        
        # Ventaja potencial para ciclos si tenemos más piezas en el lado enemigo
        if player_on_opponent_side >= 2 and player_on_opponent_side > opponent_on_player_side:
            evaluation += 3.0
        
        # 6. Estrategia defensiva - valorar bloqueos
        for p_row, p_col in player_pieces:
            for o_row, o_col in opponent_pieces:
                # Si nuestra pieza está bloqueando diagonalmente a una del oponente
                if abs(p_col - o_col) == 1 and (p_row - o_row) * direction == 1:
                    evaluation += 0.8
        
        return evaluation




































































# """
# Minimax strategy implementation for Kulibrat AI.
# """

# import random
# import time
# from typing import Optional, Tuple

# from src.core.game_state_cy import GameState
# from src.core.move import Move
# from src.core.player_color import PlayerColor
# from src.players.ai.ai_strategy import AIStrategy


# class MinimaxStrategy(AIStrategy):
#     """AI strategy using the minimax algorithm with optional alpha-beta pruning."""

#     def __init__(self, max_depth: int = 5, use_alpha_beta: bool = True):
#         """
#         Initialize the minimax strategy.

#         Args:
#             max_depth: Maximum search depth
#             use_alpha_beta: Whether to use alpha-beta pruning
#         """
#         self.max_depth = max_depth
#         self.use_alpha_beta = use_alpha_beta
#         self.nodes_evaluated = 0

#     def select_move(
#         self, game_state: GameState, player_color: PlayerColor
#     ) -> Optional[Move]:
#         """
#         Select the best move using minimax.

#         Args:
#             game_state: Current state of the game
#             player_color: Color of the player making the move

#         Returns:
#             The best move according to minimax
#         """
#         start_time = time.time()
#         self.nodes_evaluated = 0

#         valid_moves = game_state.get_valid_moves()
#         if not valid_moves:
#             return None

#         # If there's only one valid move, return it immediately
#         if len(valid_moves) == 1:
#             return valid_moves[0]

#         best_score = float("-inf")
#         best_moves = []

#         for move in valid_moves:
#             # Apply the move to a copy of the state
#             next_state = game_state.copy()
#             next_state.apply_move(move)

#             # Run minimax
#             if self.use_alpha_beta:
#                 score, _ = self._alpha_beta(
#                     state=next_state,
#                     depth=self.max_depth - 1,
#                     alpha=float("-inf"),
#                     beta=float("inf"),
#                     current_player=player_color.opposite(),
#                     maximizing_player=player_color,
#                 )
#             else:
#                 score, _ = self._minimax(
#                     state=next_state,
#                     depth=self.max_depth - 1,
#                     current_player=player_color.opposite(),
#                     maximizing_player=player_color,
#                 )

#             # Track best moves
#             if score > best_score:
#                 best_score = score
#                 best_moves = [move]
#             elif score == best_score:
#                 best_moves.append(move)

#         # Choose randomly from equally good moves for variety
#         selected_move = random.choice(best_moves)

#         end_time = time.time()
#         elapsed = end_time - start_time
#         print(f"Minimax evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s")
#         print(f"Selected move: {selected_move} with score: {best_score}")

#         return selected_move

#     def _minimax(
#         self,
#         state: GameState,
#         depth: int,
#         current_player: PlayerColor,
#         maximizing_player: PlayerColor,
#     ) -> Tuple[float, Optional[Move]]:
#         """
#         Minimax algorithm without alpha-beta pruning.

#         Args:
#             state: Current game state
#             depth: Remaining search depth
#             current_player: Player making the move at this level
#             maximizing_player: The AI player we're finding the best move for

#         Returns:
#             Tuple of (score, move)
#         """
#         self.nodes_evaluated += 1

#         # Terminal conditions
#         if state.is_game_over() or depth == 0:
#             return self._evaluate(state, maximizing_player), None

#         valid_moves = state.get_valid_moves()

#         # No valid moves means the other player gets an extra turn
#         if not valid_moves:
#             return self._minimax(
#                 state, depth - 1, current_player.opposite(), maximizing_player
#             )

#         is_maximizing = current_player == maximizing_player
#         best_score = float("-inf") if is_maximizing else float("inf")
#         best_move = None

#         for move in valid_moves:
#             # Apply the move to a copy of the state
#             next_state = state.copy()
#             next_state.apply_move(move)

#             # Recursively evaluate
#             score, _ = self._minimax(
#                 next_state, depth - 1, next_state.current_player, maximizing_player
#             )

#             # Update best score
#             if is_maximizing:
#                 if score > best_score:
#                     best_score = score
#                     best_move = move
#             else:
#                 if score < best_score:
#                     best_score = score
#                     best_move = move

#         return best_score, best_move

#     def _alpha_beta(
#         self,
#         state: GameState,
#         depth: int,
#         alpha: float,
#         beta: float,
#         current_player: PlayerColor,
#         maximizing_player: PlayerColor,
#     ) -> Tuple[float, Optional[Move]]:
#         """
#         Minimax algorithm with alpha-beta pruning.

#         Args:
#             state: Current game state
#             depth: Remaining search depth
#             alpha: Alpha value for pruning
#             beta: Beta value for pruning
#             current_player: Player making the move at this level
#             maximizing_player: The AI player we're finding the best move for

#         Returns:
#             Tuple of (score, move)
#         """
#         self.nodes_evaluated += 1

#         # Terminal conditions
#         if state.is_game_over() or depth == 0:
#             return self._evaluate(state, maximizing_player), None

#         valid_moves = state.get_valid_moves()

#         # No valid moves means the other player gets an extra turn
#         if not valid_moves:
#             return self._alpha_beta(
#                 state,
#                 depth - 1,
#                 alpha,
#                 beta,
#                 current_player.opposite(),
#                 maximizing_player,
#             )

#         is_maximizing = current_player == maximizing_player
#         best_move = None

#         if is_maximizing:
#             best_score = float("-inf")

#             for move in valid_moves:
#                 # Apply the move to a copy of the state
#                 next_state = state.copy()
#                 next_state.apply_move(move)

#                 # Recursively evaluate
#                 score, _ = self._alpha_beta(
#                     next_state,
#                     depth - 1,
#                     alpha,
#                     beta,
#                     next_state.current_player,
#                     maximizing_player,
#                 )

#                 # Update best score
#                 if score > best_score:
#                     best_score = score
#                     best_move = move

#                 # Update alpha
#                 alpha = max(alpha, best_score)

#                 # Alpha-beta pruning
#                 if beta <= alpha:
#                     break

#             return best_score, best_move
#         else:
#             best_score = float("inf")

#             for move in valid_moves:
#                 # Apply the move to a copy of the state
#                 next_state = state.copy()
#                 next_state.apply_move(move)

#                 # Recursively evaluate
#                 score, _ = self._alpha_beta(
#                     next_state,
#                     depth - 1,
#                     alpha,
#                     beta,
#                     next_state.current_player,
#                     maximizing_player,
#                 )

#                 # Update best score
#                 if score < best_score:
#                     best_score = score
#                     best_move = move

#                 # Update beta
#                 beta = min(beta, best_score)

#                 # Alpha-beta pruning
#                 if beta <= alpha:
#                     break

#             return best_score, best_move

#     def _evaluate(self, state: GameState, player_color: PlayerColor) -> float:
#         """
#         Heuristic: Evaluate the state for the given player.

#         Args:
#             state: Game state to evaluate
#             player_color: Player to evaluate the state for

#         Returns:
#             Score representing how good the state is for the player
#         """
#         board = state.board # 4x3 numpy array representing the board
#         pieces_off_board = state.pieces_off_board # Number of pieces available to be inserted (not on board)
#         scores = state.scores # Current scores
#         target_state = state.target_score # How many points to win
#         #-----------------------------------------
#         # HERE IS WHERE YOU IMPLEMENT YOUR HEURISTIC
#         #-----------------------------------------
#         return score
