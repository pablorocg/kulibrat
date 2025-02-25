# Kulibrat

A minimalist board game with simple rules yet non-trivial strategy. Kulibrat ("marble on a board") features a 3x4 board where two players move pieces diagonally, attempting to cross the board while blocking opponents through strategic placement, attacks, and jumps.

## Game Overview

Kulibrat was invented by Thomas Bolander with the aim of creating a game with the smallest possible board and simplest possible rules that would still be non-trivial in terms of strategy.

- **Players**: 2 players (Red and Black)
- **Board**: 3x4 grid
- **Pieces**: 4 pieces per player
- **Goal**: Be the first to score 5 points by moving pieces across the board

## Rules

Each player has a starting row on their side of the board. On each turn, a player can:

1. **Insert** a piece on an empty square in their starting row
2. **Move diagonally** forward to an empty square
3. **Attack** an opponent's piece directly in front, taking its place
4. **Jump** over a line of opponent pieces (1-3 pieces) to an empty square or off the board

Points are scored when a piece moves off the opponent's edge of the board, either through a diagonal move or jump.

## Installation

### Requirements

- Python 3.6+
- Pygame (for the graphical interface)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/kulibrat.git
cd kulibrat

# Install dependencies
pip install pygame
```

## Usage

### Graphical Interface

```bash
# Run with default settings
python main.py

# Run with custom win score
python main.py --win-score 10
```

### Command Line Interface

```bash
# Play in command line mode (human vs. human)
python main.py --cli

# Play against AI (human as Black, AI as Red)
python main.py --cli --red-ai

# AI vs. AI game
python main.py --cli --black-ai --red-ai
```

## Controls

- **GUI Mode**:
  - Click on a piece to select it
  - Click on a highlighted square to move the selected piece
  - Click on a highlighted square in your start row to insert a new piece
  - After game over, press 'R' to restart or 'Q' to quit

- **CLI Mode**:
  - Follow the on-screen prompts to select your move

## Project Structure

```
kulibrat/
├── __init__.py
├── src/
│   ├── __init__.py
│   ├── board.py       # Board representation and game state
│   ├── moves.py       # Move validation and execution
│   ├── player.py      # Player logic
│   └── game.py        # Main game loop and scoring
├── gui/
│   ├── __init__.py
│   ├── assets/
│   │   ├── red_piece.png
│   │   └── black_piece.png
│   └── ui.py          # GUI implementation
└── main.py            # Entry point
```

## Creating Assets

The game looks for piece images in the `gui/assets/` directory:

- `red_piece.png` - Image for red player pieces
- `black_piece.png` - Image for black player pieces

If these files are not found, the game will fallback to using colored circles.

## License

This project is released under the MIT License.

## Acknowledgements

- Thomas Bolander for inventing the Kulibrat game and documenting its rules