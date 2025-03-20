# Kulibrat

A minimalist board game with simple rules and non-trivial strategy, implemented in Python.

## About the Game

Kulibrat is a strategic board game invented by Thomas Bolander, designed to create a complex gameplay experience on a small 3x4 board. The name is an abbreviation of "kugle i bræt" (Danish for "marble on a board").

### Game Characteristics
- Board: 3x4 grid
- Players: 2 (Black and Red)
- Pieces: 4 per player
- Objective: Be the first to score 5 points by moving pieces across the board

## Project Structure

```
kulibrat/
├── src/
│   ├── core/           # Game core logic
│   │   ├── game_engine.py
│   │   ├── game_rules.py
│   │   ├── game_state.py
│   │   └── ...
│   ├── players/        # AI and player strategies
│   │   ├── human_player/
│   │   ├── mcts_player/
│   │   ├── minimax_player/
│   │   └── random_player/
│   ├── tournament/     # Tournament management
│   │   ├── runner.py
│   │   ├── evaluator.py
│   │   └── ...
│   └── ui/             # User interfaces
│       ├── console_interface.py
│       ├── pygame_interface.py
│       └── ...
├── main.py             # Game entry point
├── run_tournament.py   # Tournament runner
└── requirements.txt    # Project dependencies
```

## Key Features

### Gameplay Mechanics
- Multiple move types:
  1. Insert: Place a piece on the start row
  2. Diagonal move: Move diagonally forward
  3. Attack: Capture opponent's piece
  4. Jump: Leap over opponent's pieces

### AI Strategies
- Random Player: Moves randomly
- Minimax Player: Uses minimax algorithm with configurable depth
- Monte Carlo Tree Search (MCTS): Probabilistic search strategy

### User Interfaces
- Console Interface: Text-based gameplay
- Pygame GUI: Graphical interface with interactive board

## Installation

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup
```bash
# Clone the repository
git clone https://github.com/pablorocg/kulibrat.git
cd kulibrat

# Install dependencies
pip install -r requirements.txt
```

## Running the Game

### Graphical Interface
```bash
# Default game
python main.py

# Customize win score
python main.py --win-score 10
```

### Command Line Interface
```bash
# Human vs Human
python main.py --cli

# Human vs AI
python main.py --cli --red-ai

# AI vs AI
python main.py --cli --black-ai --red-ai
```

## Tournament Mode

Run AI strategy tournaments:

```bash
python run_tournament.py --config tournament_config.yaml
```

### Tournament Features
- Configurable match parameters
- Multiple AI strategy comparisons
- Detailed performance reporting

## Advanced Features

- Genetic algorithm for AI heuristic optimization
- Comprehensive game statistics tracking
- Flexible configuration via YAML
- Extensible player and strategy design

## Configuration

Key configuration files:
- `kulibrat_config.yaml`: Game settings
- `tournament_config.yaml`: Tournament parameters

## Development

### Running Tests
```bash
pytest
```

### Code Quality
- Black for formatting
- Mypy for type checking
- Pylint for linting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

Distributed under the MIT License.

## Acknowledgements

- Thomas Bolander: Game Inventor
- Contributors and open-source community

## Contact

Project Link: [GitHub Repository URL]
Maintainer: Pablo Rocamora-García, 