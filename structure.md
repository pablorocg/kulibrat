game_project/
│
├── src/                        # Source code directory
│   ├── core/                   # Core game components
│   │   ├── __init__.py
│   │   ├── board.py            # Board classes
│   │   ├── game_state.py       # GameState classes
│   │   ├── game_rules.py       # Rules implementations
│   │   └── game.py             # Game class
│   │
│   ├── players/                # Player implementations
│   │   ├── __init__.py
│   │   ├── player.py           # Base Player class and interfaces
│   │   ├── human_player.py     # HumanPlayer implementation
│   │   └── ai/                 # AI player implementations
│   │       ├── __init__.py
│   │       ├── ai_player.py    # Base AIPlayer class
│   │       ├── strategies.py   # AI strategy interface
│   │       ├── minimax.py      # Minimax algorithms
│   │       └── mcts.py         # Monte Carlo Tree Search
│   │
│   ├── ui/                     # User interface implementations
│   │   ├── __init__.py
│   │   ├── renderer.py         # Base renderer interface
│   │   ├── console_ui.py       # Console UI implementation
│   │   └── gui_ui.py           # GUI implementation
│   │
│   ├── games/                  # Game-specific implementations
│   │   ├── __init__.py
│   │   ├── tic_tac_toe/        # Tic Tac Toe game
│   │   │   ├── __init__.py
│   │   │   └── tic_tac_toe.py  # TicTacToe-specific classes
│   │   │
│   │   └── kulibrat/           # Kulibrat game
│   │       ├── __init__.py
│   │       └── kulibrat.py     # Kulibrat-specific classes
│   │
│   ├── factory/                # Factory pattern implementations
│   │   ├── __init__.py
│   │   └── game_factory.py     # GameFactory class
│   │
│   └── utils/                  # Utility functions and classes
│       ├── __init__.py
│       └── common.py           # Common utility functions
│
├── scripts/                    # Executable scripts
│   ├── play_tic_tac_toe.py     # Main entry point for Tic Tac Toe
│   └── play_kulibrat.py        # Main entry point for Kulibrat
│
├── tests/                      # Test directory
│   ├── __init__.py
│   ├── test_board.py
│   ├── test_game_rules.py
│   ├── test_game_state.py
│   ├── test_ai_strategies.py
│   └── test_kulibrat.py
│
├── requirements.txt            # Project dependencies
├── setup.py                    # Installation script
└── README.md                   # Project documentation