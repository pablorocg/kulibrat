## Kulibrat rules

By Thomas Bolander

April 15, 2019

### 1 Introduction to the game

Kulibrat is a rule invented by Thomas Bolander. The aim was to develop a game with the smallest
possible board and simplest possible rules that would still be non-trivial in the sense of there not
being any obvious or easily computable winning strategy. The starting point was games like tic-
tac-toe that are really small in terms of board size and rule complexity, but where simple optimal
strategiesdoexist. The name Kulibrat is an abbreviation of “kugle i bræt” (“marble on a board”),
due to the first version of the game being made by drilling holes in a wooden board and using
marbles as the pieces.

### 2 Introduction to the rules

It is a game played by two players, red and black. Each player has 4 identical pieces of their
respective color. It is played on a 3x4 board. It can easily be played e.g. on a chess board, by
only using a 3x4 corner of the board and using pawns as the pieces. The following figure shows
an example of a state of the game, and the position of the two players:

```
Start row of red→
```
```
Start row of black→
```
```
Score
Black: 0
Red: 0
```
Each player has a start row where her pieces enter the board. The player is expected to sit at this
end of the board. In the illustrations used in this rulebook, we will always take the start row of
black to be the top row, and the start row of red to be the bottom one. The goal of each player is
to move as many pieces as possible across the board from her own end to the other end. In each
turn, when black is to move, she can choose among the actions described in the following section
(symmetrically for red).


## 3 Possible moves (for black)

#### Inserting a piece

Black can insert a piece on one of the three squares of her start row. The conditions are: the
square needs to be free, and black needs to have a piece available (that is, not all 4 black pieces
are on the board already). Example:

```
Score
Black: 0
Red: 0
```
# →

```
Score
Black: 0
Red: 0
```
#### Diagonal move

Black can move one of her pieces diagonally forward, that is, the piece is moved one row further
away from her start row while at the same time moving one column left or right. Such a move is
only legal when the destination square is free. Note that from the middle column one can choose
between moving diagonally left and diagonally right, whereas from the leftmost and rightmost
columns only one diagonal move is possible. Example:

```
Score
Black: 0
Red: 0
```
# →

```
Score
Black: 0
Red: 0
```
A special case of a diagonal move of black is when the piece is on the start row of red. In this
case, the diagonal move will move the piece outside the board, and one point is scored for black.
Example:

```
Score
Black: 0
Red: 0
```
# →

```
Score
Black: 1
Red: 0
```
The piece can later be reused, that is, be used in an insert move.

#### Attack

If a black piece is right in front of a red piece (as seen from the perspective of the black player),
the black piece can take over the square of the red piece, and the red piece is given back to the
red player. Example:


```
Score
Black: 1
Red: 0
```
# →

```
Score
Black: 1
Red: 0
```
This move doesn’t affect the scores.

#### Jump

If a there is line of red pieces behind a black piece, then the black piece can jump over this line.
The condition is that the square behind the line of red pieces is either free or outside the board
(the latter happens when the line of red pieces ends at the start row of red). If the jump ends
outside the board, one point is scored for black. A line of red pieces can consist of either 1, 2 or
3 pieces. A jump over a line consisting of only one red piece could look as follows:

```
Score
Black: 1
Red: 0
```
# →

```
Score
Black: 1
Red: 0
```
A jump over a line of two red pieces could look as follows:

```
Score
Black: 1
Red: 0
```
# →

```
Score
Black: 2
Red: 0
```
Note that black scored a point in this case.

## 4 Further rules

Before the game begins, it is agreed to how many points the game is played, usually 5 or 10. The
first player to reach this number of points has won. There are certain special conditions to be
aware of, described in the following.

#### No available moves for one of the players

It is possible to reach game positions where the player who’s turn it is doesn’t have any available
legal moves. In this case it is the other player’s turn until the first player again has an available
move. Consider e.g. the following state:


```
Score
Black: 1
Red: 0
```
Even if it is black’s turn in this state, black doesn’t have any available legal moves (no free square
on the start row to insert a new piece into, no free squares to move diagonally into, no attack or
jumps possible). Hence red gets an extra move (in some case it could be several).

#### No available moves for both players

It is even possible to reach a state where none of the players can move:

```
Score
Black: 1
Red: 0
```
If the game reaches such a state, the game is lost for the last player to make a move, that is, the
player who is responsible for locking the game. It is always possible to choose another action that
doesn’t lead to a lock for both players. For instance, if the last move in the situation above was
black entering a piece in the upper left corner, black could instead have entered the piece in the
middle row, thereby giving herself an extra move (since red would still be locked).

#### Winning cycles

There are certain states of the game from which it is easy for one of the players to get all the
remaining points by enforcing the game to become cyclic. This is not a flaw of the game, and it
can be quite hard to get the game into one of these states if the other player is aware of the risk.
It is for you to find out what those states are and see if you can force the game into them to get
a quick win...


