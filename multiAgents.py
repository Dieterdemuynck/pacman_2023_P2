# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random
import util
from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Previous state data:
        previous_food = currentGameState.getFood().asList()
        # New state:
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        # Food:
        new_position = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood().asList()
        # Ghosts:
        new_ghost_states = successor_game_state.getGhostStates()
        new_ghost_positions = successor_game_state.getGhostPositions()
        new_scared_timed = (ghostState.scaredTimer for ghostState in new_ghost_states)
        new_ghost_positions_with_timers = zip(new_ghost_positions, new_scared_timed)

        # Scores:
        food_score = self.get_food_score(new_position, new_food)
        # It's possible we ate a pellet in the action: add 1 to the food_score
        if self.ate_pellet(previous_food, new_food):
            food_score += 1
        ghost_score = self.get_ghost_score(new_position, new_ghost_positions_with_timers)

        return food_score + ghost_score

    @staticmethod
    def get_food_score(pacman_position, food_locations):
        """
        Calculates a value for how good pac-man's position is as opposed to the food locations. Returns the sum of the
        inverse of 2 to the power of the manhattan distances of each pellet to pac-man's position.
        """
        food_distance = (manhattanDistance(pacman_position, food_location) for food_location in food_locations)
        return inverse_exp_sum(food_distance)

    @staticmethod
    def get_ghost_score(pacman_position, ghost_positions_with_timers):
        score = 0.0_0  # hmm... What could this mean?

        for ghost_position, scared_value in ghost_positions_with_timers:
            distance = manhattanDistance(pacman_position, ghost_position)

            # We never want to move next to or on a ghost:
            if distance == 1 or distance == 0:
                return -float("inf")

            # We'd rather stay a good distance from the ghost, but shouldn't care as long as we're far enough away:
            if distance == 2:
                score -= 1
            elif distance == 3:
                score -= 0.5

        return score

    @staticmethod
    def ate_pellet(previous_food, new_food):
        return len(previous_food) != len(new_food)


def inverse_exp_sum(iterable):
    return sum(2**(-x) for x in iterable)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    PACMAN_INDEX = 0

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        max_evaluation = -float("inf")
        best_action = Directions.STOP
        
        for action in gameState.getLegalActions(self.PACMAN_INDEX):
            new_state = gameState.generateSuccessor(self.PACMAN_INDEX, action)
            new_evaluation = self.minimax(new_state, 0, self.PACMAN_INDEX + 1)

            if new_evaluation >= max_evaluation:
                max_evaluation = new_evaluation
                best_action = action

        return best_action

    def minimax(self, state: GameState, depth, agent):
        if agent >= state.getNumAgents():
            # All ghosts have been cycled through
            agent = self.PACMAN_INDEX
            depth += 1  # We are now in a deeper level

        if state.isLose() or state.isWin() or self._is_deepest_layer(state, depth, agent):
            return self.evaluationFunction(state)

        if agent == self.PACMAN_INDEX:
            legal_actions = (action for action in state.getLegalActions(self.PACMAN_INDEX))
            successor_states = (state.generateSuccessor(self.PACMAN_INDEX, action) for action in legal_actions)

            return max(self.minimax(successor, depth, agent + 1) for successor in successor_states)

        else:  # else statement for clarity
            legal_actions = (action for action in state.getLegalActions(agent))
            successor_states = (state.generateSuccessor(agent, action) for action in legal_actions)

            return min(self.minimax(successor, depth, agent + 1) for successor in successor_states)

    def _is_deepest_layer(self, state: GameState, depth, agent):
        # In an attempt to bugfix, I tried this:
        # return depth == self.depth-1 and agent == state.getNumAgents()-1
        # Turns out the bug was not in the minimax method, but rather in the getAuction method.
        return depth == self.depth


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
