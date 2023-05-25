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
from dataclasses import dataclass
from functools import lru_cache


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

        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluationFunction(state)

        if agent == self.PACMAN_INDEX:
            legal_actions = (action for action in state.getLegalActions(self.PACMAN_INDEX))
            successor_states = (state.generateSuccessor(self.PACMAN_INDEX, action) for action in legal_actions)
            evaluations = (self.minimax(successor, depth, agent + 1) for successor in successor_states)

            return max(evaluations, default=-float("inf"))

        else:  # else statement for clarity
            legal_actions = (action for action in state.getLegalActions(agent))
            successor_states = (state.generateSuccessor(agent, action) for action in legal_actions)
            evaluations = (self.minimax(successor, depth, agent + 1) for successor in successor_states)

            return min(evaluations, default=float("inf"))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)

    STUDENT NOTE: code is largely based on pseudo code in this video:
    https://youtu.be/l-hh51ncgDI
    technically I don't need to add this, but I felt like it. It's a decent video.
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        oo = float("inf")  # Note: oo stands for infinity (it slightly resembles the infinity symbol)
        alpha = -oo
        beta = +oo

        max_evaluation = -float("inf")
        best_action = Directions.STOP

        for action in gameState.getLegalActions(self.PACMAN_INDEX):
            new_state = gameState.generateSuccessor(self.PACMAN_INDEX, action)
            new_evaluation = self.minimax(new_state, 0, self.PACMAN_INDEX + 1, alpha, beta)

            if new_evaluation >= max_evaluation:
                max_evaluation = new_evaluation
                best_action = action

            alpha = max(new_evaluation, alpha)
            if beta < alpha:
                # This is where I start thinking there should definitely be a more efficient way to do write this...
                break

        return best_action

    def minimax(self, state: GameState, depth, agent, alpha, beta):
        if agent >= state.getNumAgents():
            # All ghosts have been cycled through
            agent = self.PACMAN_INDEX
            depth += 1  # We are now in a deeper level

        if state.isLose() or state.isWin() or self.depth == depth:
            return self.evaluationFunction(state)

        if agent == self.PACMAN_INDEX:
            legal_actions = (action for action in state.getLegalActions(self.PACMAN_INDEX))
            successor_states = (state.generateSuccessor(self.PACMAN_INDEX, action) for action in legal_actions)

            max_evaluation = -float("inf")
            for successor in successor_states:
                evaluation = self.minimax(successor, depth, agent + 1, alpha, beta)
                max_evaluation = max(max_evaluation, evaluation)
                alpha = max(alpha, evaluation)
                if beta < alpha:
                    break

            return max_evaluation

        else:  # I hope you don't have phasmophobia! Because we're gonna talk GHOSTS!
            legal_actions = (action for action in state.getLegalActions(agent))
            successor_states = (state.generateSuccessor(agent, action) for action in legal_actions)

            min_evaluation = float("inf")
            for successor in successor_states:
                evaluation = self.minimax(successor, depth, agent + 1, alpha, beta)
                min_evaluation = min(min_evaluation, evaluation)
                beta = min(beta, evaluation)
                if beta < alpha:
                    break

            return min_evaluation


@dataclass(frozen=True)
class ActionNode:
    """
    the only reason for this class is to not have to repeat any logic or code in the getAction method.
    Could've (and should've) used this before, and made expectimax the exception to not use it.
    We all make mistakes in life.
    """
    evaluation: float
    action: str


def _uniform_chance_of_action(state: GameState, agent: int, action: str):
    legal_action_count = len(state.getLegalActions(agent))
    if action in state.getLegalActions(agent):
        # I don't understand the problem with truncation? / is for division (to float), // for integer division
        return 1/legal_action_count
    return 0


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
        return self.expectimax(gameState, 0, self.PACMAN_INDEX).action  # all this code just to get to put .action here

    def expectimax(self, state: GameState, depth: int, agent: int, *, chance_function=_uniform_chance_of_action)\
            -> ActionNode:
        """
        Note that this function functions differently from minimax. Instead of simply returning the value, we also
        return the action for this node. Both values are returned using an ActionNode object.
        """
        if agent >= state.getNumAgents():
            # All ghosts have been cycled through
            agent = self.PACMAN_INDEX
            depth += 1  # We are now in a deeper level

        if state.isLose() or state.isWin() or self.depth == depth:
            return ActionNode(self.evaluationFunction(state), Directions.STOP)

        if agent == self.PACMAN_INDEX:
            return self.max_node(state, depth, agent)

        else:  # GHOSTS!! AAH!!! THEY'RE CHASING ME!!!1!
            return self.expected_node(state, depth, agent, chance_function=chance_function)

    def max_node(self, state: GameState, depth: int, agent: int):
        """
        Returns the ActionNode with the largest value
        """
        legal_actions = (action for action in state.getLegalActions(agent))
        successor_states_and_actions = ((state.generateSuccessor(agent, action), action) for action in legal_actions)
        max_value_actions = (ActionNode(self.expectimax(successor, depth, agent + 1).evaluation,
                                        action) for successor, action in successor_states_and_actions)

        return max(max_value_actions, default=ActionNode(0, Directions.STOP), key=lambda x: x.evaluation)

    def expected_node(self, state: GameState, depth: int, agent: int, *, chance_function=_uniform_chance_of_action):
        """
        Returns the ActionNode with an expected value per the chance_function. The action given is the most likely
        action.

        STUDENT NOTE: woops. I forgot that the chance-node can't just return one action...
        Workaround: return the most likely one.
        Why do things the easy way when things can be done the hard way too?
        """
        legal_actions = (action for action in state.getLegalActions(agent))
        successor_states_and_actions = ((state.generateSuccessor(agent, action), action) for action in legal_actions)

        most_likely_action = Directions.STOP
        highest_probability = 0
        expected_value = 0
        for successor, action in successor_states_and_actions:
            probability = chance_function(state, agent, action)
            expected_value += probability * self.expectimax(successor, depth, agent + 1).evaluation

            if probability > highest_probability:
                highest_probability = probability
                most_likely_action = action

        return ActionNode(expected_value, most_likely_action)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Before I mention what I've done, let me mention that I've tried a few things as you can see in the
                 code, and ended up having to settle for this, as through trial and error this is the highest I could
                 get the score to go.

    First of all, we check the trivial conditions, and give a high/low score to each. Since it is generally
    easier for Pac-Man to lose, the low score for losing is less punishing than the high score for winning is
    rewarding.
    Next, we track all locations, and then calculate the weight of each value for the evaluation. First, the shortest
    distance to a pellet is calculated using a reciprocal weight function.

    """

    # Trivial conditions: winning should be prioritised at all costs, losing avoided at all costs
    if currentGameState.isWin():
        return 1000000
    if currentGameState.isLose():
        return -10000

    # Relevant weight functions
    pellet_weight_function = reciprocal()

    # locations of relevant agents/pellets
    food_locations = tuple(currentGameState.getFood().asList())
    pacman_location = currentGameState.getPacmanPosition()
    ghost_locations = tuple(currentGameState.getGhostPositions())

    # Pellet weights
    pellet_distance_weight = 1 / shortest_distance(pacman_location, food_locations)

    # Ghost weight
    ghost_distance_weight = 0
    ghost_distance = shortest_distance(pacman_location, ghost_locations)
    if ghost_distance < 2:
        ghost_distance_weight = 1000

    evaluated_score = currentGameState.getScore()

    return ghost_distance_weight + pellet_distance_weight * 30 + evaluated_score * 200


"""
[UNUSED]
STUDENT NOTE: I'm reusing my code from P1, where I constructed a minimal spanning tree between the food pellets.
Side note: I've noticed my original heuristic in P1 was not actually consistent, and I believe I could fix it by
    constructing an MST between the food pellets (using manhattanDistance preferably) and then find the shortest
    (manhattan-) distance from Pac-Man to the nearest food pellet. It's too late for this though, but I did want
    you to know.
"""


class Graph:
    """
    [UNUSED]
    A Graph which takes in the nodes, and an optional weight function. An adjacency matrix is created, linking every
    node to every other node, and adds a weight to that (undirected) edge equal to the weigh function applied to both
    nodes.

    This Graph class also implements prim's algorithms to convert the graph to a minimal spanning tree of the graph. It
    also implements a method to get the total weight of this (undirected) graph

    This class and relative functions are based on the code at:
    https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/minimum-spanning-trees-prims-algorithm/
    """
    def __init__(self, nodes, weight_function=manhattanDistance):
        self.node_count = len(nodes)

        # Construct the adjacency matrix
        self.adjacency_matrix = [[0 for _ in range(self.node_count)] for _ in range(self.node_count)]

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                self.adjacency_matrix[i][j] = weight_function(node1, node2)

    def apply_prim(self):
        """Applies prim's algorithm to this graph, and updates the adjacency matrix accordingly."""
        inf = float("inf")
        selected_nodes = [False for _ in range(self.node_count)]

        result = [[0 for _ in range(self.node_count)] for _ in range(self.node_count)]
        start, end = 0, 0  # start and end indicate the index of the starting and ending node of a new edge

        # oof, look at this nesting... disgusting.
        while False in selected_nodes:
            cheapest = inf

            for i in range(self.node_count):
                if selected_nodes[i]:
                    for j in range(self.node_count):
                        if not selected_nodes[j] and 0 < self.adjacency_matrix[i][j] < cheapest:
                            cheapest = self.adjacency_matrix[i][j]
                            start, end = i, j

            selected_nodes[end] = True
            result[start][end] = cheapest
            result[end][start] = result[start][end]

            # Technically irrelevant, adding for completeness
            # I'd also think this could break the algorithm, since end never updates if this statement is true...?
            if cheapest == inf:
                result[start][end] = 0

        self.adjacency_matrix = result

    def get_total_weight(self):
        return sum(sum(self.adjacency_matrix[i]) for i in range(self.node_count)) / 2


@lru_cache(maxsize=256)  # caching last few pellet configurations. Why repeat the same work?
def minimal_spanning_tree_weight(nodes: tuple, weight_function=manhattanDistance):
    """
    [UNUSED]
    Constructs a minimal spanning tree (MST) between the given nodes, with the weight function used to create the
    weights of the edges between each pair of nodes. Lastly, returns the total weight of this MST.
    """
    graph = Graph(nodes, weight_function=weight_function)
    graph.apply_prim()
    return graph.get_total_weight()


def shortest_distance(start_node: tuple, goal_nodes: tuple, *, weight_function=manhattanDistance):
    """
    Returns the shortest distance from a start node (e.g. Pac-Man) to any goal node (e.g. the closest pellet), with the
    distance computed using the weight function.
    """
    return min(weight_function(start_node, node) for node in goal_nodes)


def summed_distance(start_node: tuple, goal_nodes: tuple, *, weight_function=manhattanDistance):
    """
    [UNUSED]
    Returns the sum of the distances from a start node (e.g. Pac-Man) to each goal node (e.g. the closest pellet), with
    the distance computed using the weight function.
    """
    return sum(weight_function(start_node, node) for node in goal_nodes)


def reciprocal(distance_function=manhattanDistance, *, unit_distance=1, power: int = 1, if_zero=1):
    """
    Returns a new function, which is an updated version of the passed in distance function. This new distance function
    returns the reciprocal of the return value of the original distance function, to some power (default=1). This value
    is then also multiplied, causing the unit distance (originally 1) to be raised/lowered to a new value (default=10).
    If the original distance function were to return 0, to prevent a divide by 0 exception, the if_zero value
    (default=1) is returned instead.
    """
    def diminishing_distance_function(*args, **kwargs):  # I love wrapper functions.
        distance = distance_function(*args, **kwargs)
        if distance == 0:
            return if_zero
        return unit_distance * distance**(-power)
    return diminishing_distance_function


# Abbreviation
better = betterEvaluationFunction
