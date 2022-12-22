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


from cmath import exp, inf
from util import manhattanDistance
from game import Configuration, Directions
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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        # Useful information you can extract from a GameState (pacman.py)
        succ_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = succ_game_state.getPacmanPosition()
        new_food = succ_game_state.getFood()
        new_ghost_states = succ_game_state.getGhostStates()
        new_scared_times = [
            ghostState.scaredTimer for ghostState in new_ghost_states]

        "*** YOUR CODE HERE ***"
        man_dist = util.manhattanDistance
        inf, n_inf = float("inf"), float("-inf")
        total, closest_food, result = 0, 0, 0

        if action == "Stop":
            return n_inf
        for i in new_food.asList():
            if inf > man_dist(i, new_pos):
                closest_food = i
        if closest_food:
            total -= (man_dist(new_pos, closest_food) / 4)
        result = total + succ_game_state.data.score
        return result


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.inf = float("inf")
        self.n_inf = float("-inf")


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
        "*** YOUR CODE HERE ***"
        maxx = self.n_inf
        result = None
        legal_action = gameState.getLegalActions(agentIndex=0)

        for i in legal_action:
            succ_val = self.get_val(gameState.generateSuccessor(0, i), 1, 0)
            if maxx < succ_val:
                maxx, result = succ_val, i
        return result

    def win_or_lose(self, game_state):
        return game_state.isWin() or game_state.isLose()

    def get_val(self, game_state, agent_index, depth):
        result = None

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        if agent_index != 0:
            return self.min_val(game_state, agent_index, depth)
        else:
            return self.max_val(game_state, depth)

    def min_val(self, game_state, agent_index, depth):
        minn = self.inf
        result = None
        legal_action = game_state.getLegalActions(agent_index)

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        for i in legal_action:
            if (game_state.getNumAgents() - 1) != agent_index:
                minn = min(self.get_val(game_state.generateSuccessor(
                    agent_index, i), (agent_index + 1), depth), minn)
                result = minn
            else:
                minn = min(self.get_val(game_state.generateSuccessor(
                    agent_index, i), 0, (depth + 1)), minn)
                result = minn
        return result

    def max_val(self, game_state, depth):
        maxx = self.n_inf
        result = None
        legal_action = game_state.getLegalActions(agentIndex=0)

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        for i in legal_action:
            maxx = max(self.get_val(
                game_state.generateSuccessor(0, i), 1, depth), maxx)
            result = maxx
        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxx, alpha = self.n_inf, self.n_inf
        result = None
        legal_action = gameState.getLegalActions(0)

        for i in legal_action:
            succ_val = self.get_val(gameState.generateSuccessor(
                0, i), 1, 0, alpha, self.inf)
            if maxx < succ_val:
                maxx, result = succ_val, i
            alpha = max(maxx, alpha)
        return result

    def win_or_lose(self, game_state):
        return game_state.isWin() or game_state.isLose()

    def get_val(self, game_state, agent_index, depth, alpha, beta):
        result = None
        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        if agent_index != 0:
            return self.min_val(game_state, agent_index, depth, alpha, beta)
        else:
            return self.max_val(game_state, depth, alpha, beta)

    def min_val(self, game_state, agent_index, depth, alpha, beta):
        minn = self.inf
        result = None
        legal_action = game_state.getLegalActions(agent_index)

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        for i in legal_action:
            if (game_state.getNumAgents() - 1) != agent_index:
                minn = min(self.get_val(game_state.generateSuccessor(
                    agent_index, i), (agent_index + 1), depth, alpha, beta), minn)
                result = minn
            else:
                minn = min(self.get_val(game_state.generateSuccessor(
                    agent_index, i), 0, (depth + 1), alpha, beta), minn)
                result = minn
            if alpha > result:
                return result
            beta = min(result, beta)
        return result

    def max_val(self, game_state, depth, alpha, beta):
        maxx = self.n_inf
        result = None
        legal_action = game_state.getLegalActions(0)

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        for i in legal_action:
            maxx = max(self.get_val(
                game_state.generateSuccessor(0, i), 1, depth, alpha, beta), maxx)
            result = maxx
            if beta < result:
                return result
            alpha = max(result, alpha)
        return result


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
        best = self.n_inf
        result = None
        legal_action = gameState.getLegalActions(0)

        for i in legal_action:
            succ_val = self.get_val(gameState.generateSuccessor(0, i), 1, 0)
            if best < succ_val:
                best = succ_val
                result = i
        return result

    def win_or_lose(self, game_state):
        return game_state.isWin() or game_state.isLose()

    def get_val(self, game_state, agent_index, depth):
        result = None
        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        if agent_index != 0:
            return self.expected_val(game_state, agent_index, depth)
        else:
            return self.max_val(game_state, depth)

    def max_val(self, game_state, depth):
        maxx = self.n_inf
        result = None
        legal_action = game_state.getLegalActions(0)

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        for i in legal_action:
            maxx = max(self.get_val(
                game_state.generateSuccessor(0, i), 1, depth), maxx)
            result = maxx
        return result

    def expected_val(self, game_state, agent_index, depth):
        exp_val = 0
        result = None
        legal_action = game_state.getLegalActions(agent_index)

        if depth == self.depth or self.win_or_lose(game_state):
            return self.evaluationFunction(game_state)
        for i in legal_action:
            gen_succ = game_state.generateSuccessor(agent_index, i)

            if (game_state.getNumAgents() - 1) != agent_index:
                exp_val += self.get_val(gen_succ,
                                        (agent_index + 1), depth)
            else:
                exp_val += self.get_val(gen_succ, 0, (depth + 1))
        return exp_val


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    man_dist = util.manhattanDistance
    inf, n_inf = float('inf'), float('-inf')
    ghost_state = currentGameState.getGhostStates()
    pacman_pos = currentGameState.getPacmanPosition()
    food_lst = currentGameState.getFood().asList()
    scared_time = [i.scaredTimer for i in ghost_state]
    num_food = len(food_lst)
    num_caps = len(currentGameState.getCapsules())

    if currentGameState.isWin():
        return inf
    if currentGameState.isLose():
        return n_inf

    def near_food_dist():
        result = inf

        for i in food_lst:
            dist = man_dist(pacman_pos, i)
            if dist < result:
                result = dist
        return result

    def near_ghost_dist():
        result, scared, count = 0, 0, 0
        ghost1, ghost2 = inf, inf

        for i in ghost_state:
            dist = man_dist(i.getPosition(), pacman_pos)
            if scared_time[count] != 0 and ghost2 > dist:
                ghost2 = dist
                scared = scared_time[count]
            if ghost1 > dist:
                ghost1 = dist
            count += 1
        if ghost2 != inf:
            result = (1 / ghost2) + scared
        else:
            result = ghost1
        return result

    return (-4 * num_food) + (-20 * num_caps) + currentGameState.getScore() + (-1.5)\
        * (near_food_dist() + 2) * near_ghost_dist()


# Abbreviation
better = betterEvaluationFunction
