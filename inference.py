# inference.py
# ------------
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
#
# This file modified by Christopher Lin (chrislin@cs.washington.edu) 


import itertools
import util
import random
import busters
import game
from copy import deepcopy

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent, prior=False):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position
        self.prior = prior
        
    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        if self.prior:
            for p in self.prior: self.beliefs[p] = 1.0
        else:
            for p in self.legalPositions: self.beliefs[p] = 1.0

        self.beliefs.normalize()

    def observe(self, observation, gameState):

        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"
        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
        allPossible = util.Counter()
        # first handle when the ghost is eaten
        if noisyDistance == None:
            allPossible[self.getJailPosition()] = 1.0
        else:
            # when the ghost is still alive somewhere
            for p in self.legalPositions:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                allPossible[p] = emissionModel[trueDistance] * self.beliefs[p]

        "*** END YOUR CODE HERE ***"

        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        "*** YOUR CODE HERE ***"
        allPossible = util.Counter()
        # iterate through all the old positions
        pacmanPosition = gameState.getPacmanState().getPosition()
        pacmanDirection = gameState.getPacmanState().getDirection()

        if pacmanDirection == game.Directions.NORTH:
            pacmanDirection = game.Directions.SOUTH
        elif pacmanDirection == game.Directions.SOUTH:
            pacmanDirection = game.Directions.NORTH
        elif pacmanDirection == game.Directions.EAST:
            pacmanDirection = game.Directions.WEST
        elif pacmanDirection == game.Directions.WEST:
            pacmanDirection = game.Directions.EAST

        #print pacmanDirection
        oldGameState = deepcopy(gameState)
        busters.PacmanRules.applyAction(oldGameState, pacmanDirection)
        oldPacmanPosition = oldGameState.getPacmanState().getPosition()
        
        
        for oldPos in self.legalPositions:
            if oldPos == oldPacmanPosition:
                continue
            # In order to obtain the distribution over new positions for the ghost,
            # given its previous position (oldPos) as well as Pacman's current position
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            # go through all the new pos and prob to calculate the new value and update
            for newPos, prob in newPosDist.items():
                allPossible[newPos] += prob * self.beliefs[oldPos]
        # normalize and update
        allPossible.normalize()
        #print allPossible
        self.beliefs = allPossible

    def getBeliefDistribution(self):
        return self.beliefs


def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState

