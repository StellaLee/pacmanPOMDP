# bustersAgents.py
# ----------------
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
#


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
from math import sqrt, log
from random import sample, random
import inference
import busters
from copy import deepcopy
import random

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True, obsOnStopOnly = False,
                  prior = False, rolloutPolicy='Uniform'):
        inferenceType = util.lookup(inference, globals())
        if prior:
            self.inferenceModules = [inferenceType(a, prior) for a in ghostAgents]
        else:
            self.inferenceModules = [inferenceType(a) for a in ghostAgents]

        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.ghostAgents = ghostAgents
        self.obsOnStopOnly = obsOnStopOnly
        self.lastAction = None
        self.rolloutPolicy = rolloutPolicy
        
        self.gamma = 0.99
        self.epsilon = 0.01
        self.c = 500.0
        self.numSimulations = 200
        self.depth = 10

        self.searchTree = {}
        self.searchTree[()] = [0,0]


    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

        "A history is a list of actions and observations"        
        self.history = ()

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        #gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        for i in range(1, len(agents)):
            agents[i].setPosition(None)

        if self.obsOnStopOnly :
            if self.lastAction == Directions.STOP:
                observation = gameState.getNoisyGhostDistances()
                self.history = self.history + (tuple(observation),)
            else:
                self.history = self.history + (tuple([None]),)

        else:
            observation = gameState.getNoisyGhostDistances()
            self.history = self.history + (tuple(observation),)
        

        
        return gameState


    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(gameState)
            self.firstMove = False
            if self.obsOnStopOnly:
                if self.lastAction == Directions.STOP:
                    if self.observeEnable:
                        inf.observeState(gameState)
            else:
                if self.observeEnable:
                    inf.observeState(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
        self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

    
    def sampleState(self, gameState, mle=False):
        """
        Samples a state from the agent's current beliefs. If mle is set to 
        True, the maximum likelihood estimate is always returned.
        """
        
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)]
        for (i, ghost) in zip(range(1, len(livingGhostPositionDistributions)+1),
                              livingGhostPositionDistributions):
            if mle:
                ghostPosition = ghost.argMax()
                gameState = deepcopy(gameState)
                gameState.setAgentPosition(ghostPosition, i)
            else:
                ghostPosition = util.sample(ghost)
                gameState = deepcopy(gameState)
                gameState.setAgentPosition(ghostPosition, i)

                    
        return gameState




from distanceCalculator import Distancer
from game import Actions
from game import Directions

    
class MDPAgent(BustersAgent):
    "An MDP agent that tries to get to the goal without hitting a ghost"

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)        


    def getMinimalState(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        minimalState = (pacmanPosition,)
        for ghostIndex in range(1, gameState.getNumAgents()):
            minimalState += (gameState.getGhostPosition(ghostIndex),)

        return minimalState
    
    def simulate(self, gameState, depth):
        """
        self.getMinimalState(gameState) gives you a Minimal State.
        A Minimal state contains a tuple with just pacman's position and
        the positions of the ghosts. Use Minimal states as keys in the UCT
        search tree instead of the full gameState in order to reduce
        memory usage.  

        self.searchTree is a dictionary that holds the monte-carlo search tree.
        Keys into the search tree are either minimal states 
        or a (minimal state, action) pair, and values are tuples
        (V, N), where V is the value, and N is the visitation count. 
        
        Use gameState.getLegalPacmanActions() to get the list of available
        actions.

        gameState.isWin() or gameState.isLose() are available for you to 
        determine whether the game is over.
        
        Use self.c, self.depth to get the 
        exploration constant and maximum search depth.
        
        Use sqrt() and log() to compute square roots and logarithms
        
        Use gameState.generateStateObservationReward(self, self.ghostAgents, a) 
        in order to generate a (nextState, observation, reward) tuple
        from the given gameState and action a, by sampling from the world
        dynamics. You may not need to use the observation that is returned.

        """

        "*** YOUR CODE HERE ***"
        #pass
        actions = gameState.getLegalPacmanActions()
        state = self.getMinimalState(gameState)
        #print gameState.isWin()
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return 0
        if state not in self.searchTree:
            self.searchTree[state] = 0
            for a in actions:
                self.searchTree[(state, a)] = (0,0)
            return self.rollout(gameState, depth) 
        
        bestAction = None
        for a in actions:     
            if self.searchTree[(state, a)]==(0,0):
                bestAction = a
                break

        if bestAction == None:
            Naction = util.Counter()
            Qvalue = util.Counter()
            for a in actions:
                Naction[(state, a)] = list(self.searchTree[(state,a)])[0]
                Qvalue[(state, a)] =  list(self.searchTree[(state,a)])[1]
            #Ntotal = sum(Naction.values())    
            #bestAction = None
            MaxQ = -float('inf')
            Ntotal = self.searchTree[state]
            #print Naction
            for a in actions:
                temp = Qvalue[(state, a)] + self.c*sqrt(log(Ntotal)*1.0/Naction[(state,a)])
                if temp>MaxQ:
                    MaxQ = temp
                    bestAction = a
        nextState, observation, reward = gameState.generateStateObservationReward(self, self.ghostAgents, bestAction) 
        R = reward + self.gamma*self.simulate(nextState, depth+1)

        self.searchTree[state] += 1

        resultlist = list(self.searchTree[(state,bestAction)])
        resultlist[0] += 1


        resultlist[1] += (R-resultlist[1])*1.0/resultlist[0]
        self.searchTree[(state, bestAction)] = tuple(resultlist)
        return R



    
    def rollout(self, gameState, depth):
        "*** YOUR CODE HERE ***"        
        #pass
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return 0
        actions = gameState.getLegalPacmanActions()
        bestAction = random.choice(actions)
        nextState, observation, reward = gameState.generateStateObservationReward(self, self.ghostAgents, bestAction) 
        R = reward + self.gamma*self.rollout(nextState, depth+1)
        return R
        

            
    def chooseAction(self, gameState):

        sampledGameState = self.sampleState(gameState, mle=True)
            
        for i in range(self.numSimulations):
            self.simulate(sampledGameState, 0)
        

        bestAction = None
        bestValue = float('-inf')

        values = {}
        legal = [a for a in gameState.getLegalPacmanActions()]
        minimalState = self.getMinimalState(sampledGameState)
        
        for action in legal:
            t_ha = self.searchTree[(minimalState, action)]
            value = t_ha[1]
            values[action] = value
            if value > bestValue:
                bestAction = action
                bestValue = value        
        
        return bestAction

class POMDPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        

    

    def simulate(self, gameState, history, depth):
        """
        self.searchTree is a dictionary that holds the monte-carlo search tree.
        Keys into the search tree are either histories
        or a (history, action) pair, and values are tuples
        (V, N), where V is the value, and N is the visitation count. 
        
        Use gameState.getLegalPacmanActions() to get the list of available
        actions.

        gameState.isWin() or gameState.isLose() are available for you to 
        determine whether the game is over.
        
        Use self.c, self.depth to get the 
        exploration constant and maximum search depth.
        
        Use sqrt() and log() to compute square roots and logarithms
        
        Use gameState.generateStateObservationReward(self, self.ghostAgents, a) 
        in order to generate a (nextState, observation, reward) tuple
        from the given gameState and action a, by sampling from the world
        dynamics. 

        """

        "*** YOUR CODE HERE ***"
        #pass
        actions = gameState.getLegalPacmanActions()
        #state = self.getMinimalState(gameState)
        #print gameState.isWin()
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return 0
        if history not in self.searchTree:
            self.searchTree[history] = 0
            for a in actions:
                self.searchTree[(history+(a,))] = (0,0)
            return self.rollout(gameState,history,depth) 
        
        bestAction = None
        for a in actions:     
            if self.searchTree[(history+(a,))]==(0,0):
                bestAction = a
                break

        if bestAction == None:
            Naction = util.Counter()
            Qvalue = util.Counter()
            for a in actions:
                Naction[(history+(a,))] = list(self.searchTree[(history+(a,))])[0]
                Qvalue[(history+(a,))] =  list(self.searchTree[(history+(a,))])[1]
            #Ntotal = sum(Naction.values())   
            Ntotal = self.searchTree[history] 
            #bestAction = None
            MaxQ = -9999999
            for a in actions:
                temp = Qvalue[(history+(a,))] + self.c*sqrt(log(Ntotal)*1.0/Naction[(history+(a,))])
                if temp>MaxQ:
                    MaxQ = temp
                    bestAction = a
        nextState, observation, reward = gameState.generateStateObservationReward(self, self.ghostAgents, bestAction) 
        R = reward + self.gamma*self.simulate(nextState, history + (bestAction, observation), depth+1)

        self.searchTree[history] += 1
        resultlist = list(self.searchTree[(history+(bestAction,))])
        resultlist[0] += 1

        resultlist[1] += (R-resultlist[1])*1.0/resultlist[0]
        self.searchTree[(history+(bestAction,))] = tuple(resultlist)
        return R


    
    def rollout(self, gameState, history, depth):
        if self.rolloutPolicy == 'Uniform':
            return self.rolloutUniform(gameState, history, depth)
        elif self.rolloutPolicy == 'Better':
            return self.rolloutBetter(gameState, history, depth)
        
    def rolloutUniform(self, gameState, history, depth):
        "*** YOUR CODE HERE ***"
        #pass
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return 0
        actions = gameState.getLegalPacmanActions()
        bestAction = random.choice(actions)
        nextState, observation, reward = gameState.generateStateObservationReward(self, self.ghostAgents, bestAction) 
        R = reward + self.gamma*self.rollout(nextState, history + (bestAction, observation), depth+1)
        return R


    
    def rolloutBetter(self, gameState, history, depth):        
        "*** YOUR CODE HERE ***"
        #pass
        if depth > self.depth or gameState.isLose() or gameState.isWin():
            return 0
        actions = gameState.getLegalPacmanActions()
        
        bestAction = None
        bestValue = float('inf')
        newFood = gameState.getFood()
        pos = newFood.asList()[0]
        tuning = 10.0
        for action in actions:
            nextGameState = gameState.generateSuccessor(0, action)
            newPos = nextGameState.getPacmanPosition()
            d = self.distancer.getDistance(pos, newPos)
            if nextGameState.isWin():
                bestAction = action
                break
            if self.history + (action,) in self.searchTree:    
                Qvalue = self.searchTree[self.history + (action,)][1]
                value = Qvalue + tuning / d
                if value > bestValue:
                    bestAction = action
                    bestValue = value    
            if d < bestValue:
                bestAction = action
                bestValue = d      
        if bestAction == None:
            bestAction = random.choice(actions)                
        nextState, observation, reward= gameState.generateStateObservationReward(self,self.ghostAgents,bestAction)

        R = reward + self.gamma * self.rolloutBetter(nextState,history+(bestAction,observation),depth+1)        
        return R

        
    def chooseAction(self, gameState):
        legal = gameState.getLegalPacmanActions()
        
        for i in range(self.numSimulations):
            
            sampledGameState = self.sampleState(gameState)
            self.simulate(sampledGameState, self.history, 0)
        

        bestAction = None
        bestValue = float('-inf')

        values = {}
        for action in legal:
            t_ha = self.searchTree[self.history + (action,)]
            value = t_ha[1]
            values[action] = value
            if value > bestValue:
                bestAction = action
                bestValue = value

        self.history = self.history + (bestAction,)
        
        
        return bestAction
