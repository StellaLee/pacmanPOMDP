# trackingTestClasses.py
# ----------------------
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
# This file modified by Christopher Lin (chrislin@cs.washington.edu) 

# trackingTestClasses.py
# ----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import testClasses
import busters
import layout
import bustersAgents
from game import Agent
from game import Actions
from game import Directions
import random
import time
import util
import json
import re
import copy
from util import manhattanDistance

class GameScoreTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(GameScoreTest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'

    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [SeededRandomGhostAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.GreedyBustersAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass


class POMDPStaticTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(POMDPStaticTest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'
        if 'prior' in self.testDict:
            self.prior = [eval(p) for p in self.testDict['prior'].split(';')]
        else:
            self.prior = False
        
    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [DoNothingAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.POMDPAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable, obsOnStopOnly = True, prior = self.prior)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass

class POMDPBigTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(POMDPBigTest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'
        if 'prior' in self.testDict:
            self.prior = [eval(p) for p in self.testDict['prior'].split(';')]
        else:
            self.prior = False
        if 'rolloutPolicy' in self.testDict:
            self.rolloutPolicy = self.testDict['rolloutPolicy']
        else:
            self.rolloutPolicy = 'Uniform'
        
    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [SeededRandomGhostAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.POMDPAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable, obsOnStopOnly = False, prior = self.prior, rolloutPolicy=self.rolloutPolicy)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass
    
class MDPStaticTest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(MDPStaticTest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'
        if 'prior' in self.testDict:
            self.prior = [eval(p) for p in self.testDict['prior'].split(';')]
        else:
            self.prior = False
        
    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [DoNothingAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.MDPAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable, obsOnStopOnly = True, prior = self.prior)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass


class MDPMLETest(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(MDPMLETest, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'
        if 'prior' in self.testDict:
            self.prior = [eval(p) for p in self.testDict['prior'].split(';')]
        else:
            self.prior = False
        
    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [SeededRandomGhostAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.MDPAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable, obsOnStopOnly = False, prior = self.prior)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass


class POMDPTest2(testClasses.TestCase):

    def __init__(self, question, testDict):
        super(POMDPTest2, self).__init__(question, testDict)
        self.maxMoves = int(self.testDict['maxMoves'])
        self.inference = self.testDict['inference']
        self.layout_str = self.testDict['layout_str'].split('\n')
        self.numRuns = int(self.testDict['numRuns'])
        self.numWinsForCredit = int(self.testDict['numWinsForCredit'])
        self.numGhosts = int(self.testDict['numGhosts'])
        self.layout_name = self.testDict['layout_name']
        self.min_score = int(self.testDict['min_score'])
        self.observe_enable = self.testDict['observe'] == 'True'
        self.elapse_enable = self.testDict['elapse'] == 'True'
        
    def execute(self, grades, moduleDict, solutionDict):
        ghosts = [GoNorthAgent(i) for i in range(1,self.numGhosts+1)]
        pac = bustersAgents.POMDPAgent(0, inference = self.inference, ghostAgents = ghosts, observeEnable = self.observe_enable, elapseTimeEnable = self.elapse_enable, obsOnStopOnly = True)
        #if self.inference == "ExactInference":
        #    pac.inferenceModules = [moduleDict['inference'].ExactInference(a) for a in ghosts]
        #else:
        #    print "Error inference type %s -- not implemented" % self.inference
        #    return

        stats = run(self.layout_str, pac, ghosts, self.question.getDisplay(), nGames=self.numRuns, maxMoves=self.maxMoves, quiet = False)
        aboveCount = [s >= self.min_score for s in stats['scores']].count(True)
        msg = "%s) Games won on %s with score above %d: %d/%d" % (self.layout_name, grades.currentQuestion, self.min_score, aboveCount, self.numRuns)
        grades.addMessage(msg)
        if aboveCount >= self.numWinsForCredit:
            grades.assignFullCredit()
            return self.testPass(grades)
        else:
            return self.testFail(grades)

    def writeSolution(self, moduleDict, filePath):
        handle = open(filePath, 'w')
        handle.write('# You must win at least %d/10 games with at least %d points' % (self.numWinsForCredit, self.min_score))
        handle.close()

    def createPublicVersion(self):
        pass
    



def run(layout_str, pac, ghosts, disp, nGames = 1, name = 'games', maxMoves=-1, quiet = True):
    "Runs a few games and outputs their statistics."

    starttime = time.time()
    lay = layout.Layout(layout_str)

    #print '*** Running %s on' % name, layname,'%d time(s).' % nGames
    games = busters.runGames(lay, pac, ghosts, disp, nGames, maxMoves)

    #print '*** Finished running %s on' % name, layname, 'after %d seconds.' % (time.time() - starttime)

    stats = {'time': time.time() - starttime, \
      'wins': [g.state.isWin() for g in games].count(True), \
      'games': games, 'scores': [g.state.getScore() for g in games]}
    statTuple = (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))
    if not quiet:
        print '*** Won %d out of %d games. Average score: %f ***' % statTuple
    return stats


class SeededRandomGhostAgent(Agent):
    def __init__(self, index):
        self.index = index;

    def getAction(self, state):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = self.sample( dist )
            return action

    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

    def sample(self, distribution, values = None):
        if type(distribution) == util.Counter:
            items = distribution.items()
            distribution = [i[1] for i in items]
            values = [i[0] for i in items]
        if sum(distribution) != 1:
            distribution = normalize(distribution)
        choice = random.random()
        i, total= 0, distribution[0]
        while choice > total:
            i += 1
            total += distribution[i]
        return values[i]


class GoNorthAgent(Agent):
    def __init__(self, index):
        self.index = index;

    def getAction(self, state):
        dist = util.Counter()
        #for a in state.getLegalActions( self.index ):
        #    dist[a] = 1.0
        if Directions.NORTH in state.getLegalActions(self.index):
            dist[Directions.NORTH] = 1.0
        dist.normalize()
        if len(dist) == 0:
            return Directions.STOP
        else:
            action = self.sample( dist )
            return action

    def getDistribution( self, state ):
        dist = util.Counter()
        #for a in state.getLegalActions( self.index ):
        #    dist[a] = 1.0
        if Directions.NORTH in state.getLegalActions(self.index):
            dist[Directions.NORTH] = 1.0
        dist.normalize()
        if len(dist) == 0:
            dist[Directions.STOP] = 1.0
            
        return dist

    def sample(self, distribution, values = None):
        if type(distribution) == util.Counter:
            items = distribution.items()
            distribution = [i[1] for i in items]
            values = [i[0] for i in items]
        if sum(distribution) != 1:
            distribution = util.normalize(distribution)
        choice = random.random()
        i, total= 0, distribution[0]
        while choice > total:
            i += 1
            total += distribution[i]
        return values[i]

class DoNothingAgent(Agent):
    def __init__(self, index):
        self.index = index;

    def getAction(self, state):
        return Directions.STOP

    def getDistribution( self, state ):
        dist = util.Counter()
        dist[Directions.STOP] = 1.0
        return dist
            

    def sample(self, distribution, values = None):
        if type(distribution) == util.Counter:
            items = distribution.items()
            distribution = [i[1] for i in items]
            values = [i[0] for i in items]
        if sum(distribution) != 1:
            distribution = util.normalize(distribution)
        choice = random.random()
        i, total= 0, distribution[0]
        while choice > total:
            i += 1
            total += distribution[i]
        return values[i]
