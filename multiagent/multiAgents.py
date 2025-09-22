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
import random, util

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}/
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"   
        
        food = currentGameState.getFood()
        
        # If there's food left, meaning pacman won, we return the highest possible value
        if len(newFood.asList()) == 0 :
            return float('inf')
  
        # For each ghost, we check if the proposed position leads pacman to said ghost
        for ghost in newGhostStates: 
      
            ghost_pos = ghost.getPosition()
            # If the ghosts aren't scared and pacman crashes into one of them, we return the lowest possible value
            if ghost_pos == newPos and ghost.scaredTimer == 0:
                return float('-inf')
        
        dist_from_food = []
  
        for food_dot in food.asList():
            
            # If pacman eat's a food dot, we return the highest possible value
            if manhattanDistance(newPos,food_dot) == 0:
                return float('inf')
            # We store the reciprocal number, in order to have the smallest distance be stored as the highest number
            dist_from_food.append( 1/(manhattanDistance(newPos,food_dot)) )
            
        # We return the max number of the list, which represents the shortest possible distance to a dot, since all numbers are 
        # the reciprocal of the original value
        value_of_action = max(dist_from_food)
        
        return value_of_action
  
  

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"
      
        def minimax(gameState,depth,agent):
            
            # If we've checked all agents in a certain δepth, move on to the next depth, starting again from agent 0(pacman)
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            elif agent == 0:
                return max_value(gameState,depth,agent)
            else:
                return min_value(gameState,depth,agent)

        def max_value(gameState, depth, agent):
            
            actions = gameState.getLegalActions(agent)
            
            if not actions:
                return self.evaluationFunction(gameState)
   
            best_value =("None", float('-inf'))

            # For each action for this agent we examine it's successors
            for action in actions:

                successor = gameState.generateSuccessor(agent, action)

                temp = minimax(successor,depth,agent+1)
                
                if type(temp) is tuple:
                    value = temp[1]
                else:
                    value = temp
                
                # If the value of the action we're examining is better than the previous one, we replace it (better in this case means bigger)
                if value > best_value[1]:
                    best_value = (action,value)
          
            # We return the tuple that consists of the best possible action and it's value for this agent
            return best_value
   
        def min_value(gameState, depth, agent):
      
            actions = gameState.getLegalActions(agent)
            
            if not actions:
                return self.evaluationFunction(gameState)
            
            best_value =("None", float('inf'))
            
            # For each action for this agent we examine it's successors
            for action in actions:
                
                successor = gameState.generateSuccessor(agent, action)
                
                temp = minimax(successor,depth,agent + 1)
                
                if type(temp) is tuple:
                    value = temp[1]
                else:
                    value = temp
                    
                # If the value of the action we're examining is better than the previous one, we replace it (better in this case means smaller)
                if value < best_value[1]:
                    best_value = (action,value)
            
            # We return the tuple that consists of the best possible action and it's value for this agent
            return best_value
        
        # Begins from agent 0 (pacman) and depth 0
        action = minimax(gameState,0,0)
        
        return action[0]
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState,agent,depth,a,b):
            
            # If we've checked all agents in a certain depth, move on to the next depth, starting again from agent 0(pacman)
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            elif agent == 0:
                return max_value(gameState,agent,depth,a,b)
            else:
                return min_value(gameState,agent,depth,a,b)

        
        def max_value(gameState,agent,depth,a,b):
            
            best_value =("None", float('-inf'))
            
            actions = gameState.getLegalActions(agent)
            
            if not actions:
                return self.evaluationFunction(gameState)
                
            for action in actions:
                
                successor = gameState.generateSuccessor(agent, action)
                    
                temp = minimax(successor,agent+1,depth,a,b)
                
                if type(temp) is tuple:
                    value = temp[1]
                else:
                    value = temp
                        
                if value > best_value[1]:
                    best_value = (action,value)
                
                if value > b:
                    return (action,value)
                
                a = max(a,best_value[1])

            return best_value
            
        def min_value(gameState,agent,depth,a,b):

            best_value = ("None",float('inf'))           
            
            actions = gameState.getLegalActions(agent)
            
            if not actions:
                return self.evaluationFunction(gameState)
                
                
            for action in actions:
                
                successor = gameState.generateSuccessor(agent, action)
                    
                temp = minimax(successor,agent+1,depth,a,b)

                if type(temp) is tuple:
                    value = temp[1]
                else:
                    value = temp
                        
                if value < best_value[1]:
                    best_value = (action,value)
                    
                if value < a:
                    return (action,value)
                        
                b = min(b,best_value[1])
                    
            return best_value
        
        
        action = minimax(gameState,0,0,float('-inf'),float('inf'))
        
        return action[0]
        

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
        
        def expectimax(gameState,agent,depth):
            
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            elif agent == 0:
                return max_value(gameState,agent,depth)
            else:
                return min_value(gameState,agent,depth)
            
        def max_value(gameState,agent,depth):
                        
            actions = gameState.getLegalActions(agent)
            
            if not actions:
                return self.evaluationFunction(gameState)
              
            best_value = ("None", float('-inf'))
            
            for action in actions:
                
                successor = gameState.generateSuccessor(agent, action)  
                
                temp = expectimax(successor,agent+1,depth)
                
                if type(temp) is tuple:
                    value = temp[1]
                else:
                    value = temp
                
                # If agent is pacman we return the optimal move 
                if value > best_value[1]:
                    best_value = (action,value)
            
            return best_value
        
        def min_value(gameState,agent,depth):
                                    
            actions = gameState.getLegalActions(agent)
            
            if not actions:
                return self.evaluationFunction(gameState)
            
            best_value =("None", float('inf'))

            count = 0 
            total = 0
            best_value = []
            
            for action in actions:
                
                count +=1

                successor = gameState.generateSuccessor(agent, action)  
                temp = expectimax(successor,agent+1,depth)
                
                if type(temp) is tuple:
                    value = temp[1]
                else:
                    value = temp
                
                total +=value
                
            avg = total/count
            # We return a random tuple of an action and it's value, which is the average of all values for these agents' actions
            best_value = (random.choice(actions),avg)
            
            return best_value
            
        action = expectimax(gameState,0,0)
        return action[0]  
                
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    """
    "*** YOUR CODE HERE ***"
    
    food = currentGameState.getFood()

    # No food left means pacman won so we score the state very highly
    if len(food.asList()) == 0:
        return float('inf')
    
    # The new position in the state we're evaluating
    newPos = currentGameState.getPacmanPosition()
    
    ghosts = currentGameState.getGhostStates()  
    pellets = currentGameState.getCapsules()

    score = 0
    dist_from_ghost = []
    dist_from_pellets = []
    
    for ghost in ghosts: 
      
        ghost_pos = ghost.getPosition()
        
        # If ghosts aren't scared we take into account the distance from each one, as well as the distance from each power pellet
        if ghost.scaredTimer == 0:
            
            # If pacman crashes to a ghost, we consider the state evaluated a loosing state and therefore return a very low score
            if newPos == ghost_pos:
                return float('-inf')
            else:
                # Orherwise we add the reciprocal number of the distance from the ghost to the list
                dist_from_ghost.append(1/manhattanDistance(newPos,ghost_pos))
        
            for pellet in pellets:
                # Again we add the reciprocal number
                dist_from_pellets.append(1/manhattanDistance(newPos,pellet))
                       
        
    dist_from_food = []
    
    for food_dot in food.asList():
          
        # If pacman eats a food_dot in, we consider the state evaluated a winning state and therefore return a very high score
        if manhattanDistance(newPos,food_dot) == 0:
            return float('inf')
        
        # We store the opposite number, in order to have the smallest distance be stored as the highest number
        dist_from_food.append( 1/(manhattanDistance(newPos,food_dot)) )
               
    score += sum(dist_from_food) + currentGameState.getScore() + sum(dist_from_ghost) + sum(dist_from_pellets)
    
    return score


# Abbreviation
better = betterEvaluationFunction
