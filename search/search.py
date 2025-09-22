# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
	"""
	This class outlines the structure of a search problem, but doesn't implement
	any of the methods (in object-oriented terminology: an abstract class).

	You do not need to change anything in this class, ever.
	"""

	def getStartState(self):
		"""
		Returns the start state for the search problem.
		"""
		util.raiseNotDefined()

	def isGoalState(self, state):
		"""
		  state: Search state

		Returns True if and only if the state is a valid goal state.
		"""
		util.raiseNotDefined()

	def getSuccessors(self, state):
		"""
		  state: Search state

		For a given state, this should return a list of triples, (successor,
		action, stepCost), where 'successor' is a successor to the current
		state, 'action' is the action required to get there, and 'stepCost' is
		the incremental cost of expanding to that successor.
		"""
		util.raiseNotDefined()

	def getCostOfActions(self, actions):
		"""
		 actions: A list of actions to take

		This method returns the total cost of a particular sequence of actions.
		The sequence must be composed of legal moves.
		"""
		util.raiseNotDefined()


def tinyMazeSearch(problem):
	"""
	Returns a sequence of moves that solves tinyMaze.  For any other maze, the
	sequence of moves will be incorrect, so only use this for tinyMaze.
	"""
	from game import Directions
	s = Directions.SOUTH
	w = Directions.WEST
	return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
	"""
	Search the deepest nodes in the search tree first.

	Your search algorithm needs to return a list of actions that reaches the
	goal. Make sure to implement a graph search algorithm.

	To get started, you might want to try some of these simple commands to
	understand the search problem that is being passed in:

	print("Start:", problem.getStartState())
	print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
	print("Start's successors:", problem.getSuccessors(problem.getStartState()))
	"""
	"*** YOUR CODE HERE ***"
	
	visited = set() # Empty set for the nodes we've already visited
	not_visited = util.Stack() # Stack for the nodes left to visit
	
	# We put the first node on top of the stack
	# Said node consists of the item (start state for the first one) and the path (empty list) 
	not_visited.push( (problem.getStartState(),[]) ) 
	
	while not not_visited.isEmpty(): # We repeat the process until there's no more nodes to be visited
		
		top = not_visited.pop() # Take the top item of the stack to add it to the visited set
		item = top[0]
		path = top[1]

		if item not in visited: # If we haven't already visited this node, we add it to the visited set
			visited.add(item)
			
			if problem.isGoalState(item):
				return path
			
			successors = problem.getSuccessors(item) # We get the adjacent nodes of the item
			for successor in successors: # For each one we check if we've visited it 
				if successor[0] not in visited:  # If not, we add the item and it's updated path to the not_visited stack
					not_visited.push( (successor[0], path +[successor[1]] ) )
	
	return [] 

	util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
	"""Search the shallowest nodes in the search tree first."""
	"*** YOUR CODE HERE ***"
	
	visited = set() # Empty set for the nodes we've already visited
	not_visited = util.Queue() # Queue for the nodes left to visit
	
	# We put the first node in the front of the queue
	# Said node consists of the item (start state for the first one) and the path (empty list) 
	not_visited.push( (problem.getStartState(),[]) ) 
	
	while not not_visited.isEmpty(): # We repeat the process until there's no more nodes to be visited
		
		front = not_visited.pop() # Take the front item of the queue to add it to the visited set
		item = front[0]
		path = front[1]

		if item not in visited: # If we haven't already visited this node, we add it to the visited set
			visited.add(item)
			
			if problem.isGoalState(item):
				return path
			
			successors = problem.getSuccessors(item) # We get the adjacent nodes of the item
			for successor in successors: # For each one we check if we've visited it 
				if successor[0] not in visited:  # If not, we add the item and it's updated path to the not_visited queue
					not_visited.push( (successor[0], path +[successor[1]] ) )
	
	return [] 
	
	util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
	"""Search the node of least total cost first."""
	"*** YOUR CODE HERE ***"
	
	visited = set() # Empty set for the nodes we've already visited
	not_visited = util.PriorityQueue() # Priority Queue for the nodes left to visit
 
	node = {"state" : problem.getStartState(),
			"previous" : None, 
			"action" : None, 
			"path" : 0 } #path = prioprity
	
	not_visited.push(node,node["path"])
	
	while not not_visited.isEmpty(): # We repeat the process until there's no more nodes to be visited

		node = not_visited.pop() # Take the highest priority item of the queue to add it to the visited set
		item = node["state"]
		path = node["path"]
		

		if item not in visited: #If we haven't already visited this node, we add it to the visited set
			visited.add(item)
			
			if problem.isGoalState(item):
				break
			
			successors = problem.getSuccessors(item) # We get the adjacent nodes of the item
			
			for successor in successors: # For each one we check if we've visited it 
				if successor[0] not in visited:  # If not, we add the item and it's updated path to the not_visited queue
					new_node = {"state" : successor[0],
								"previous": node ,
								"action" : successor[1],
								"path" : successor[2]+path } #path = prioprity					not_visited.push(new_node,new_node["path"])
					not_visited.push(new_node,new_node["path"])
				
	actions=[]
	
	while node["action"] != None:
		actions.insert(0,node["action"])
		node=node["previous"]
	return actions

	util.raiseNotDefined()

def nullHeuristic(state, problem=None):
	"""
	A heuristic function estimates the cost from the current state to the nearest
	goal in the provided SearchProblem.  This heuristic is trivial.
	"""
	return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
	"""Search the node that has the lowest combined cost and heuristic first."""
	"*** YOUR CODE HERE ***"
	
	visited = set() # Empty set for the nodes we've already visited
	not_visited = util.PriorityQueue() # Priority Queue for the nodes left to visit
	node = {"state" : problem.getStartState(),
			"previous" : None,
			"action" : None,
			"path" : 0,
			"h" : heuristic(problem.getStartState(), problem)	} 
 
	not_visited.push(node,node["path"] + node["h"])
	
	while not not_visited.isEmpty(): # We repeat the process until there's no more nodes to be visited

		node = not_visited.pop() # Take the highest priority item of the queue to add it to the visited set
		item = node["state"]
		path = node["path"]
		

		if item not in visited: #If we haven't already visited this node, we add it to the visited set
			visited.add(item)
			
			if problem.isGoalState(item):
				break
			
			successors = problem.getSuccessors(item) # We get the adjacent nodes of the item
			
			for successor in successors: # For each one we check if we've visited it 
				if successor[0] not in visited:  # If not, we add the item and it's updated path to the not_visited queue
					new_node = {"state" : successor[0],
								"previous": node ,
								"action" : successor[1],
								"path" : successor[2]+path, #path = prioprity	
								"h" : heuristic(successor[0], problem) } 
					not_visited.push(new_node,new_node["path"] +new_node["h"])
				
	actions=[]
	
	while node["action"] != None:
		actions.insert(0,node["action"])
		node=node["previous"]
	return actions
	util.raiseNotDefined()
	


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
