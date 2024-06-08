# Search - In this section we will go over our implementation of search. This is where we would like the AI to be able
#to search for solutions to some kind of problem, no matter what that problem might be.

# Search Problems - A search problem has some sort of initial state, some place we begin, some sort of action 
#we can take or multiple actions that we can take in any given state. And it has a transtion model, some way
#of defining what happens when we go from one state and take one action, what state do we end up with as a result.
# In addition to that, we need some goal test to know whether or not we've reached a goal. And then we need
#a path cost function that tells us for any particular path, by following some sequence of actions, how expensive
#is that path. What does it cost in money or time or some other resource that we are trying to minimize 
#our usage of.
# The goal ultimately is to find a solution.


# Agent - An agent is just some entity that preceives its environment. 

# State - A state is just some configuration of the agent in its environment. 

# Intial State - The state in which the agent begins.

# Actions - Choices that can be made in a state.
# Actions(s) return the set of actions that can be executed in state s
# Actions(s) which takes an input "s", where "s" is going to be some state that exists inside of our environment
# Actions of (s) is going to take as input and return as output the set of all actions that can be executed in
#that state.

# Transition Model - A description of what state results from performing any applicable action in any state.
# Results(s,a) is a function that takes 2 parameters (s,a) and returns the state resulting from performing
#action a  in state s.

# State Space - The set of all states reachable from the initial state by any sequence of actions
#ie. The set of all the states we can get from the initial state via any sequence of actions, by taking 
#0 or 1 or 2 or more actions.

# Goal Test - A way to determine whether a given state is a goal state. 

# Path Cost - Is a numerical cost associated with a given path.
# When formulating search problems, we will often give every path some sort of numerical cost, some number
#telling us how expensive it is to take this particular option, and then tell our AI that instead of just
#finding a solution, some way of getting from initial state to goal, we'd like to find one that minimizes 
#this path cost. One that is less expensive, or takes less time, or minimizes some other numerical value.

# Solution - A sequence of actions that leads from the initial state to a goal state.
# Ideally, we would like to find not just any solution, but the optimal solution, which is a solution that 
#has the lowest path cost among all of the possible solutions.
# Some cases may produce multiple optimal solutions. But an optimal solution just means there is no way
#that we could have done better in terms of finding that solution.

# Node - A data state structure that keeps track of -
# - A state
# - A parent (node that generated this node)
# - An action (action applied to parent to get node)
# - A path cost (from initial state to node)
# Nodes help us keep track of what lead us to the goal and what led us to that state, and what lead us to 
#the state before that, so on and so forth, backtracking our way to the beginning so that we know the entire
#sequence of actions we needed in order to get from the beginning to the end.

# Approach 
# - Start with a frontier
# - Repeat
#   - If the frontier is empty, then no solution.
#   - Remove a node from the frontier.
#   - If node contains goal state, return the solution.
#   - Expand node, add existing nodes to the frontier.
# How might we actually begin to solve the problem? 
# What we're going to do is start at one particular state, and we're just going to explore from there.
# The intuition is that from a given state, we have multiple options that we could take, and we're going
#to explore those options. And once we explore those options, we'll find that more options that that are
#going to make themselves available.
# We're going to consider all of the available options to be stored inside of a single data structure 
#that we'll call frontier.
# Frontier - Represents all of the things that we could explore next that we haven't explored yet 
#or visited.
# The frontier contains the Initial State, because at the beginning, that's the only state we know of.
# Then, our search algorithm is effectively going to follow a loop.
# If ever our frontier is empty, that means there is nothing left to explore. We can report that there is no way
#to get to the goal because there is no solution. 
# There certain types of problems that an AI might try to explore and realize that there is no way to 
#solve that problem. 
# Otherwise, what we'll do is we'll remove a node from the frontier. 
# If that node happens to be a goal, then we found a solution. 
# So we remove a node from the frontier and ask ourselves, is this a goal? And we do that by applying the goal 
#test that we mentioned earlier, asking if we're at the destination. 
# Otherwise, what we'll need to do is expand the node. 
# To expand the node just means to look at all of the neighbors of that node. In other words, consider all
#of the possible actions that we could take from the state that this node is representing, and what nodes
#could I get to from there.
# We're going to take all of those nodes, the next nodes that we can get to from this current one we are
#looking at, and add those to the frontier.
# Then we repeat the process.

# Revised Approach 
# - Start with a frontier that contains the initial state.
# - Start with an empty explored set - This will just be a set of nodes that we have already explored. This helps us 
#elimate the chance of exploring states that we have already explored
# - Repeat:
#   - If the frontier is empty, then there is no solution.
#   - Remove a node from the frontier - It is also important how we decide to structure our frontier, how we 
#add and how we remove our nodes. The frontier is data structure we need to make a choice about in what order
#are we going to be removing elements. One of the simplest data structures for adding and removing elements
#is something called a "Stack".
#   - If node contains goal state, return the solution.
#   - Add the node to the explored set. - If it happens that we remove a node from the frontier, and it's not
#the goal, we'll add it to the explored set, so that we know that we've already explored it. This lets us
#know that we don't have to go back over it again if it happens to come up later.
#   - Expand node, add resulting nodes to the frontier if they aren't already in the frontier or explored set. 
# Again, this "Revised Approach" is to ensure that we don't go back and forth between 2 nodes.

# Stack 
# - Last-in first-out data type - This means that the last thing that we add to the frontier is going to be
#the first thing that we remove from the frontier. The most recent thing to go into the stack or the frontier,
#in this case, is going to be the node that we explore. 

# Depth-First-Search 
# - Search algorithm where we always explore the deepest node in the frontier.

# Breadth-First-Search
# - Search algorithm that always expands the shallowest node in the frontier - Instead of always exploring
#the deepest node in the search tree, the way the "Depth-First-Search" does, Breadth-First-Search is always 
#going to explore the shallowest node in the frontier.
# This means that instead of using a stack which D-F-S used, where the most recent item is added to the
#frontier is the one we'll explore next, in B-F-S, we'll instead use a queue, where a queue is a first in 
#first out data type, where the very first thing we add to the frontier is the one we'll explore and they
#effectively form a line or a queue, where the earlier you arrive in the frontier, the earlier you get explored.

# Queue
# - First-in First-Out Data Type.

# Uninformed Search - Search strategy that uses no problem specific knowledge
# Uninformed Search algorithms are algorithms like DFS and BFS. 
# DFS and BFS don't really care about the structure of the maze or anything about the way that a maze is in order
#to solve the problem. They just lookat the actions available and choose from those actions. It doesn't matter
#whether it's a maze or some other problem, the solution or the way that it tries to solve the problem is 
#really fundamentally going to be the same.

# Informed Search - Search strategy that uses problem specific knowledge to find solutions more efficiently.
# This is considered a more improved version of the Uninformed search.
# There are many types of informed searches.
# Greedy best-first search - Search algorithm that expands the node that is closest to the goal, as estimated
#by a heuristic function h(n). This search is often abbreviated as G BFS
# Heuristic Function h(n) - A way of estimating whether or not we're close to the goal. h(n) takes a state
#as an input.
# Manhattan Distance - A specific type of Heuristic, where the heuristic is how many squares vertically 
#and horizontally and then left to right, not diagonally. This heuristic tells us how many steps we need
#to take to get from each of these cells to the goal.

# A * Search - search algorithm that expands the node with lowest value of g(n) + h(n)
# g(n) - cost to reach node / how many steps we had to take to get to our current position
# h(n) - estimated cost to goal
# A * search is going to solve the problem by instead of just considering the heuristc, also considering
#how long it took us to get to any particular state.
# A * Search is the optimal approach, on condition.
# Optimal if -
#   - h(n) is admissible (never overestimates the true cost), and
#   - h(n) is consistent (for every node n and successor n'with step cost c.h(n) <= h(n') + c)
# n = node     n' = n prime   c =cost   
# A * Search tends to use a lot of memory

# Adversarial Search - Making intelligent decisions against an opposing force
# Minimax - An algorithm that helps us deal with adversarial type of search situations.
# This is how we can formulas what is happening with Minimax
#   -Given a state s:
#      -MAX picks action a in ACTIONS(s) that produces highest value of MIN-VAUE(RESULT(s,a))
#      -MIN picks action a in ACTIONS(s) that produces smallest value of MAX-VALUE(RESULT(s,a))
# This is an example of implementation of a Minimax
# function MAX-VALUE(state):
#   if TERMINAL(state):
#       return UTILITY(state)
#   v=-infinite 0
#   for action in ACTIONS(state):
#       v=MAX(v, MIN-VALUE(RESULT(state,action)))
#   return v
# This formula is going through all of our possible actions and asking the question, how do we maximize the 
#score given what our opponent will attempt to do?
# After the entire loop, we return v, and that is now the value of that particular state.
# For the Min we will apply the same logic but backwards
# function MIN-VALUE(state):
#   if TERMINAL(state):
#       return UTILITY(state)
#   v=infinity
#   for action in ACTIONS(state)
#       v=MIN(v, MAX-VALUE(RESULT(state,action)))
#   return v
# This formula will get us the smallest possible value of v that we then return back to the user.
# This was a pseudocode for Minimax. This is how we take a gain and figure out what the best move to make is 
#by recursively using these max value and min value functions, where max value calls min value, min value 
#calls max value back and forth all the way until we reach a terminal state, at which point our algorithm
#can simply return the utility of that particular state.

# Optimizations
# This is where we will take a look at optimizations
# Alpha-Beta Pruning - Alpha and Beta stand for the two values that we'll have to keep track of,
#the best we can do so far, and worst we can do so far. Pruning is the idea that if we have a big, long,
#deep search tree, we might be able to search it more efficiently if we don't have to search through everything,
#if we can remove some of the nodes to try and optimize the way that we look through the entire search space.
# So alpha, beta pruning can definitely save us a lot of time as we go about the search process by making our
#searches more efficient.

# Depth-Limited Minimax - After a certain amount of moves, the Depth-Limited Minimax will stop and not
#consider additional moves that might come after that, just because it would be computational intractable
#to consider all of those possible options.
# Evaluation Function - Function that estimates the expected utility of the game from a given state.
# This function pretty estimates how good the game state happens to be. And depending on how good that 
#evaluation function is, that is ultimately what is going to constrain how good the AI is. The better
#the AI is at estimating how good or how bad any particular game state is, the better the AI is going 
#to be able to play that game. If the evaluation function is worse and not as good as it estimating
#what the expected utility is, then it's going to be a whole lot harder. 
# There are many more variants of Minimax that add additional features in order to help it peform better 
#under these larger, more computationally untractable situations where we couldn't possibly explore all
#of the possible moves. 