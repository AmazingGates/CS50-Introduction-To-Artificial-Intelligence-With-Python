# Optimization - This is where we will focus on problems of when the computer is trying to optimize for some sort
#of goal, especially in a situation where there might be multiple ways that a computer might solve a problem, but 
#we're looking for a better way, or potentially the best way, if that's at all possible.

# Optimazation is all about choosing the best option from a set of options.

# What we are going to look at here is the category of types of problems and algorithms used to solve them, that
#can be used in order to deal with a broader range of potential optimazation problems. 

# The first of the algorithms we are going to look at is known as local search.

#   Local Search -
#                - Search algorithms that maintain a single node and searches by moving to a neighboring node.

# This is generally useful when we really don't care about the path at all, and all we care about is what the
#solution is.
# Local search is going to come up in cases where figuring out exactly what the solution is. Exactly what the
#goal looks like is the heart of the challenge.

# To give an example of one of these types of problems, we'll consider a scenario where we have two buildings.
# We have houses and hospitals. And our goal might be, in a grided world, where we have a bunch of houses, is 
#to find a way to place two hospitals on our grided map.
# The problem is that we want place two hospitals on our map, but we want to do so with some sort of objective.
# And our objective in this case, is to try and minimize the distance of any of the houses from the hospital.
# So we might imagine what's the distance from each of the houses to there nearest hospital. 
# There are a number of ways we can calculate that distance, but one way is using the manahattan distance.
# The idea of how many rows and columns we have to move inside of our grid layout in order to get to a hospital.

# It turns out that if we atke each of the four houses and figure out how close are they to their nearest
#hospital, we get something like the path way in our grid below. 
# One house is 3 spaces away. One is 6 spaces away. And two are four away (8).
# And if we add al those numbers up, we get a cost of 17.
# So for this particular configuration of hospitals, that state, we might say, has a cost of 17.

# The goal of this problem now is can we solve this problem to find a way to minimize that cost.

# If we thinkk about this problem a little more abstractly, abstracting away from this specific problem,
#and thinking more generally about problems like it, we can often formulate these problems by thinking about
#them as state space landscape.
# Generally speaking, when we have a space landcsape, we want to do one of two things. We might be trying to
#maximize the value of this function trying to find a global maximum. A single state whose value is higher
#than all of the other states that we could possibly choose from. 
# And generally in this case, when we are trying to find a global maximum, we'll call the function that we are
#trying to optimize some objective function.
# Some function that measures for any given state, how good is that state. Such that we can take any given state,
#pass in into the objective function and get a value for how good that state is.

# Ultimately what our goal is is to find one of these states that has the highest possible value for that objective
#function.

# An equivalent but reverse problem is finding the global minimum.
# Some state that has a value after we pass it into that function that is lower than all of the other possible
#values. 

# Generally when we are trying to find the global minimum, we call the function a cost function.
# Generally each state has some sort of cost. Whether that cost is a monetary cost, or a time cost, or in the
#case of the houses and hospitals, a distance cost.
# And we're trying to minimize the cost. Find the state that has the lowest possible value of that cost.


# Grid Map
#                                                                          Cost: 17
#|       |       |    |--|-------|-Hosp  |       |       |       | House-|----   |    
#|       |       | House |       |   |   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       |   |   |
#|       | House |-------|-------|----   |       |   |---|-------|-------|-Hosp  |
#|       |       |       |       |       |       | House |       |       |       |


#       Space Landscape
#  
# Objective     |            Global Maximum Model
#   Function    |
#               |
#               |   |
#               |   |   |
#               |   |   |   |
#       |   |   |   |   |   |
#       |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |



#       Space Landscape
#  
#     Cost      |             Global Minimum Model
#   Function    |
#               |
#               |   |
#               |   |   |
#               |   |   |   |
#       |   |   |   |   |   |
#       |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |

# These are the general types of ideas we might be trying to go for within a state space landscape.
# Trying to find a global maximum or a global minimum.
# And how exactly do we do that?

# Recall that in local search, we generally operate this algorithm by maintaing just a single state,
#just some current state represented inside some node, or maybe inside some data structure, where we are 
#keeping track of where we are currently.
# And then ultimately what we are going to do is, from that state, move to one of its neighbor states.
# So in the case of our state space landscape, represented by a one dimensional state immediately to the left 
#or right of it.
# But for a different problem we might define what it means to be a neighbor of a particular state.

# In the case of our hospitals, for example, what we were just looking at, a neighbor might be moving one 
#hospital one space to the left or right, or one space up or down. Some state that is close to our current
#state, but slightly different, and as a result, might have a slihtly different value in terms of its
#objective function, or in terms of its cost function.

# So this is going to be our generaaly strategy in local search. To be able to take a state maintaining
#some current node and move where we're looking at in a state space landscape in order to try and find
#a global mximum or global minimum somehow.
# And perhaps the simplest of algorithms that we can use to implement this idea of local search, is
#an algorithm known as hill climbing.

# The basic idea of hill climbing is, let's say we're trying to maximize the value of our state,
#and trying to figure out where the global maximum is. We're going to start at a state, and generally what
#hill climbing is going to do, is it's going to consider the neighbors of that state.
# And from that state, we can go left or we can go right.
# In hill climbing if we are trying to maximize the value, we'll generally pick the highest state we can
#between the states to the left and right us. After identifying which neighboring state is highest, we'll
#move our position to consider that state instead.
# Then we will repeat this process, continually looking at all of our neighbors, and picking the highest
#neighbor, doing the same thing, looking at our neighbors, and picking the highest of our neighbors, until
#we get to a point where we consider both of our neighbors, and both of our neighbors has a lower value
#than we do. 
# At that point, the algorithm terminates, and we can we have found the solution.

# And the samething works in the opposite way when trying to find the global minimum. The algorithm is 
#fundamentally the same.

# We can formulate this graphical idea in terms of psuedo code.
# The psuedo code might looklike this.
# We define some function called hill climb, which takes as input the problem we are trying to solve.
# And generally we are going to start in some sort of initial state. 
# So we'll start with a variable called current that is keeping track of our initial state, like an
#initial configuration of hospitals, and maybe some problems lend themselves to an initial state, a
#place where we begin, and in other cases maybe not, in which case we might just randomly generate 
#some initial state, just by choosing two locations for hospitals at random for example, and firguring 
#out from there how we might be able to improve.
# But that initial state, we are going to store inside of current.

# And now, here comes our loop. Some repetitive process we are going to do again and again, until
#the algorithm terminates.
# And what we are going to do, is first say, let's figure out all of the neighbors of our current state,
#from our state, what are all the neighboring states, for some definition of what it means to be a neighbor,
# and we'll choose the highest value of all those neighbors, and save it inside of our variable called 
#neightbor. Keeping track of the highest valued neighbor.
# This is in the case where we are trying to find the global maximum.
# Inn the case where we are trying to find the global minimum, we'll be doing the opposite process,
#looking for the lowest possible valued. 

# These ideas are really fundamentally the interchangable.

# And it is possible in some cases where there might be multiple neighbors that each have a equally high
#value, or an equally low value. In that case we can just choose randomly from among them. Just choose one
#of them, and save it inside our variable called neighbor.

# And then the key question to ask, is, is this neighbor better than my current state.
# And is the neighbor the best neighbor we were able to find?
# If not better than our current state, then the algorithm is over and we just return the current state.
# If no neighbors are better, we may as well stay where we are.  
# Otherwise, if the neighbor is better, then we may as well move to that neighbor.
# So we'll set current equal to neighbor.

# But the genenral idea is, if we are at a current state and we see a neighbor that is better than us,
#then we'll just go head and move to tha neighbor.
# Then we'll repeat the process, continually moving towards a better neighbor, until we reach a point
#where none of our neighbors are better than we are.
# Then at that point, we just say that the algorithm has terminated.


#function Hill-Climb(problem):
#  current = initial state of problem
#  repeat:
#    neighbor = highest valued neighbor of current
#    if neighbor not better than current:
#       return current
#    current = neighbor


# So let's take a look at a real example of this, with our houses and hospitals.

# We put the hospitals in these two locations, that has a total cost of 17. Now we need to define if we
#are going to implement this hill climbing algorithm, what it means to take this particular configuration 
#of hospitals, this particular state, and get a neighbor of that state. 

# And a simple defination of neighbor might just be let's pick one of the hospitals and move it by one
#square. The left, or right, or up, or down. 
# And that would mean we have six possible neighbors, from this particular configuration.
# and what we might do is say alright, here are the locations and distances between each of the houses
#and there nearest hospital. Let us consider all of the neighbors, and see if any of them can do better
#than a cost of 17.



# Grid Map
#                                                                          Cost: 17
#|       |       |    |--|--pm---|-Hosp  | pm    |       |       | House-|----   |    
#|       |       | House |       | pm|   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       | pm|   |
#|       | House |-------|-------|----   |       |   |---|-------|--pm---|-Hosp  |
#|       |       |       |       |       |       | House |       |       | pm    |
# Note: pm = possible move

# It turns out that there are a couple of ways that we can do that, and it doesn't matter if we randomly
#choose among all the ways that are the best, but one such possible way, is by taking a look at the
#hospital to the right of the grid, and considering the directions it might move, if we hold the 
#hospital to the left constant.
# If we take the hospital to the right and move it one square up for example, that doesn't really help
#us. 
# But if we take the hospital on the right, and move it one square down, it still doesn't help us much.

# The real idea, the goal should be to move the hospital on the right one square to the left.
# By moving it one square to the left, we move it closer to both the houses in its quardrant, without
#changing anything about the houses on the left.

# So we're able to improve the situation, by picking a neighbor that results in a decrease in our
#total cost.

# So we'll do that. We'll take our hospital from its current state and move it one square to the left
#to its neighbor. Our new cost would be 15. See example below

# Grid Map
#                                                                          Cost: 15
#|       |       |    |--|--pm---|-Hosp  | pm    |       |       | House-|----   |    
#|       |       | House |       | pm|   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       | pm|   |
#|       | House |-------|-------|----   |       |   |---|-------|--Hosp-|       |
#|       |       |       |       |       |       | House |       |       | pm    |
# Note: pm = possible move

# At this point, there's not a whole bunch that can be done with the hospital to he right, but 
#there are still other optimazations we can make, other neighbors we can move to that are going
#to have a better value.

# If we consider the hospital to the left, for example, we might imagine that right now it's a 
#bit far up, and both the houses in its quardrant a little bit lower, so we might be able to do
#better by taking that hospital, and moving it one square down so that now the cost went from 15
#to 13, for this particular configuration. See example below

# Grid Map
#                                                                          Cost: 13
#|       |       |    |--|--pm---|       | pm    |       |       | House-|----   |    
#|       |       | House |       | Hosp  |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       |   |   |
#|       |       |       |       |   |   |       |       |       |       | pm|   |
#|       | House |-------|-------|----   |       |   |---|-------|--Hosp-|       |
#|       |       |       |       |       |       | House |       |       | pm    |
# Note: pm = possible move

# And we can do even better by taking the same hospital and moving it one square to the left.
# Now instead of a cost of 13, we have a cost of 11. See example below

# Grid Map
#                                                                          Cost: 11
#|       |       |       |       |       |       |       |       | House |       |    
#|       |       | House |  Hosp |       |       |       |       |       |       |
#|       |       |       |       |       |       |       |       |       |       |
#|       |       |       |       |       |       |       |       |       |       |
#|       | House |-------|       |       |       |   |---|-------|--Hosp |       |
#|       |       |       |       |       |       | House |       |       |       |
# Note: pm = possible move

# So we've been able to do much better than that initial cost that we had using the initial configuration,
#just be taking every state and asking ourselves the question, can we do better by just making small
#incremental changes. Moving to a better neighbor reapeatedly.

# And now at this point, we can potentially see that this algorithm is going to terminate.
# There's actually no neighbor we can move to, that is going to improve our current state, or get us a 
#cost that is less than 11.

# So the question we might now ask is, is this the best we can do?
# Is this the best placement of the hospitals we can possibly have?
# And it turns out the answer is no, because there is a better way we can place the hospitals.

# And in particular, there are a number of ways we can do this, but one of the ways is by taking the
#hospital on the left and moving it diagonally to a closer square, giving us a new cost of 9
#which was not part of our definition of neighbor, where we could only move left, right, up, or down. 
# See example below.

# Grid Map
#                                                                          Cost: 9
#|       |       |       |       |       |       |       |       | House |       |    
#|       |       | House |       |       |       |       |       |       |       |
#|       |       | Hosp  |       |       |       |       |       |       |       |
#|       |       |       |       |       |       |       |       |       |       |
#|       | House |       |       |       |       |   |---|-------|--Hosp |       |
#|       |       |       |       |       |       | House |       |       |       |
# Note: pm = possible move

# But we weren't able to find it because in order to get there we had to go through a state that actually
#wasn't any better than the current state that we had been in previously.

# This appears to be a limitation, or a concern that we might have as we go about trying to implement a 
#hill climbing algorithm, is that it might not always give us the optimal solution.

# If we are trying to maximize the value of any particular state, we're trrying to find the global maximum,
#a concern might be that we could get stuck at one of the local maxima.
# A local maxima is any state whose value is higher than any of its neighbors.

# The same issue can also occur when looking for the global minimum. We might get stuck in a local minima.
# A local minima is any state whose value is lower than any of its neighbors.

# The takeaway here is that it's not always going to be the case when we run our hill climbing algorithm
#that we are always going to find the #optimal solution. There are things that can go wrong.


# And other problems that we might imagine, just by taking a look at our state space landscape, are these
#various different plateaus.
# Something like this flat local maximum below, where all 4 of the even states each have the exact same value,
#and so, in the case of the algorithm we showed before, none of the neighbors are better, so we might just
#get stuck at this flat local maximum. 

# And even if we allowed ourselves to move to one of the neighbors, it wouldn't be clear which neighbor we
#would ultimately move to. We might get stuck here as well.

    

#               |       Flat Local Maximum
#               |
#               |
#               |   |
#               |   |   |
#               |   |   |   |   |   |   |
#       |   |   |   |   |   |   |   |   |
#       |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |


# There is another example of a flat plateau to the left, which is represented by 2 even, equally valued states.
# This one is called a shoulder. It's not really a local maximum because there are still places we can move 
#higher, and not a local minimum because we can also go lower.
# So we can still make progress, but it's still this flat area, where if we have a local search algorithm,
#it has the potential to get lost here, unable to make some upward or downward progress depending on whether
#we're trying to maximize or minimize, and therefore, another potential for us to be able to find the solution
#that might not actually be the optimal solution



#               |            
#               |
#    Shoulder   |
#               |   |
#               |   |   |
#               |   |   |   |   |   |   |
#       |   |   |   |   |   |   |   |   |
#       |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |
#   |   |   |   |   |   |   |   |   |   |


# And so because of this potential, the potential that hill climbing has to not always find the us the
#optimal result, it turns out that there a number of different variaties and variations on the the 
#hill climbing algorithm that help to solve the problem better depending on the context.
# And depending on the specific type of problem, some of these variants might be more applicable then
#others. 

# What we've taken a look at so far is a version of hill climbing generally called steepest ascent hill climbing.
# Where the idea of steepest ascent hill climbing is we are going to choose the highest valued neighbor, in the
#case where we are trying to maximize, and the lowest valued neighbor in the case where we are trying to 
#minimize. 
# But generally speaking, if we have 5 neighbors, and they are all better than our current state, we will pick
#the best one of those 5. 

# Now sometimes that might work pretty well. A sort of gradient approach, trying to take the best operation
#at any particular time step. But it might not always work. There might be cases where actually we want to choose
#an option that is slightly better than us, but is not the best one, because that might later on lead to a 
#better outcome ultimately.

# So there are other variants that we might consider of this basic hill climbing algorithm.

# One is known as the stochastic hill climbing.
# And in this case we choose randomly from all of our higher valued neighbors.

# So if we are at our current state, and there are 5 neighbors that all better than we are, rather than choosing
#the best one, as steepest ascent would do, stochastic we choose randomly from one of them. Thinking that if
#it's better, then it's better, and maybe there's a potential to make forward progress, even if it is not locally
#the best option we can possibly choose.

# First choice hill climbing, ends up just choosing the very first highest valued neighbor that it follows,
#behaving in a similar idea, rather than consider all of the neighbors, as soon as we find a neighbor that is
#better than our current state, we'll go ahead and move there.
# So maybe some efficiency improvemnets there, and maybe has the potential to find the solution that the other
#stragtegies weren't able to find. 

# And with all of these variants, we still suffer from the same potential risk. This risk where we might end up
#at a local minimnum, or a local maximum.
# We can reduce that risk by repeating the process multiple times.


# So one variant in hill climbing is random restart, where the general idea is we'll conduct hill climbing
#multiple times.
# If we apply steepest acsent hill climbing, for example we'll start at some random state, try and figure out 
#how to solve the problem, and figure out what is the local maximum or local minimum we get to, and then we'll 
#just randomly restart, and try again. Choose a new starting configuration, try and figure out what the local
#maximum and minimum is, and do this some number of times, and after we've done it some number of times, we
#can pick the best one out of all the ones we've taken a look at.
# So there's another option we have access to as well.

# And then, even though we said that generally local search usually just keeps track of a single node, and then 
#moves to one of its neighbors, there are variants of hill climbing that are known as local beam searches,
#where rather than just keep track of one current rest state, we're keeping track of k highest valued neighbors.
# Such that rather than starting at one random initial configuration, we might start with 3 or 4 or 5, randomly
#generate all of the neighbors, and then pick the 3 or 4 or 5 best of all of the neighbors that we find, and
#continually repeat this process, with the idea being that now we have more options that we are considering,
#more ways that we can potentially navigate ourselves to the optimal solution.that might exist for a particular
#problem. 


#     Hill Climbing Variants

#     Variant      ||       Definition
#-----------------------------------------------------------
# Steepest-ascent  || Choose the highest valued neighbor
# Stochastic       || Choose randomly from higher-valued neighbors
# First-choice     || Choose the first higher-valued neighbor
# Random Re-start  || Conduct Hill Climbing multiple times
# Local Beam Search|| Chooses the k highest-valued neighbors


# So let's now take a look at some actual code that can implement these kinds of ideas. Something like,
#Steepest ascent hill climbing for example, for trying to solve this hospital problem.

import random


class Space():
  def __init__(self, height, width, num_hospitals):
    "Create a new state space with given dimensions."
    self.height = height
    self.width = width
    self.num_hospitals = num_hospitals
    self.houses = set()
    self.hospitals = set()

  def add_house(self, row, col):
    "Add a house at aparticular location in state space."
    self.houses.add((row, col))

  def available_spaces(self):
      "Returns all cells not currently used by a house or hospital"

      # Consider all possible cells
      candidates = set(
         (row, col)
         for row in range(self.height)
         for col in range(self.width)
      )

      # Remove all houses and hospitals 
      for house in self.houses:
         candidates.remove(house)
      for hospitals in self.hospitals:
         candidates.remove(hospitals)
      return candidates
  
  def hill_climb(self, maximum=None, image_prefix=None, log=False):
     "Performs hill climbing to find a solution"
     count = 0


    # Start by intializing hospitals randomly
     self.hospitals = set()
     for i in range(self.num_hospitals):
        self.hospitals.add(random.choice(list(self.available_spaces())))
     if log:
        print("Intial state: cost", self.get_cost(self.hospitals))
     if image_prefix:
        self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")


     # Continue until we reach maximum number of iterations
     while maximum is None or count < maximum:
        count += 1
        best_neighbors = []
        best_neighbor_cost = None


        # Consider all hospitals to move
        for hospital in self.hospitals:
           
           # Consider all neighbors for the hospital
           for replacement in self.get_neighbors(*hospital):
              
              # Generate a neighboring state of hospitals
              neighbor = self.hospitals.copy()
              neighbor.remove(hospital)
              neighbor.add(replacement)

              # Check if hospital is best so far
              cost = self.get_cost(neighbor)
              if best_neighbor_cost is None or cost < best_neighbor_cost:
                 best_neighbor_cost = cost
                 best_neighbors  = [neighbor]
              elif best_neighbor_cost == cost:
                 best_neighbors.append(neighbor)


           # None of the neighbors are better than the current state
           if best_neighbor_cost >= self.get_cost(self.hospitals):
              return self.hospitals
           
           # Move to highest valued neighbor
           else:
              if log:
                 print(f"Found better neighbor: cost {best_neighbor_cost}")
              self.hospitals = random.choice(best_neighbors)

           # Generate Image
           if image_prefix:
              self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")
         
         # Here we will implement a function for random restart     
        def random_restart(self, maximum, image_prefix=None, log=False):
           "Repeats hill climbing multple times"
           best_hospitals = None
           best_cost = None

           # Repeat Hill-Climbing a fixed number of times
           for i in range(maximum):
              hospitals = self.hill_climb()
              cost = self.get_cost(hospitals)
              if best_cost is None or cost < best_cost:
                 best_cost = cost
                 best_hospitals = hospitals
                 if log:
                    print(f"{i}: Found new best state: cost {cost}")
              else:
                 if log:
                    print(f"{i}: Found State: cost {cost}")

              if image_prefix:
                 pass
              else:
                 SystemExit
def done():
  pass
                 


# This is just an example of how we can start the code that can implement the ideas of hill climbing

# These types of algorithms can be quite useful when solving these problems.
              
# But the real problem with many of these different types of hill climbing variants is that they never make a
#move that makes our situation worse. They're always going to cells in our current state, look at the 
#neighbors and consider if can we do better than our current state, and move to one of those neighbors.
# Which of the neighbors we choose may vary based on the algorithms, but we never go from a current position
#to a position that is worse than our current position.
# Ultimately, that is what we are going to need to do, if we want to be able to find global maximum
#or a global minimum, because sometimes if we get stuck, we wanna find some way of dislodging ourslves from
#our local maximum and local minimum, in order to find the global maximum or global minimum, or increase 
#the probability that we do find it.

# And so the most popular technique for trying to approach the problem from that angle, is a technique known
#as simulated annealing.
# Simulated because it is modeling after a real physical process of annealing.
# We can think about this in terms of physics. A physical situation, where we have some system of particles,
#and we might imagine we that when we heat up a particulr physical system, there's a lot of energy there,
#things are moving around quite randomly, but for times when the system cools down, it eventually settles
#into some final position.

# And that is going to be the general idea of simulated annealing. We are going to simualte that process of some high
#temperature system, where things are moving around randomly quite frequently, but over time decreasing that
#temperature, until we eventually settle at our final solution.

# And the idea is going to be, if we have some state space landscapes that look like this, and we begin at 
#its initial state, where ever it is, if we're looking for a global maximum, and we're trying to maximize
#the value of a state, our traditional hill climbing algorithms will just take that state and look at the two neighbors,
#and always pick the one that is going to increase the value of the state.
# See example below




#               STATE SPACE LANDSCAPE

# Global  | Maiximum

#         |                                         
#         |                                         
#         |                                         
#         | |                                       
#         | |               |                       
#   | | | | | |           | |                         |
#   | | | | | | |       | | | |                     | |
#   | | | | | | | |   | | | | |       | | | | | |   | |
# | | | | | | | | | | | | | | | |     | | | | | | | | |
# | | | | | | | | | | | | | | | |   | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
# | | | | | | | | | | | | | | | | | | | | | | | | | | |
#       Initial state |

# So if we want some chance of being able to find the global maximum, we can't always make the good moves. We have to
#sometimes make bad moves, and allow ourselves to make a move in a direction that actually seems like it makes 
#our situation worse in the present, such that later we can find our way up to our global maximum.

# And once we get to our global maximum, then we don't want to be moving to states that are worse than our 
#current state. And so this is where this metaphor for annealing starts to come in. 
# Where we want to start making more random moves, and over time start to make fewer of those random moves,
#based on a particular temperature schedule.


#       Simulated Annealing -
#                           - Early on, higher "temperature": more likely to acccept neighbors that are 
#                             worse than current state.
#                           - Later on, lower "temperature": less likely to accept neighbors that are worse
#                             than current state.

# Now to formalize this and put a little psuedo code to this, this is what that algorithm might
#look like. We have a function called simulated annealing that takes as input the problem we're trying to solve
#and also potentially some maximum number of times we might want to run the simulated annealing process, and
#how many different neighbors we're going to try and look for.
# And that value is going to vary based on the problem we're trying to solve.
# We'll, again, start with some current state that will be equal to the problem.
# But now we need to repeat this process over and over for max number of times.
# Repeat some process some number of times where we're first going calculate a temperature.
# And this temperature function takes the current time t, starting at 1 and going all the way up to max
#and then gives us some temperature that we can use in our computation, where the idea is that this
#temperature is going to be higher early on, and it's going to be lower later on.

# So there are a number of ways this temperature function could often work. 
# One of the simplest ways is just to say it is like the proportion of time that we still have remaining.
# Out of max units of time, how much time do we have remaining.
# We start off with a lot of that time remaining.
# And as time goes on, the temperature is going to decrease, because we have less and less of that remaining
#time still available to us.

# So we calculate a temperature for current time.
# And then we pick a random neighbor of the current state.
# No longer are we going to be picking the best neighbor that we possibly can, or just one of the better 
#neighbors that we can. 

# We're going to pick a random neighbor.
# It might be better. It might be worse.
# But we're going to calculate that.

# We're going to calculate delta E, E for energy in this case, which is just how much better is the neighbor
#than the current state.
# So if delta E is positive, that means the neighbor is better than our current state.
# If delta E is negative, that means the neighbor is worse than our current state.

# And so we can then have a condition that looks like this.

# If delta is greaterr than 0, that means the neighbor state is better than our current state.
# And if ever that situation arises, we'll just go ahead and update current to be that neighbor.
# Same as before, move where we are currently, to the neighbor state because the neighbor state is better 
#than our current state.
# We'll go ahead and accept that.

# But now the difference is that whereas before, we never ever wanted to take a move that made our situation worse,
#now we sometimes want to make a move that is actually going to make our situation worse because sometimes we're
#going to need to dislodge ourselves from a local minimum or local maximum to increase the probabilty that we're
#able to find the global minimum or the global maximum a littl bit later.

# And so, how do we do that?

# How do we decide to sometimes accept some state that might actually be worse?
# Well, we're going to accept a worse state with some probability.
# And that probability needs to be based on a couple of factors.
# It needs to be based in part on the temperature, where if the temperature is higher, we're more likely to move
#to a worse neighbor.
# And if the temperature is lower, we're less likely to move to a worse neighbor.

# But it also in some degree, should be based on delta E.
# If the neighbor is much worse than the current state, we probably want to be less likely to choose that
#than if the neighbor is just a little bit worse than the current state.

# So again, there are a couple of ways we could calculate this. But it turns out that one of the most 
#popular is just to calculate E to the power of delta E over T, where E is just a constant.
# Delta E over T are based on delta E and T in our function.

# We calculate that value.
# And that will be some value between 0 and 1.
# And that is the probabilty with which we should just say, all right, lets go ahead and move to that next 
#neighbor.

# And it turns out that if we do the math for that value, when delta E is such that the neighbor is not that 
#much worse than the current state, thats's going to be more likely that we're going to go ahead and move to that
#state.

# And likewise, when the temperature is lower, we're going to be less likely to move to that neighboring state
#as well.

# So now this is the big picture for simulated annealing, this process of taking the problem and going ahead and 
#generating random neighbors will always move to a neighbor if it's better than our current state.
# But even if the neighbor is worse than our current state, we'll sometimes move there depending on how much worse 
#it is, and also based on the temperature.


# And as a result, the hope, the goal of this whole process is that as we begin to try and find our way out
#to the global maximum or the global minimum, we can dislodge ourselves if we ever get stuck at a local maximum
#or locall minimum in orer to eventually make our way to exploring the part of the state space that is going 
#to be best.
# And then as the temperature decreases, eventually we settle there without moving around too much from what we've
#found to be the globally best thing that we can do thus far.

# So at the very end, we just return whatever the current state happens to be.
# And that is the conclusion of this algorithm.

# We've been able to figure out what the solution is.



#              Simulated Annealing

#    function Simulated-Annealing(promblem,max):
#      current = initial state of problem
#      for t = 1 to max:
#          T = Temperature(t)
#          neighbor = random neighbor of current 
#    Delta E = how much better neighbor is than current
# if Delta E > 0.
#         current = neighbor
# with probability e to the power of Delta E/T set current = neighbor


# These types of algorithms have a lot of different applications.

# Anytime we can take a problem, and formulate it as something where we can explore a particular configuration,
#and then ask, are any of the neighbors better than the current configuration, and have some way of 
#measuring that, then there is an applicable case for these hill climbing simulated annealing types of
#algorithms.
# Sometimes it can be for pursuing these location type problems, like for when we're trying to find a city and 
#firgure out where the hospital should be.
# But there are definitely more applications as well.

# One of the most famous problems in computer science, is the traveling salesman problem.
 


#     Traveling Salesman Problem

#                 .
#       .    
#           .
# .
#                    .
#     .       .     .


# The traveling salesman problem generally is formulated like this.

# We have a whole bunch of cities here indicated by these dots.
# And what we'd like to do, is find some route that takes us through all of the cities, and ends up back 
#where we started.

# And what we might like to do, is minimize the total distance, that we have to travel, or total cost
#of taking this entire path.

# And we can imagine, this is a problem that is very applicable in situations like when delivery companies
#are trying to deliver things to a whole bunch of different houses.
# They wanna figure out, how do they get from the warehouse, to all these various houses, and get back again.
# All using as minimal time and distance and energy as possible. 
# So we might want to try and solve these sorts of problems.

# But it turns out, that solving this particular kind of problem, is very computationally difficult. 
# It is a very computationally expensive task to be able to figure it out. 
# This falls under the category of what are known as n p complete problems.
# Problems where there is no known sufficient way to try and solve these sorts of problems.

# So what we ultimately have to do is, come up with some approximation. Some way of trying to find a good
#solution, even if we aren't going to find the globally best solution that we possibly can.
# At least not in a feasible or trackedable amount of time.

# So what we could do is, take the traveling salesman problem, and try to formulate it using local search.
# Then ask a question, like alright, we can pick some state, some configuration, some route between 
#all of these nodes, and we can measure the cost of that state, and figure out what the distance is.
# And we might want try and minimize that cost as much as possible.

# And then the only question is, what does it mean to have a neighbor of this state.
# What does it mean to take this particular route, and have some neighboring route that is close to it,
#but slightly different, such that it might have a different total distance.

