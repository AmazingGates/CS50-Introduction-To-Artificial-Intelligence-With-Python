import scipy.optimize

# So if we want to applly Arc Consistency to a larger graph, there are ways we can do that too.

# We can begin to formalize what the psuedo code might look like for trying to write an algorithm that will
#enforce this Arc Consistency.

# And we'll start by defining a function called revise.

# Revise is going to take as input, a csp (constraint satisfaction problem)
# And also 2 variables, X and Y.

# And what Revise is going to do is,it is going to make X Arc Consistent, with respect to Y.
# Meaning remove anything from X's domain that doesn't allow for a possible option for Y.

# How does this work?
# First we'll keep track of if we made a revision.

# Revised is ultimately going to return True or False.
# It will return True in the event that we did make a revision to X's domain.
# It will return False if we didn't make any changes to X's domain.

# And we'll see in a moment why that is going to be helpful.

# But we start by saying that revised = False because we haven't made any changes.

# Then we'll say, let's go ahead and loop over all of the possible values in X's domain.
# We want to make sure that for each choice in little x, we have some choice available in Y that satisfies
#the binary constraints that are defined inside of our csp.

# So if ever is the case that there is no value Y in Y's domain that satisfies the constraint for X and Y,
#then that means that this value X shouldn't be in X's domain.
# So we'll go ahead and delete x from X's domain.

# Then we'll set revised to equal True, because we did change X's domain by removing little x.
# And we removed little x because it wasn't Arc Consistent.
# There was no way we could choose a value for Y that would satisfy our (X,Y) constraint.

# So in this case we'll go ahead and set revised equal True.

# And we'll do this again and again for every value in X's domain.

# Sometimes it might be fine, in other cases it might not allow for a possible choice for Y.
# In that case we need to remove that value from X's domain.

# And in the end, we just return revised to indicate whether or not we actually made a change. 

# So this REVISE function is effectively an implementation of what we did a moment ago, and it makes one
#variable X, Arc Consistent with another variable, in this case, Y.



#def REVISE(csp, X, Y):
#    revised = False
#    for x in X.domain:
#        if no y in Y.domain satisfies constraint for (X,Y):
#            delete x from X.domain
#            revised = True
#    return revised 


# But generally speaking, we want to enforce Arc Consistency for the entire Constraint Satisfaction Problem.

# And it turns out that there is an algorithm to do that as well.

# That algorithm is known as function AC-3(csp):

# AC-3 takes a Constraint Satisfaction Problem, and enforces Arc Consistency across the entire problem.

# How does it do that?

# It's going to basically maintain a queue.
# A line of all of the Arc's that it needs to make consistent.

# And over time, we might remove things from that queue, and we might need to add things as well, if there are more
#things we need to make Arc Consistent. 

# So we'll go ahead and start with a queue that contains all of the arcs in the constraint satisfaction problem,
#all of the edges that connect two nodes that have some sort of binary constraint between them.

# And now, as long as the queue is non-empty, there is work to be done.
# The queue is all of the things that we need to make Arc Consistent.
# So as long as the queue is non-empty, there's still things we have to do.

# What do we have to do?

# Well we'll start, by dequeuing the queue, remove something from th queue.
# Ans strictly speaking, it doesn't need to be a queue, but a queue is the traditional way of doing this.

# We'll dequeue from the queue, and that will give us an arc, X and Y, these two variables where we would 
#like to make X arc consistent with Y.

# So how do we make X arc consistent with Y?

# We can do this by just using that revise function that we talked about a moment ago.

# We called the revise function, passing as input the constraint satisfaction problem, and also these
#variables, X and Y, because we want to make X arc consistent with Y.

# In other words, remove any values from X's domain that doesn't leave an available option for Y.

# And recall that revised returns True if we actually made a change, if we rremoved something from X's
#domain, because there wasn't an available option for Y, for example.

# And it returns False if we didn't make any changes to X's domain at all.

# And it turns out that if revised returns False, if we didn't make any changes, then there is not a whole
#lot more work to be done here for this arc.

# We can just move ahead to the next arc that's in the queue.

# But if we did make a change, if we did reduce X's domain by removing values from X's domain, then we
#might realize is that this creates potential problems later on.
# It mean that some arc that was arc consistent with X, that node might no longer be arc consistent with X,
#because while there used to be an option that we could choose for X, now there might not be, because now
#we might have removed something from X that was necessary for some other arc to be arc consistent.

# And so if ever we did revise X's domain, we're going to need to add some things to the queue, some 
#additional arcs that we might want to check.

# How do we do that?

# First thing we want to check is that X's doamin does not equal zero.
# If X's domain is zero, that means that there are no available options for X at all.
# That means that there's no way to solve the constraint satisfaction problem.
# If we removed everything from X's domian, we'll just go ahead and return False here to indicate
#there's no way to solve the problem, because there's nothing left in X's domain.

# But otherwise, if there are things left in X's domain, but fewer things than before, then what we'll 
#do is, we'll loop over each variable Z that is in all of X's neighbors, except for Y, Y we already handled.

# But we'll consider all of X's other neighbors and ask ourselves, will that arc from each of those Z's to X,
#that arc might no longer be arc consistent, because while for each Z, there might have been a possible
#option we could choose for X to correspond with each of Z's possible values, now there might not be
#because we removed some elements from X's domain.

# And so what we'll do here is go ahead and enqueue, adding something to the queue, this arc Z and X, for all
#those neighbors Z.

# So we need to add back some arcs to the queue in order to continue to enforce arc consistency.

# At the very end, if we make it through all of this process, then we can return True.

# But this now is AC-3, this algorithm for enforcing arc consistency, on a constraint satisfaction problem.

# And the big idea is really just keep track of all of the arcs that we might need to make arc consistent,
#make it arc consistent by calling the revise function.

# And if we did revise it, then there are some new arcs that might need to be added to the queue in order
#to make sure that everything is still arc consistent, even after we've removed some of the elements from
#a particular variable's domain.  
 


#function AC-3(csp):
#   queue = all arcs in csp:
#   while queue non-empty:
#    (X,Y) =  DEQUEUE(queue)
#    if Revised(csp,X,Y)
#       if size of X.domain == 0:
#          return False
#       for each Z in X.neighbors - {Y}:
#          ENQUEUE(queue,(Z,X))
#   return True


# So what then would happen, if we tried to enforce arc consistency on a graph like this.
# On a grapd where each of these variables has a domain of {Mon, Tues, Wed}.

# Well it turns out that by enforcing arc consistency on this graph, while it solve some types of problems
#nothing actually changes.

# For any particular arc, just considering two variables, there's always a way for any of the changes we
#make for one of them, make a choice for the other ones, because there are three options, and we just 
#need two of them to be different from each other. 

# This is actually quite easy, to just take an arc, and declare that it is arc consistent.
# For example, if we pick {Monday} for D, then we just pick something that isn't {Monday} for B.

# In arc consistency, we only consider consistency between a binary constraint between two nodes, 
#and we're not really considering all of the rest of the nodes.  

# So just using AC-3, the enforcement arc consistency, that can sometimes have the effect of reducing domains,
#making it easier to find solutions, but it will not always actually solve the problem.
# We might need to still somehow search to try and find a solution.

# And we can use classical traditional serach algorithms to try and do so.




#              Contraints Graph:
#
#   {Mon, Tues, Wed}  .A
#                    /   \
# {Mon, Tues, Wed}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# If we recall, that a search problem general consists of these parts

#        Serach Problems

# Initial State
# Actions
# Transition Model
# Goal Test
# Path Cost Function


# So, we could formulate a csp (constraint satisfaction problem), as one of these types of search problems.

#        CSP's as Search Problems

# Initial State: Empty assignment (No variables)
# Actions: Add a {variable = value} to assignment
# Transition Model: Shows how adding an assignment changes the assignment
# Goal Test: Check if all variables assigned and constraints are satisfied
# Path Cost Function: All paths have same cost 

# The problem here though, is that if we just implement this naive search algorithm just implementing
#something like breadth first search, or depth first search for example, this is going to be very ineffecient.

# And there are ways we can take advantage of effeciencies in a structure of a constraint satisfaction problem
#itself. 

# And one of these ideas is that we can really just order these variables.
# It really doesn't matter what order we assign variables in.
# The assignment A = 2 and then B = 8 is identical to the assignment B = 8 and then A = 2.
# Switching the order doesn't really change anything about the fundamental nature of that assignment.

# And so there are some ways we can try and revise this idea of a search algorithm, to apply specifically
#for a problem like a constraint satisfaction problem.

# And it turns out that the search algorithm we'll generally use when talking about constarint satisfaction
#problems, is a backtracking search.

# And the big idea behind backtracking search is we'll go ahead make assignments from  variables to
#values, and ever we get stuck, where we can no longer make any forward progress within our constraints,
#we'll go ahead and backtrack, and try something else instead.

# So the very basic sketch of what backtracking search looks like is this



#        Backtracking Search

#function BACKTRACK(assignment, csp):
#  if assignment complete: return assignment
#  var = SELECT-UNASSIGNED-VAR(assignment, csp)
#  for value in DOMAIN-VALUES(var, assignment, csp):
#    if value consistent with assignment:
#       add {var = value} to assignment
#       result = BACKTRACK(assignment, csp)
#       if result != failure: return result
#     remove {var = value} from assignment
#  return failure


# This is the idea for backtracking search.
# Take each of the variables, try values for them, and recursively try backtracking to see if we can make
#progress.

# And if ever we run into a dead end we return failure, which takes to the top of the algorithm and try
#something else instead.


# So now, let's put this algorithm into practice.

# Let's use backtracking search to actually try and solve this problem.

# We need to find out how to assign each of these courses to an exam slot, in such a way that it satisfies
#these constraints. Constraints {Mon, Tues, Wed} 

# Each of these edges means that those two classes cannot have an exam on the same day.



#              Contraints Graph:
#
#   {Mon, Tues, Wed}  .A
#                    /   \
# {Mon, Tues, Wed}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}



# So we can start by just starting at a node. It doesn't really matter which one.
# But in this case let's just start with A.

# Now we'll loop over the values in the domain and decide to start with Monday. 

# So we'll go ahead and assign A to Monday.

#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
# {Mon, Tues, Wed}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# Next let's consider node B.

# So we've made an assignment to A and recursively call backtrack, and with this new assignment we're free to
#pick another unassigned variable like B.

# And this time we'll start with Monday again, because it is the first value and we are not randomly selecting
#right now.

# Then we ask ourselves if by picking Monday for B, do we violate any constraints?
# It turns out yes we do.
# It violates the constraint because A and B are now on Monday and that vilotes the constraint that says
#two nodes connected by an edge cannot have an exam on the same day.
# That doesn't work, so we'll instead try Tuesday, the next value in B's domain.
# Now we'll ask is that consistent with the assignment?
# The answer is yes. A and B now have exams on different days.

# That is good so far, now we can recursively call backtrack and try another unassigned variable.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# Now we'll pick D and go through it's possible values by looping over its domain.

# Again, we'll choose the first value, Monday, and see if the assignment is consistent.
# So far yes it is. Because B is on a Tuesday so that doesn't conflict with the constraints.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#           {Mon}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# So we'll try again and move on to the variable E.
# We'll see if we can make that consistent by looping through the possible values, recursively call backtrack,
#and start the process over.
# We'll try to start with Monday, but we'll notice that D and E cannot have an exam on the same day, so the
#would make that option unconsistent.
# So we'll try the next value and try that one, Which is Tuesday.
# The answer is no, because B and E have are connected by an edge so they can't have an exam on the same day.
# So we'll move on to the next value, which is Wednesday and check to see if that is consistent.
# It turns out that yes it is because it doesn't violate any constraints.

# Now we recursively call backtrack and check another unassigned variable.

#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#           {Mon}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
#            {Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# This time we will choose C to loop over. 
# Again, we'll start with Monday, which isn't a good option because of the edge with A.
# Next we'll try Tuesday, which isn't a good option because of the edge with B.
# And lastly, we'll try Wednesday, which isn't a good option either because of the edge with E.
# So now we've gone through all the possible values for C {Mon, Tues, Wed}, and none of them are consistent.
# Since there's no way we can have a consistent assignment, backtrack in this case will return a failure.

# We'll move on by backtracking to E
# Now for E, we've tried all of the values in its domain {Mon,Tues, Wed}, and none of those worked, because 
#it turns out Wednesday led to a failure.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#           {Mon}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# So that means that there is no possible way we can assign E, so that's a failure too, which means we have
#to backtrack to D.
# This means that Monday's assignment to D must be wrong so we have to try something else.
# Since Tuesday would be unconsistent, the only other option for D would be Wednesday.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# Now we can begin to make forward progress again.
# We'll go back to E and figure out which of these values work.
# Monday turns out to work without violating any constraints.

#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# Now we can go on to C and check it again.
# Monday doesn't work because it violates a constraint.
# Tuesday doesn't work because it violates a constraint.
# It turns out the only option for C which is consistent is Wednesday

#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# Now we can move on to F and loop over its values.
# Monday doesn't work because of the edge with E.
# Wednesday doesn't work because of its edge with C.
# That leaves the only consistent option for F to be is Tuesday

#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Tues}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# And finally we can move on to our last unassigned variable, G, recursively calling backtrack one more time.
# The only option we have for the variable G that stays consistent is Wednesday.

#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Tues}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Wed}


# At this point, we now call recursive backtrack one last time.
# We now have a satisfactory assignment of all of the variables, and at this point, we can that we are done.

# We have be successfully able to assign a value to each one of these variables in such a way that we are
#not violating any constraints.

# So this is a graphical look at how this might work.

# Let's now take a look at some code we could use to actually try and solve this problem as well.

VARIABLES = ["A", "B", "C", "D", "E", "F", "G"]
CONSTRAINTS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "E"),
    ("C", "F"),
    ("D", "E"),
    ("E", "F"),
    ("E", "G"),
    ("F", "G")
]

def backtrack(assignment):
    """ Runs backtracking search to find an assignment """

    # Check if assignment is complete
    if len(assignment) == len(VARIABLES):
        return assignment
    
    # Try a new variable
    var = select_unassigned_variable(assignment)
    for value in ["Monday", "Tuesday", "Wednesday"]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        if consistent(new_assignment):
            result = backtrack(new_assignment)
            if result is not None:
                return result
    return None

def select_unassigned_variable(assignment):
    """ Chooses a variable not yet assigned, in order """

    for variable in VARIABLES:
        if variable not in assignment:
            return variable
    return None

def consistent(assignment):
    """ Checks to see if an assignment is consistent """

    for (x,y) in CONSTRAINTS:
        # Only consider arcs where both are assigned
        if x not in assignment or y not in assignment:
            continue

        # If both have the same value, then not consistent
        if assignment[x] == assignment[y]:
            return False
        
    # If nothing is inconsistent, the assignment is consistent
    return True

solution = backtrack(dict())
print(solution) # Output 
# {'A': 'Monday', 'B': 'Tuesday', 'C': 'Wednesday', 'D': 'Wednesday', 'E': 'Monday', 'F': 'Tuesday', 'G': 'Wednesday'}

# Notice that all the assignments are just as we predicted when we walked through the process with the 
#Constraints Graph


# This was the implementation of a very simple backtracking search method. Where really we just went through
#each of the variables, picked one that wasn't assigned, tried the possible values the variable can take on,
#and if it didn't violate any constraints, then we kept trying other variables.

# And if ever we hit a dead end, we had to backtrack.

# But ultimately, we might be able to be a little more intelligent about how we do this, in order to more
#effeciently solve these types of problems.

# And one thing we might imagine trying to do, is going back to this idea of inference.
# Using the knowledge we know, to be able to draw conclusions, in order to make the rest of the problem solving
#process a little bit easier.

# And let's now go back to where we got got stuck in this problem the first time.

# When we were solving this constraints satisfaction problem.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# We dealt with B, and then we went on to D, and we went ahead and just assigned D to Monday.
# Beacause that seemed to work with the assignment so far.
# It didn't violate any constraints.
# But it turned out, that later on that choice turned out to be a bad one.
# That choice wasn't consistent with the rest of the values.

# The question is, is there anything we can do to avoid getting into a situation like this.
# Avoid trying to go down a path that ultimately isn't going to lead anywhere, by taking advantage of 
#knowledge that we have initially.

# And it turns out, we do have that kind of knowledge.

# We can look at the structure of the graph so far, and we can say, right now that C's domain, for example,
#contains values {Mon, Tues, Wed}.

# And based on those values, we can say that this graph is not Arc Consistent. 

# Remember that arc consistency is all about making sure that for every possible value for a particular 
#node, that there is some other value that we are able to choose.

# And as we can see, Monday and Tuesday are not going to be possible values that we can choose for C.
# They are not going to be consistent with A or B, beacause they would end up with exams on the same day.

# So using that information and making C arc consistent with A and B, we can just make C Wednesday.

# And if we continue to try and enforce arc consistency, we see there are some other conclusions we can
#draw as well.

# We see that B's only option is Tuesday, and C's only option is Wednesday.



#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# And so if we wanted to make E arc consistent, it can't be Tuesday because of the edge with B, and it can't
#be Wednesday because of its edge with C.
# So we can just go ahead set that to Monday, for example.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# And then we can begin to do this process again and again

# In order to make D arc consistent with E, then D would have to be Wednesday.
# That's the only possible solution to keep it arc consistent


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# And likewise we can make the same judgement for F and G as well.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#           {Wed}.D  |   /   .F {Tues}
#                  \ |  /   /  \
#            {Mon}  .E /___/    \
#                    \_________.G {Wed}