# # And it turns out, that without having to do any additional search, just be enforcing arc consistency,
#we were able to actually figure out what the assignment of all the variables should be, without needing
#to backtrack at all.

# And the algorithm to do this is called maintaining arc-consistency

#       Maintaining Arc-Consistency -
# - Algorithm for enforcing arc-consistency everytime we make a new assignment.

# So sometimes we can enforce consistency using the AC-3 algorithm, at the very beginning of the problem, before
#we even begin searching, in order to limit the domain and the variables, in order to make it easier to search.

# But we can also take advantage of the interleeving of enforcing arc-consistency with search, such that everytime
#in the search process we make a new assignment, we go ahead and enforce arc-consistency as well, to make sure
#we are just eliminating possible values from domains whenever possible.

# And how do we do this?

#       Maintaining Arc-Consistency 
# When we make a new assignment to X, calls AC-3, starting with queue of all arcs(Y,X), where Y is a neighbor of X

# This is really just equivalent to everytime we make a new assignment, to variable X, we'll go ahead and call our
#AC-3 algorithm. This algorithm that enforces arc-consistency on a constraints satisfaction problem.
# And we go head and call that, starting with the queue of all the arcs that we want to make consistent with X.
# This thing that we just made an assignment to. 
# So all arcs (Y,X), where Y is a neighbor of X.
# Like something that shares a constraint with X, for example.

# And by maintaining arc-consistency in the backtracking search process, we can ultimately make our search process
#a little bit more effecient.

# And so this is a revised version of our backtrack function.
# Everytime we add a new variable equals value to our assignment, we'll go ahead and run the inference procedure,
#which might do a number of fifferent things, but one thing it can do is call the maintaining arc-consistency
#algorithm, to make sure we are able to enforce arc consistency on the problem.

# And we might be able to draw new inferences as a result of that process.
# This variable needs to be equal to that value, for example.

# And so long as those inferences are not a failure, as long as they don't lead to a situation where there is no
#possible way to make forward progress, well then we can go ahead and add those inferences.
# The new pieces of knowledge we know about what variables should be assigned to what values.
# We can add those to the assignment, in order to more quickly make more forward progress, by taking advantage of
#information that we can deduce based on the rest of the structure of the constraints satisfaction problem.

# And the only other change we'll need to make now is if it turns out that this value doesn't work, we'll need
#to remove variable equals value, and also, any of those inferences that we made, remove that from the assignment
#as well.


#def BACKTRACK(assignment,csp):
#    if assignment complete:
#        return assignment
#    var = SELECT-UNASSIGNED-VAR(assignment,csp)
#    for value in DOMAIN-VALUES(var,assignment,csp):
#      if value consistent with assignment:
#         add {var = value} to assignment
#         inferences = INFERENCE(assignment,csp)
#         if inferences != failure:
#            add inferences to assignment
#            result = BACKTRACK(assignment,csp)
#            if result != failure:
#               return result
#      remove {var = value} and inferences from assignment
#    return failure


# And so here then, we're often able to solve the problem by backtracking less than we originally had needed to,
#just by taking advantage of the fact that everytime we make a new assignment of one variable to one value, 
#that might reduce the domain of other variables as well, and we can use that information to more quickly begin
#to draw conclusions, in order to try and solve the problem more efficiently as well.

# It turns out that there are other heuristics we can use to try and improve the efficiency of our search processing
#as well.

# And it really boils down to a couple of these functions that we've talked about, but haven't really talked
#about how they're working. 

# And one of them is the SELECT-UNASSIGNED-VAR Function.
# Where we are selecting some variable in the constraints satisfaction problem that has not yet been assigned.

# So far we've been selecting variables randomly, just by picking one unassigned to loop over to deciding 
#where to go from there.

# But it turns out that just by being a little more intelligent, by following certain heuristics, we might
#be able to make the search process much more efficient, just by choosing, very carefully, which variable
#we should explore next.


#           SELECT-UNASSIGNED-VAR -

# - Minimum remaining values (mrv) Heuristic: Select variable that has the smallest domain
# - Degree Heuristic: Select variable that has the highest degree

# So what would it look like if we used select-unassigned-var on our search problem?

# Let's take another look at our constraints graph

# In this particular case, we've made an assignment for A, B, and D already.

# And the question is, what should we look at next. 

# According to the minimum remaining values hueristic, what we should choose is the variable that has the
#fewest remaining possible values.

# And in this case, that is going to be node C, which only has a value of {Wed}, which is a very reasonable
#choice of next assignment to make, because we know it's the only option.
# We know that the only possible option for C is Wednesday, so we might as well make that assignment and 
#then potentially explore the rest of the space after that.


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
#            {Tues}.B -----.C {Wed}
#                  / |    / \
#      {Mon, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed}


# But meanwhile, at the very start of the problem, when we didn't have any knowledge of what nodes should 
#have what values yet, we still had to pick what node should be the first one that we try and assign a value 
#to.

# And for convience, we just choose the node at the top, Node A originally.

# But we can be more intelligent about that.

# We can look at this particular graph.
# All of the nodes have the same domains of the same size, domain of size 3.
# So minimum remaining values doeesn't really help us there.

# But we might notice that node E has the highest degree.
# It has the most connections to the other nodes.

# And so perhaps it makes sense to start our search there.
# Start by searching node E, because from there, that's going to much more easily allow us to enforce
#the constraints that are near by, eliminating large portions of the search space that we might not need
#to search through.

# And in fact, by starting with E, we can immediately assign other variables.
# And following that, we can actually assign the rest of the variables without needing to do any backtracking
#at all, even if we're not using the inference procedure.

# Just by starting with the node that has a high degree, that is very quickly going to restrict the possible
#values that other nodes can take on.

# So that then is how we can go about selecting an unassigned variable in a particular order.

# Rather than randomly picking a particular order, if we are a little bit intelligent about how we choose it,
#we can make our search process much more efficient by making sure we don't have to search through portions 
#of the search space that ultimately aren't going to matter. 


#              Contraints Graph:
#
#   {Mon, Tues, Wed}  .A
#                    /   \
# {Mon, Tues, Wed}.B -----.C {Mon, Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues, Wed}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Mon, Tues, Wed} # Note, we should start our search here at E because it has a 
#high degree (most connections to other nodes.)


# The other variable we haven't talked about, the other function here, is the domain-values function.

# The Domain-Values function that takes a variable and gives us back a sequence of all of the values inside 
#of that variable's domain.

# The naive way to approach it is what we did before, which is just go in order, Monday, then Tuesday,
#then Wednesday.

# But the problem is that going in that order might not be the most efficient order to search in, sometimes,
#it might be more efficient to choose values that are likely to be solutions first, and then go to 
#other values.

# Now, how do we assess whether a value is liklier to lead to a solution or less likelier to lead to 
#a solution?

# Well one thing we can take a look at is how many constraints get added, how many things get removed
#from domains as we make this new assignment of a variable to this particular value.

# And the hueristic we can use here is the least constraining value huesristic, which is the idea that we 
#should return variables in order based on the nnumber of choices that are ruled out for neighboring values.
# And we want to start with the least constraining value, the value that rules out the fewest possible
#options.

# The general idea is, that when we are picking a variable, we would like to prune large portions of the search
#space by just choosing a variable that is going to allow me to quickly elminate possible options.


#       Domain-Values-
# - Least-Constraining Values huesristic: Return variables in order by number of choices that are ruled out
#for neighboring variables.
#       - Try least-constraining values first

# So an example of this might be this situation here, if we're trying to choose a variable for a value for
#node C, where C is equql to either Tuesday or Wednesday.
# We know it can't be Monday Because that conflicts with node A.

# So the question is, should we try Tuesday first, or should we try Wednesday first?

# If we tried Tuesday, what gets ruled out?
# B, E, and F all would get ruled out because Tuesday is a value in their domain.

# If we chose Wednesday, we would be ruling out B and E, because Wednesday is a value in their domains.

# So we have two choices. We can choose Tuesday that rules out 3 options, or choose Wednesday that rules out
#2 options.

# According to the least-constraining value hueristic, what we should probably do is go ahead and choose 
#Wednesday, the one that rules out the fewest number of possible options, leaving open as many chances as 
#possible for us to eventually find the solution inside of the state space. 


#              Contraints Graph:
#
#              {Mon}  .A
#                    /   \
# {Mon, Tues, Wed}.B -----.C {Tues, Wed}
#                  / |    / \
#{Mon, Tues, Wed}.D  |   /   .F {Mon, Tues}
#                  \ |  /   /  \
# {Mon, Tues, Wed}  .E /___/    \
#                    \_________.G {Wed}


# So the big takeaway now with all of this is that there a number of different ways we can formulate a problem.

# Some of the example we look at in this module are

#           Problem Solutions -
# - Local Search
# - Linear Programming
# - Constraint Satisfaction

# And the takeaway from all this is if we have problems that we would like AI to solve, if we can formulate
#that problem, as one of these sorts of problems, then we can use these known algorithms and techniques 
#to begin to solve a whole variety of problems all in this world of optimization inside of artificial intelligence.
