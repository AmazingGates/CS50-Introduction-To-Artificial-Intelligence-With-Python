# And there are a number of total definitions for what a neighbor of traveling salesman configuration
#might look like.

# But one way is just to say, a neighbor is what happens if we pick 2 of these edges, between nodes,
#and switch them, effectively.

# For example, let's say we take node 3 and switch it with node 4, and then switch node 6 with node 7.

# And what that pocess would generally look like is, removing both edges from the graph, taking node 3,
#connecting it to the node that it wasn't connected to, connecting to node 6 instead.



#     Traveling Salesman Problem

#                 .6
#       .2    
#           .3
# .1
#                      .5
#     .8       .7     .4


# This is what our new travel path will look like after we connected the original node 3 to the original node 6.


#     Traveling Salesman Problem

#                 .4
#       .2    
#           .3
# .1
#                      .5
#     .8       .7     .6


# So by taking two edges, and just switching them, we have been able to consider one possible neighbor of this
#particular confighuration.

# It looks like this neighbor is actually better. It looks like it probably travels a shorter distance in order
#to get to through all the cities through this route than the current state did.

# So we could imagine implementing this idea inside of a hill climbing, or simulated annealing algorithm, where
#we repeat this process and try to take a state from the traveling salesman problem, look at all the neighbors
#and then move to the neighbors if they are better, or maybe even more to the neighbors if they're worse, until
#we eventually settle on some best solution we are able to find.

# It turns out that these type of approximation algorithms, even if they don't find the very best solution, they do
#pretty well at trying to find solutions that are helpful.

# That was a look at local search. A particular category of algorithms that we can use for solving a particular
#type of problem. We don't really care about the path of the solution, we just care about the solution itself.


# Another type of algortithm that might come up, are known as these category of linear programming types of problems.
# And linear programming often comes up in the context when we are trying to optimize for some mathematical function.
# But often time linear programming will come up when we have to deal with real number values.
# It's not just discreet fixed values we might have, but any decimal values that we might want to be able to calculate.

# And so, linear programming is a family of types of problems, where we might have a situation that looks like this.
# Where the goal of linear programming is to minimize a cost function. We can also reverse the numbers to try and 
#maximize it. But often we'll frame it as trying to minimize a cost function. 
# It has some number of variables X1, X2, X3, all the way up until Xn, or some number of variables that are involved
#things that we want to know the values to.
# And this cost function might have coefficients in front of those variables. 
# And this is what we are calling a linear equation.
# We just have all of these variables that might be multiplied by a coeffiecient and then added together.
# We're not going to square anything, or do anything that will give us different kinds of equations.
# With linear programming we are just dealing with linear equations, in addition to linear constarints.
# Where a constarint is going to look something like if we sum up this particular equation that is just
#some linear combination of all of these variables. It is less than or equal to some bound b.
# And we might have a whole number of these various different constraints that we might place onto our linear
#programming exercise.
# And likewise, just as we can have constraints that are saying this linear equation is less than or equal to
#some bound b, it might also be equal to some bound b.

# It turns out, that if we can take a problem, and formulate it in these terms, formulate the problem, as our goal
#is to minimize the cost function, and we're minimizng that cost function subject to particular constraints.
# Subject to equations that are of the form like this, of some sequence of variables less than bound, or equal
#to some particular value.
# Then, there are a number of algorithms that already exist for solving thses types of problems.


#           Linear Programming

# Mininmize a cost function C1X1 + C2X2 + .... + CnXn
# With constraints of form A1X1 + A2X2 + .... + AnXn < b
#or of form A1X1 + A2X2 + .... + AnXn = b
# With bounds for each variable 1i < Xi < Ui


# Now let's go ahead and take a look at an example.


# Here is a problem that might come up often in the world of linear programming.
# Often this is going to come up when we're trying to optimize for something, and we want to be able to do some
#calculations, and we have constraints on what we're trying to optimize.

# It might be something like this.
# We have 2 machines, X1 And X2. X1 cost 50 dollars to run, X2 cost 80 dollars to run. What we want to do,
#or our objective, is to minimize the total cost.
# But we need to do so subject to certain constraints.
# So there might be a labor constraint. X1 requires 5 units of labor per hour. X2 requires 2 units of labor 
#per hour. And we have a total of 20 units of labor that we have to spend.
# So this is a constraint. We have no more than 20 units of labor that we can spend. And we have to spend it
#across X1 and X2. Each of which require a different amount of labor.
# And we might also have a constraint that tells us that X1 is going to produce 10 units of output per hour.
# X2 is going to produce 12 units of output per hour.
# And the company needs 90 units of output.

# So we have some goal, something we need to acheive. We need to acheive 90 units of output.
# But there are some constraints.
# X1 can only produce 10 units of output per hour. X2 can only produce 12 units of output per hour.

# These types of problems come up quite frequently, and we start to notice patterns in these types of problems.
# Problems where we are trying to optimize for some goal, minimizing cost, maximizing output and maximizing profit,
#or something like that. 
# And there are constraints that are placed on that process.

# And so now we just need to formulate this problem, in terms of linear equations.  



#           Linear Programming Example -

# - Two machines X1 and X2. X1 cost 50/hour to run, X2 cost 80/hour to run. Goal is to minimize cost.

# - X1 requires 5 units of labor per hour. X2 requires 2 units of labor per hour. Total of 20 units of labor
#to spend.

# - X1 produces 10 units of output per hour. X2 produces 12 units of output per hour. Company needs 90 units
#of output.

# Let's start with the first one.
# Here we will come up with an objective function, or cost function rather.
# Cost function: 50X1 + 80X2
# Where X1 is going to be a variable representing how many hours we run machine X1 for.
# X2 is going to be a variable representing how many hours we run machine X2 for.
# And what we are trying to minimize is this cost function. Which is just how much it cost to run each of
#these machines per hour, sumed up.
# This is an example of a linear equation. Just some combination of these variables, plus coeffecients that
#are placed in front of them.
# And we would like to minimize that total value. (50X1 + 80X2)

# Step Two
# But we need to do so subject to these constraints. X1 requires 50 units of labor per hour, X2 requires 2,
#and we have a total of 20 units of labor to spend.
# So that gives us a constraint of this form: 5X1 + 2X2 <= 20. 20 is the total of labor units we have to spend,
#and that's spent across X1 and x2, each of which requires a different number of units of labor per hour,
#for example.

# Step Three 
# And finally we have this contraint here: X1 produces 10 units of output per hour. X2 produces 12 units of output 
#per hour. Company needs 90 units of output. 
# This might look something like this. 10X1 + 12X2 >= 90.

# And if we recall from our formulation before, we said that generally speaking in linear programming,
#we deal with equals constraints, or less than or equals to constraints. 
# But notice we have a greater than or equals to constraint in our step 3. 
# That's not a problem, when we have a greater than or equal to equation, we can just multiple the 
#equation by negative 1, and that will flip it around to a less than or equal to negative 90 for example,
#instead of a greater than or equal to 90.
# That is going to be an equivalent expression we can use to represent this problem.
#(-10X1) + (-12X2) <= -90

# So now that we have our cost function, and the constraints that it is subject to, it turns out that
#there are number of algorithms that can be used in order to solve these types of problems.

# These problems can go a little more into geometry, or linear algebra that we're going to get into,
#but the most popular of these types of algorithms are 

# simplex: One of the first algorithms discovered for trying to solve linear programs.
# Interior-Point: Can be used to solve these types of linear program problems as well.

# The key is not to understand exactly how these algorithms work, but realize that these algorithms exist
#for efficiently finding solutions anytime we have a problem of this particular form.

# Let's take a look at an example of a production directory.

# Here we go over the process of using scipy, which was the library for a lot of science-related functions 
#within Python 

# And we can just go ahead and run this optimization function in order to run a linear program.
# .linprog is going to try and solve this linear program for us.

# Where we provide to this function call, all of the data about our linear program.

# So it needs to be in a particular format, which might be a little confusing at first, but the first
#argument to scipy.optimize.linprog is the cost function. Which in this case is just a list, [50, 80]

# Our original cost function was 50x_1 + 80x_2, so we just tell Python 50 and 80, those are the 
#coefficients that we are trying to optimize for.

# And then we provide all of the constraints.
# Constraint 1: 5x_1 + 2x_2 <= 20
# Constraint 2: -10x_1 + -12x_2 <= -90
# And so, scipy expects these contraints to be in a particular format. It first expects us to provide
#all of the coefficients for the upper bound equations (ub = upper bound). Where the coefficients of 
#the first equation are 5x_1 + 2x_2 [5, 2], and the coefficients for the second equation are -10x_1 + -12x_2
#[-10, -12]
# And then here we provided a separate argument, just to keep things separate, what the actual bound is,
#what is the upper bound for each of these constraints.
# Well for the first constraint we have [20]
# And for constarint number two, we have [-90]
# These are the upper bounds for both of our constraints [20, -90]

# This is a bit of a cryptic way of representing it, it's not as simple as just writing the mathematical
#equations.
# What really is being expected here are all of the coefficients, and all of the numbers that are in
#these equations, by first providing the coefficients for the cost function, then providing all the 
#coefficients for the inequality constraints, then providing all of the upper bounds for those inequality
#constraints.

# And once all of that information is there, then we can run any of these interior-point algorithms
#or the simplex algorithm, even if we don't know how it works, we can just run the function and figure 
#out what the result should be.

# And in our program, we state that if the result is a success, we were able to solve this problem,
#go ahead and print out what the value of X1 and X2 should be, other wise, print "No solution".

# Once we run this program, these should be our optimimal solutions
# X1 should run for 1 and a half hours
# X2 should run for 6 hours and 15 minutes.

# And we were able to do this, just by formulating the problem as a linear equation that we were trying
#to optimize, some cost that we were trying to minimize, and then some constraints that were placed on them.



import scipy.optimize

# Objective Function: 50x_1 + 80x_2
# Constraint 1: 5x_1 + 2x_2 <= 20
# Constraint 2: -10x_1 + -12x_2 <= -90

#result = scipy.optimize.linprog(
#    [50, 80], # Cost Function: 50x_1 + 80x_2
#    A_ub=[[5, 2], [-10, -12]], # Coefficients for inequalities
#    b_ub=[20, -90], # Constraints for inequalities
#)

#if result.success:
#    print(f"X1: {round(result.x[0], 2)} hours") # X1: 1.5 hours
#    print(f"X2: {round(result.x[1], 2)} hours") # X2: 6.25 hours
#else:
#    print("No solution")


# Many many problems fall into this category of problems we can solve if we just figure out how to use
#equations, and use these constraints to represent that general idea.

# This is a theme that is going to come up often. Where we are want to take some problem, and reduce it down
#to some problem we know how to solve, in order to begin to find a solution, and use existing methods 
#we can  use, in order to find a solution more effectively.
    
# It turns out that these types of problems, where we have constraints show up in other ways too.
# And there's an entire class of these problems that generally know as constraint satisfaction problems.
# Now we are going to take a look at how we might formulate a constraint satisfaction problem, and how we might
#go about solving a constraint satisfaction problem.

# But the basic idea of a constraint satisfaction problem is, we have some number of variables that need to take
#on values, and we need to figure out what values each of those variables should take on, but those variables are 
#subject to particular constraints that are going to limit what values those variables can actually take on.

# So let's take a look at a real world example.

# Let's look at exam scheduling.
    
# We have 4 students. Each of them is taking some different number of classes.
# Classes are going to represented by letters.

# Student 1 is enrolled in courses A B C
# Student 2 is enrolled in courses B D E
# Student 3 is enrolled in courses C E F
# Student 4 is enrolled in courses E F G

# And now let's say that the university is trying to schedule exams for all of these courses. 
# But there are only 3 exam slots.
# Monday, Tuesday, Wednesday
# And we have to schedule an exam for each of these courses.

# But the constraint now, the constraint we have to deal with regarding the scheduling, is that we don't 
#want anyone to have to take 2 two exams on the same day. 
# We would like to try and minimize that. Or eliminate it if at all possible.

# So how do we begin to represent this idea. How do we structure this in a way that a computer with an AI
#algorithm can begin to try and solve the problem.

# Well let's in particular just look at these classes we might take and represent each of these courses
#as some node inside of a graph.
# And what we'll do is create an edge between two nodes in his graph, if there is a constraint between those 
#two nodes.

# We can start with student 1, who is enrolled in courses A B C
# What that means is that A and B can't have an exam on the same day.
# A and C can't have an exam at the same day.
# And B and C also can't have an exam on the same day.

# And we can represent that in this graph, by just drawing edges.
# One edge between A and B. One edge bewteen B and C. And one edge between C and A.
# That encodes the idea that between those nodes there is a constarint.
# And in particular, the constraint happens to be that these two can't be equal to each other.
# But there are other types of constraints that are possible depending on the type of problem wee are 
#trying to solve.

# And then we can do the same for each of the other students.

# So for student 2, who is enrolled in courses B D E, well then that means that B D E, those all need to 
#have edges that connect them as well.

# Student 3 is enrolled in courses C E F, so we'll go ahead and connect C E F by drawing edges between them 
#too.

# And finally student 4 is enrolled in courses E F G, and we can represent that by drawing edges between those
#nodes. Since E and F already have an edge between them, we don't need another one.

# This then is what we might call our constraints graph. Just a graphical representation of all of our 
#variables, and the constraints between those possible variables.
# Where in this particular case, each of the constraints represents an inequaily constraint that an edge, 
#between B and D means whatever value the variable B takes on cannot be the value that the variable D
#takes on as well. 

    
#       Constraint Satisfaction

#   Student:            Taking classes:     |           Exam Slots:         |       Contraints Graph:
#   S1                      A B C           |             Monday            |               .A
#   S2                      B D E           |            Tuesday            |              /   \
#   S3                      C E F           |           Wednesday           |           .B -----.C
#   S4                      E F G           |                               |           / |    / \
#                                           |                               |         .D  |   /   .F
#                                           |                               |           \ |  /   /  \
#                                           |                               |            .E /___/    \
#                                           |                               |              \_________.G


# So what then actually is a constraint satisfaction problem?
# Well, a constraint satisfaction problem is just some set of variables, X1 all the way through Xn, and some
#set of domains for each of those variables.
# So every variable needs to take on some values. Maybe every variable has the same domain.
# But maybe every variable has a slightly different domain.
# And then there's a set of constraints, we'll just call a set C. That is some constraints that are placed
#upon these variables, like X1 is not equal to X2.

# But there could be other forms too, like maybe X1 equals X2 plus 1, if these variables are taking on 
#numerical values in their domain, for example.

# The types of constraints are going to vary based on the types of problems.

# And constraint satisfaction shows up all over the place as well, in any situation where we have variables
#that are subject to particular constraints.


#       Constraint Satisfaction Problem -
#   - Set of variables {X1, X2,...., Xn}
#   - Set of domains for each variable {D1, D2,...., Dn}
#   - Set of constraints C


# So one popular game is Sudoku, for example, this 9 by 9 grid where we need to fill in numbers in each
#of these cells, but we want to make sure that there is never a duplicate number in any row, or in any
#column, or in any grid of 3 by 3 cells, for example.

#     Sudoku Game Grid
#
# 5 | 3 |   ||   | 7 |   ||   |   |   |
#--------------------------------------
# 6 |   |   || 1 | 9 | 5 ||   |   |   |
#--------------------------------------
#   | 9 | 8 ||   |   |   ||   | 6 |   |
#______________________________________
# 8 |   |   ||   | 6 |   ||   |   | 3 |
#--------------------------------------
# 4 |   |   || 8 |   | 3 ||   |   | 1 |
#--------------------------------------
# 7 |   |   ||   | 2 |   ||   |   | 6 |
#______________________________________
#   | 6 |   ||   |   |   || 2 | 8 |   |
#--------------------------------------
#   |   |   || 4 | 1 | 9 ||   |   | 5 |
#--------------------------------------
#   |   |   ||   | 8 |   ||   | 7 | 9 |
#______________________________________


# So what might this look like as a constraint satisfaction problem?

# Well, my variables are all of the empty squares in the puzzle.
# Variables {(0,2), (1,1), (1,2), (2,0).....} These variables correspond to empty square location on 
#our grid.

# So represented here is just like an X,Y coordinate, for example, as all of the squares where I need 
#to plug in a value, where I don't know what value it should take on.

# The domain is just going to be all of the numbers from 1 through 9, any number that we can fill in to
#one of those cells.
# Domains {1, 2, 3, 4, 5, 6, 7, 8, 9} for each variable.
# So that is going to be the domain for each of these variables.

# And then the costraints are going to be of the form, like this cell can't be equal to this cell, can't 
#be equal to this cell, can't be equal to this cell... all of these need to be different, for example.
# And same for all of the rows, and the columns, and the 3 by 3 squares as well.

# So those constraints are going to enforce what values are actually allowed.

# And we can formulate the same idea in the case of this exam scheduling problem, where the variables 
#we have are different courses, A up through G.
# Variables {A, b, C, D, E, F, G}

# The domain for each off these variables is going to be Monday, Tuesday, Wednesday.
# Domains {Monday, Tuesday, Wednesday} for each variable.
# Those are the possible values each of the variables can take on, that in thiss case just represent
#when is the exam for that class.

# And then the constraints are of this form, A is not equal to B, A is not equal to C, meaning A and B
#can't have an exam on the same day, A and C can't have an exam on the same day.
# Contraints {A != B, A != C, B != C, B != D, B != E, C != E, C != F, D != E, E != F, E != G, F != G}

# Or more formally, these two variables cannot take on the same value within their domain.

# So that then is this formulation of a constraint satisfaction problem, that we can begin to use
#to try and solve this problem.

# And constraints can come in a number of different forms.

# There are hard constraints.
# Hard Constraints-
#                 - Constraints that must be satisfied in a correct solution.
# So something like the sudoku puzzle, you cannot have this cell and this cell that are in the same 
#row, take on the same value.

# But problems can also have soft constraints.
# Soft Constraints -
#                  - Constraints that express some notion of which solutions are preferred over others.

# These are contraints that express some notion of preference, that maybe A and B can't have an exam
#on the same day, but maybe someone has a preference that A's exam is earlier than B's exam.
# It doesn't need to be the case with some expression, that some solution is better than another solution.

# And in that case, we might formulate the problem as trying to optimize for maximizing people's preferences.
# We want people's preferences to be satisfied as much as possible.

# In this case though, we'll mostly deal with Hard Constraints, contraints that must be met in orrder to have
#the correct solution to a probblem.

# So we want to figure out some assignment to these variables to their particular values that is ultimately
#going to give us a solution to the problem, by allowing us to assign some day to each of the classes,
#such that we don't have any conflicts between any of the classes.


#              Contraints Graph:
#
#                     .A
#                    /   \
#                 .B -----.C
#                 / |    / \
#               .D  |   /   .F
#                 \ |  /   /  \
#                  .E /___/    \
#                    \_________.G


# So it turns out that we can classify the constraints in a constraints satisfaction problem into a number
#of different categories. 

# The first of those categories is perhaps the simplest of the types of contraints, which are known as 
#unary constraints, where unary contraints are just constraints that involve a single variable.

#   Unary Constraints -
#                     - Constraints involving only one variable

# For example, unary constraints might be something like, A does not equal Monday, meaning Course A cannot
#have its exam on Monday.

#   Unary Constarints

# {A != Monday}


# This is in contrast to something like Binary Constraints, which is a constraint that involves two
#variables, for example.

#   Binary Constraint -
#                     - Constraint involving two variables.

# So this would be a constraint like the ones we were looking at before. Something like A does not equal B
#is an example of a Binary Constraint, because it is a constraint that has two variables involved in it, A and B.
# And we represented that using the arc or some edge that connects variable A to variable B.

#   Binary Constraint 
# {A != B}


# And using this knowledge of, Ok, what is a unary constraint? What is a Binary Constraint?
# There are different types of things we can say about a particular constraint satisfaction problem.

# And one thing we can say is, we can try and make the problem node consistent.
# So what does node consistency mean?
# Node Consistency means we have all of the values in variables domain satisfying that variables unary
#constraints.
# So for each of the variables inside our constraints satisfaction problem, if all of the values satisfy
#the unary constraints for that particular variable, we can say that the entire problem is Node Consistent,
# or we can even say that a particular variable is Node Consistent if we want to make one Node Consistent
#within itself.

#   Node Consistency -
#                    - When all of the values in a variable's domain satisfy the variable's unary constraint.

# So what does that actual look like?

# Let's look at now a simplified example, where instead of having a bunch of different classes, we just 
#have two classes, A and B, each of which has an exam of either Monday, Tuesday, or Wednesday.

# And now let's imagine we have these constraints, A not equal to Monday, B not equal to Tuesday, B not
#equal to Monday, A not equal to B.
# So those are the contraints that we have on this particular problem.

# And what we can now try to do is enforce Node Consistency.
# And node consistency just means we make sure that all of the values for any variables domain satisfy
#its unary constraints.

# We can start by trying to make node A, node consistent. 
# Does every value inside of node A's domain satisfy its unary constraints?

# Well, initially we'll see that Monday does not satisfy A's unary constraints, because we have 
#a unary constraint that specifies that A does not equal Monday.
# But Monday is still in A's domain.
# So this is something that is not node consistent, because we have Monday in the domain, but this is not 
#a valid value for this particular node.


#       Example

#   A ---------------- B
# {Mon,Tues,Wed}    {Mon,Tues,Wed} = Domain Variables 
#{A != Mon, B != Tues, B != Mon, A != B} = Constraints

# And so how do we make this node consistent?

# To make the node, node consistent, what we'll do is we'll just go ahead and remove Monday from A's
#domain.

# Now A can only be on Tuesday or Wednesday, because we had the constraint that specified that 
#A does not equal Monday.

#   A ---------------- B
# {Tues,Wed}    {Mon,Tues,Wed} = Domain Variables 
#{A != Mon, B != Tues, B != Mon, A != B} = Constraints

# And at this point now, A is node consistent.

# For each of the values that A can take on, Tuesday and Wednesday, there is no unary constraint that
#conflicts with that idea.

# And now we can turn our attention to B. 

# B also has a domain of Monday, Tuesday, and Wednesday. 
# And we can begin to see whether those variables satisfy the unary constraints as well.

# We can see that B has a unary constraint that specifies that B does not equal Tuesday.
# And that does not appear to be satified by the domain of Monday, Tuesday, Wednesday, because 
#Tuesday, this possible value that B could take on is not consistent with this unary constraint,
#that B is not equal to Tuesday.

# So to solve that problem we'll go ahead and remove Tuesday from B's domain.
# Now B's domain only has Monday and Wednesday.

# But as it turns out, there is another unary constraint that we placed on the variable B, which
#is B does not equal Monday.

# And that means that the value Monday inside of B's domain, is not consistent with B's unary constraints,
#because we have a constraint that says that B cannot equal to Monday.
# And so we can remove Monday from B's domain.

# And now we've made it through all of the unary constraints.

# We've not yet considered the {A != B} constraint, which is a Binary Constraint.

# But we've considered all of the unary constraints, all the constraints that involve just a single variable.
# And we've made sure that every node is consistent with those unary constraints.

# We can now that we have enforced Node Consistency, that for each of these possible nodes, we can
#pick any of these values in the domain, and there won't be a unary constraint that is violated as a result
#of it.

# This is our new Node Consistent Output

#   A                       B
# {Tues, Wed}              {Wed} = Domain Variables
#{A != Mon, B != Tues, B != Mon, A != B} = Constraints

# So Node Consistency is easy fairly to enforce.
# We just take each node, make sure the values in the domain satisfy the unary constraints.

# Where things get a little more interesting, is when we consider different types of consistency.
# Something like arc consistency for example.

#   Arc Consistency -
#                   - When all the values in variables domain satisfy the variable's Binary Constraints.

# So when we're looking at trying to make A arc consistent, we're no longer just considering the unary
#constraints that involve A.

# We're trying to consider all of the Binary Constraints that involve A as well.

# So any edge that connects A to another variable inside of that constraints graph we were looking
#at before.

# Put a little more formally, Arc Consistency (And Arc is just another word for Edge that connects two of
#these nodes inside of our constraints graph) can be defined a little more precisely like this.
# In order to make some variable X arc consistent with respect to some other variable Y, we need to
#remove any elements from X's domain to make sure that every choice for X, every choice in X's domain,
#has a possible choice for Y.

# So put another way, if we have a variable X and we want to make X arc consistent, then we are going
#to look at all of the possible values that X can take on and make sure that for all of those possible
#values, there is still some choice that we can make for Y.
# If there is some arc between X and Y, to make sure that Y has a possible option that we can choose
#as well.

# So let's look at an example of that going back to this example from before.

# We enforced Node Consistency already by saying that A can only be on Tuesday or Wednesday, because we
#knew that A cannot be on Monday.

# And we said that B's domain only consists of Wednesday, because we knew that B does not equal to Monday
#or Tuesday.

# #   A                       B
# {Tues, Wed}              {Wed} = Domain Variables
#{A != Mon, B != Tues, B != Mon, A != B} = Constraints

# So now let's begin to consider the Arc Consistency.

# Let's try and make A arc consistent with B.

# And what that means is to make A arc consistent with respect to B, means that for any choice we make
#in A's domain, there is some choice we can make in B's domain that is going to be consistent.

# For A, we can choose Tuesday as a possible value for A.
# If we choose Tuesday for A, is there a value for B that satisfies the Binary Constraint?

# Yes. B's Wednesday would satisfy this constraint that A does not equal B because Tuesday does not
#equal Wednesday.

# However, if we choose Wednesday for A, well then there is no choice in B's domain that satisfies
#this Binary Constraint.

# There is no way we can choose something for B that satifies A does not equal B, because we know that
#B must equal Wednesday.

# And so if we ever run into a situation like this where we see that here is a possible value for A
#such that there is no choice for B that satisfies the Binary Constraint, well then this not 
#Arc Consistent.

# And to make it arc consistent, we would need to take Wednesday and remove it from A's domain.
# Because Wednesday was not going to be a possible choice we can make for A, because it wasn't
#consistent with this Binary Constraint for B.

# So here now, we've been able to enforce arc consistency.

#   A                       B
# {Tues}              {Wed} = Domain Variables
#{A != Mon, B != Tues, B != Mon, A != B} = Constraints 