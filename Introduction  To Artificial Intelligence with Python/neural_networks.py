# Neural Networks - Here we will take a look at how computers are able to draw inspiration from human intelligence,
#looking at the structure of the human brain, and how neural networks can be computed analog to that sort of idea,
#and how, by taking advantage of a certain type of structure of computer program, we can write neural networks that
#are able to perform tasks very, very effectively.

# If we think about the structure of a biological neural network, there are a couple of key properties scientist
#observed.

#   Neural Networks -
# - Neurons are connected to and recieve electrical signals from other neurons.
# - Neurons process input signals and can be activated.

# And so the question then became, could we take this biological idea of how it is that Humans learn with brain 
#and with neurons, and apply that to a machine as well.

# In effective, designing an artificial neural network, or an A.N.N

#   Artificial Neural Network -
# - Mathematical model for learning inspired by biological neural networks.

# And what artificial neural networks will allow us to do is -
# - Model mathematical functions from inputs to outputs based on the structure and parameters of the network.
# - Allows for learning the networks parameters based on data.

# So in order to create our A.N.N, instead of using biological neurons, we're gonna use what we call units.

# Units inside of a neural network which we can represent kinda like a node in a graph, which we'll represnt 
#here with a cirlce.

# O

# And these artificial units, these artificial neurons, can be connected to one another.

# Here for instance we have two units connected by this edge inside of this graph.

#   O -------> O

# And so what we are going to do now is think of this idea as something like mapping, from inputs to outputs.

# So we have one unit that is connected to another unit, we'll think of the left side as the input, and
#the right side as the output.

# And what we're trying to do is, figure out how to solve a problem, how to model some sort of mathematical function.

# And this might take the form of something like we saw last time, which had certain inputs,
#like variables (X1, X2), and given those inputs, we want to perform some sort of task.

# A task like predicting whether or not it's going to rain (X1, X2) ---------> Rain.

# Ideally we'd like some way, given these inputs, (X1, X2), which stand for some sort of variables to do with the
#weather, we would like to be able to predict in this case a boolean classification.

# Is it going to rain, or is it not going to rain?

# And we did this last time by way of a mathematical function.

# We define some function, h, for our hypothesis function, that took as input X1 and X2, the two inputs we cared about
#processing to determine whether we thought it was going to rain, or not going to rain.

#   h(X1, X2)

# The question then becomes, what does this hypothesis function do in order to make that determination.

# And we decided last time to use a linear combination of these input variables to determine what the output should
#be.

# So our hypothesis function was equal to something like this.

#   h(X1, X2) = W0 + W1X1 + W2X2

# So what's going on here is that X1 and X2, those are our input variables. 

# They are inputs to our hypoyhesis function.

# And each of those input variables is being multiplied by some weight, which is just some number.

# So X1 is being multiplied by Weight 1 (W1), X2 is being multiplied by Weight 2 (W2), and we have this additional
#weight, Weight 0 (W0), that doesn't get multiplied by an input variable at all, it just serves to either move the
#function value up, or move the function value down.

# We can think of it as either a weight that's just multiplied by some dummy value, like the number 1, or sometimes
#we'll see in the literature, people call this variable weight 0 a bias.

# So we can think of these variables as slightly different. 

# We have weights that are multiplied by the input, and we separately add some bias to the result as well.

# We'll hear both of those terminologies used when we talk about neural networks and machine learning.

# So in effect, what we've done here, is that in order to define the hypothesis function, we just need to decide 
#and figure out what the weights should be to determine what values to multiply by our inputs to get some sort of 
#result.

# Of course at the end of this what we need to do is make some sort of classification, like raining or not raining.

# And to do that we use some sort of function that defines some sort of threshold.

# For example, something like the step function.


#    |     STEP FUNCTION               _____________________________________________
#  1 |                                 |
#    |  g(x) = 1 if x >= 0, else 0     |
#  O |                                 |
#  u |                                 |
#  t |                                 |
#  p |                                 |
#  u |                                 |
#  t |                                 |
#    |                                 |
#    |                                 |
#  0 |_________________________________|_______________________________________________
#                                    W . X


# The step function is defined as (1 if the weight is greater than or equal to 0, else 0)

# We can think of the line down the middle as a dotted line.

# It stays at 0 all the way up until one point, and then the function steps, or jumps up to 1.

# So it's 0 before it reaches the threshold, and then it's 1 after it reaches a particular threshold.

# And so this was one way we could define what we'll come to call an activation function.

# A function that determines when it is that this output becomes active.

# Changes to a 1 instead of being a 0.

# But we also saw that if we didn't just want a purely binary classification, we didn't want purely 1 or 0, and we 
#wanted to allow for some in-between real number values, we could use a different function.

# And there are a number of choices, but the one that we looked at was the logistic sigmoid function


#    |     Logistic Sigmoid                           _______________________________
#  1 |           ex                                 _|
#    |  g(x) = --------                           _/
#  O |           ex + 1                         _/
#  u |                                        _/
#  t |                                      _/
#  p |                                     |
#  u |                                    /
#  t |                                   |
#    |                                  /
#    |                                 /
#  0 |_________________________________|_______________________________________________
#                                    W . X


# The logigistic sigmoid function has sort of an s shaped curve to it.

# We can represent this as a probability. 

# Maybe somewhere in between, the probability of rain is something like 0.5, maybe a little bit later the probability
#of rain is 0.8, and so rather than just have a binary classification of 0 or 1, we can allow for numbers that 
#are in between as well.

# And it turns out that there are many other different types of activation functions.

# Where an activation function just takes the output of multiplying the weights together and adding that bias, and
#then figuring out what the actual output should be.

# Another popular one is the rectified linear unit, other wise know as relu.


#    |     Rectified Linear Unit (Relu)           /
#  1 |                                           /
#    |  g(x) = max(0, x)                        /
#  O |                                         /
#  u |                                        /
#  t |                                       /
#  p |                                      /
#  u |                                     /
#  t |                                    /
#    |                                   /
#    |                                  /
#  0 |_________________________________/_______________________________________________
#                                    W . X


# And the way this works is it just takes an input, and takes the maximum of that input, and 0.

# So if it's positive it remains unchanged, but if it's negative, it goes ahead and levels out at 0.

# And there are other activation functions that we could choose as well.

# But in short, each of these activation functions we could just think of as a function that gets applied to the 
#result of all of this computation (h(X1, X2) = W0 + W1X1 + W2X2).

# We take some function g, and apply it to the result of all that calculation h(X1, X2) = g(W0 + W1X1 + W2X2).

# And this then is what we saw last time.

# The way of defining some hypothesis function that takes in inputs, calculates some linear combination of those
#inputs, and then passes it through some sort of activation function to get our output.

# And this actually turns out to be the model for the simplest neural networks.

# h(X1, X2) = g(W0 + W1X1 + W2X2)

# We're going to instead represent this mathematical idea graphically, by using a structure like this.


# O 
#  \
#   O
#  /
# O 


# Here is a neural network that has two inputs that we can think of as X1 and X2.

# And one output that we can think of as classifying whether we think it's going to rain or not going to rain, for
#example in this particular instance.

# So how exactly does this model work?

# Each of the two inputs represents one of our input variables, x1, and X2.

# And notice that the inputs are connected to the output, via the edges, which are going to be defined by their
#weights.


       
#       W0
# O X1  |
#  \<---|---Edge / Weight 1
#    -- O<-------------Output   g(W0 + W1X1 + W2X2)
#  /<------Edge / Weight 2
# O X2


# So these edges each have a weight associated with them, Weight 1, and Weight 2.

# And then the output unit, what it's going to do is, it's going to calculate an output based on those inputs and 
#based on those weights.

# This output unit is going to multiply all of the inputs by their weights, then add in the bias term, which we 
#can think of as an extra W0 term that gets added into it, and then we pass it through an activation function.

# So that is just graphical way of representing the same idea we saw last, just mathematically.

# And we're goonna call this a very simple, neural network.

# And we'd like for this neural network to be able to learn how to calculate some function.

# We want some function for the neural network to learn, and the neural network is going to learn, what should the 
#values of W0, W1, and W2 be.

# What should the activation function be in order to get the result that we would expect.

# We can actually take an look at an example of this.

# What then is a very simple function that we might calculate?

# Well if we recall back from when we were looking at propositional logic, one of the simplest functions we 
#looked at was something like the Or Function.


#   Or
# _____________________________
#|___X____|___Y_____|__f(x,y)__|
#|___0____|___0_____|_____0____|
#|___0____|___1_____|_____1____|
#|___1____|___0_____|_____1____|
#|___1____|___1_____|_____1____|


# This function takes two inputs, X and Y, and outputs 1, otherwise known as true, if either one of the inputs or both
#of them are 1.

#And outputs a 0, otherwise known as false, if both the inputs are 0.

# That is the Or function, and this the truth table for the Or function.

# As long as either of the inputs is 1, the output is 1.

# And the only time the output is 0 is if both the inputs is 0.

# So the question is how can we take this, and train a neural network to be able to learn this particular function?

# What would those weights look like?

# We could do something like this.

# Here's our neural network.

# And we'll propose that in order to calculate the Or function, we're going to use a value of 1 for each of the 
#weights, and we'll use a bias of -1.

# And then we'll just use a step function as our activation function.


#      -1
# 0 X1  |
#  \<---|--- 1
#    -- O<-------------Output   g(-1 + 1X1 + 1X2)
#  /<------- 1
# 0 X2


#    |     STEP FUNCTION               _____________________________________________
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |_________________________________|_______________________________________________
#           -1                         0                    1


# How then does this work?

# Well if we wanted to calculate something like 0 or 0, which we know to be 0, beacuse false and false is false.

# Then what are we going to do?

# Well our output unit is going to calculate X1 input, multiplied by the weight, 0 times 1, which is 0.

# And the same thing for the X2 input, which would end up being 0.

# And we'll add to that, the bias, or -1, which will give us a result of -1.

# If we plot that on our activation function, it's before the threshold, which means either 0 or 1.

# There's only 1 after the threshold since -1 is before the threshold, the output that this unit provides is going
#to be 0.

# And that's what we would expect it to be.

# That 0 or 0. should be 0.

# What if instead we had 1 or 0?


#      -1
# 1 X1  |
#  \<---|--- 1
#    -- O<-------------Output   g(-1 + 1X1 + 1X2)
#  /<------- 1
# 0 X2


#    |     STEP FUNCTION               _____________________________________________
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |_________________________________|_______________________________________________
#           -1                         0                    1


# In this case, in order to calculate what the output is going to be, we again have to do this weighted sum.

# 1 times 1, that's 1, and 0 times 1 is 0.

# The sum of that so far is 1.

# Add -1 to that and the output becomes 0.

# And if we plot 0 on our step function, 0 is just at the threshold, and so the output is going to be 1.

# Because the output of 1 or 0 is 1, so that's what we would expect as well.

# And just for one more example, let's say we had 1 or 1.

# What would the result be?


#      -1
# 1 X1  |
#  \<---|--- 1
#    -- O<-------------Output   g(-1 + 1X1 + 1X2)
#  /<------- 1
# 1 X2


#    |     STEP FUNCTION               _____________________________________________
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |_________________________________|_______________________________________________
#           -1                         0                    1


# Well 1 times 1 is 1, and 1 times 1 is 1.

# The sum of those is 2.

# Then we add the bias term to that, which is -1.

# We will end up with the number 1.

# 1 plotted on the step function is well beyond the threshold, so our output is going to be 1 as well.

# The output is always 0 or 1, depending on whether or not we're passed the threshold.

# And this neural network models the Or Function.

# And we can imagine trying to do this for other functions as well.

# A function like the "And" Function for example.


#   And
# _____________________________
#|___X____|___Y_____|__f(x,y)__|
#|___0____|___0_____|_____0____|
#|___0____|___1_____|_____0____|
#|___1____|___0_____|_____0____|
#|___1____|___1_____|_____1____|


# It takes two inputs and calculates whether both X and Y are true.

# So X is 1 and Y is 1, then the output of X and Y is 1, or true.

# But if in all the other cases, the output is 0, or false.

# How could we model that inside of a neural network as well?

# Well it turns out we can do it in the same way, except instead of -1 as the bias, we can use -2 as the bias instead.

# What does that end up looking like?

# Well if we had 1 and 1, that should be 1, because true and true is equal to true.

# We take 1 times 1, that's 1, and 1 times 1, that's also 1.

# That gives us a total sum of 2 so far.

# Now we add the bias of -2, which will give us the value 0.

# And 0, when we plot it on the activation function, is just past the threshold, and so the output is going to be 1.

#      -2
# 1 X1  |
#  \<---|--- 1
#    -- O<-------------Output   g(-2 + 1X1 + 1X2)
#  /<------- 1
# 1 X2

#    |     STEP FUNCTION               _____________________________________________
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |_________________________________|_______________________________________________
#           -1                         0                    1


# But what if we add any other input for example, like 1 and 0.

# We'll have 1 plus 0, which is 1, plus -2 is going to give us -1, and -1 is not past the threshold, so the output
#is 0.

#      -2
# 1 X1  |
#  \<---|--- 1
#    -- O<-------------Output   g(-2 + 1X1 + 1X2)
#  /<------- 0
# 0 X2

#    |     STEP FUNCTION               _____________________________________________
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |                                 |
#    |_________________________________|_______________________________________________
#           -1                         0                    1


# So those are some very simple functions that we can model using the neural network that has two inputs and 1
#output, where our goal is to be able to figure out what those weights shoild be, in order to determine what the
#output shpuld be.

# And we could imagine generalizing this to calculate more complex functions as well.

# That maybe given the humidity and the pressure, we want to calculate what's the probability that it's going to
#rain, for example.

# Or we might want to do a regression stlye problem, where given some amount of advertising, and given what month
#it is maybe, we want to predict what our expected sales are going to be for that particular month.

# We could imagine these inputs and outputs being different as well. 

# And it turns out that in some problems, we're not just gonna have two inputs.

# And the nice thing about these neural networks is that we can compose multiple units together.

# Make our networks more complex, just by adding more untis into that particular neural network.

# So the network we've been looking at has two inputs and one output. 

# But we could just as easily say let's go ahead and have 3 inputs.

# Or have even more inputs.

# But we ultimately decide how many inputs we have, all leading to and calculating some sort of output we care 
#about figuring out the value of.

# How then does the math work for figuring out that output?

# Well it's going to work in a very similar way. 

# In the case of the two inputs, we had two weights, indicated by their edges, and we multiplied the weights by 
#the numbers and adding the bias term.

# And we'll do the samething in the other cases as well.

# Let's imagine we have three inputs.

# We'll multiple each of the inputs by each of the weights.

# And we'd do the samething if we had five inputs, for example.

# And there could be more, how ever many nodes that we want inside of our neural network, where each time we're 
#just gonna sum up all of those input variables, multiplied by their weight, and then add the bias term at the
#very end.

# And so this allows us to represent problems that have even more inputs, just by growing the size of our 
#neural network.

# Now the next question we might ask, is a question about how it is that we train these neural networks.

# In the case of the Or function and the And function, they were simple enough functions that we knew, here are 
#what the weights should be, in order to calculate the output that we want. 

# But in general, with functions like predicting sales, or predicting whether or not it's going to rain, these are
#much trickier functions to be able to figure out.

# We would like the computer to have some mechanism of calculating what it is that the weights should be.
 
# How it is to set the weights so that our neural network is able to accurately model the function that we care 
#about trying to estimate.

# And it turns out that the strategy for doing this, inspired by the domain of calculus, is a technique called 
#gradient descent.

#   Gradient Descent -
# - Algorithm for minimizing loss when training neural networks.

# And recall that loss refers to how bad our hypothesis function happens to be.

# That we can define certain loss functions, that just give us a number for any particular hypothesis, saying
#how poorly does it model the data.

# How many examples does it get wrong, how are they worse, or less bad as compared to other hypothesis functions
#that we might define.

# And this loss function is just a mathematical function, and when we have a mathematical function, in calculus what
#we can do is calculate something that known as the gradient.

# We can think of the gradient as a slope, the direction the loss function is moving at any particular point.

# And what it's going to tell us is, in which direction should we be moving the weights in order to minimize 
#the amount of loss.

# And so generally speaking we won't get into the calculus of it, but the high level idea for gradient descent 
#is going to look like this.

# If we want to train a neural network we'll go ahead and start just by choosing the weights randomly.

# Just pick random weights for all of the weights in the neural network.

# And then we'll use the input data that we have access to in order to train the network, in order to figure out what 
#the weights should be.

# So we'll repeat this process, again and again. 

# The first step is we're going to calculate the gradient, based on all of the data points.

# So we'll look at all the data, and figure out what the gradient is at the place where we currently are, the
#current setting of the weights.

# Which means in which direction should we move the weights in order to minimize the total amount of loss, in order
#to make our solution better.

# And once we've calculated that gradient, which direction we should move in a loss function, we'll then, we can just
#update those weights, according to the gradient.

# Take a small step in the direction of those weights in order to try and make our solution a little bit better.

# And the size of the step that we take, that's going to vary, and we can choose that when we're training a 
#particular neural network.

# But in short, the idea is going to be take all the data points, figure out based on those data points in what 
#direction the weights should move, and then move the weights one small step in that direction.

# And if we repeat that process over and over again, adjusting the weights a little at a time based on all
#the data points, eventually, we should end up with a pretty good solution to trying to solve this problem.


#   Gradient Descent 

# - Start with a random choice of weights
# - Repeat:
#   - Calculate the gradient based on all the data points:
#     Direction that will lead to decreasing loss
#   - Update weights according to the gradient


# Now if we look at this algorithm, a good question to ask anytime we are analysing an algorithm, is what is 
#going to be the expensive part of doing this calculation?

# What is going to take a lot of work to try and figure out, what is going to be exspensive to calculate.

# And in particular, in the case of the gradient descent, the really exspensive part is the (all data points),
#(see line 629).

# Having to take all the data points, and using all those data points, figure out what the gradient is at this
#particular setting of all of the weights.

# Because odds are, that in a big machine learning problem, where we are trying to solve a big problem with a lot
#of data, we have a lot of data points in order to calculate, and figuring out the gradient based on all of those data
#points is going to be exspensive, and we'll have to do it many times. 

# So we'll likely repeat this process again and again and again, going through all of the data points, taking one
#small step, over and over as we try and figure out what the optimal setting of those weights happens to be.

# It turns out that we would ideally like to be able to train our neural networks faster.

# To be able to more quickly be able to converge to some sort of solution that is going to be a good solution to
#the problem.

# So, in that case there are alternatives to just the standard gradient descent.

# We can employ a method like stochastic gradient descent, which will randomly just choose one data point at a time
#to calculate the gradient based on, instead of calculating it based on all of the data points 


#   Stochastic Gradient Descent 

# - Start with a random choice of weights
# - Repeat:
#   - Calculate the gradient based on one data point:
#     Direction that will lead to decreasing loss
#   - Update weights according to the gradient


# So the idea here, is that we have some setting of the weights, we pick a data point, and based on that one data point
#we figure out in which direction should we move all of the weights, and move the weights in that small direction.

# And then take another data point and do that again.

# And repeat this process again and again.

# Maybe looking at each of the data points multiple times, but each time, only using one data point to calculate
#the gradient.

# Calculate which direction we should move in.

# Now, just using one data point instead of all of the data points probably gives us a less of a accurate estimate
#of what the gradient actually is, but on the plus side, it's going to be much faster to be able to calculate.

# We can much more quickly calculate what the gradient is based on one data point, instead of calculating based
#on all of the data points and having to do all of that computational work again and agian.

# So there are trade offs, between looking at all of the data points, and just looking at one data point.

# And it turns out that a middle ground that is also quite popular, is a technique called mini-batch gradient
#descent.


#   Mini-Batch Gradient Descent 

# - Start with a random choice of weights
# - Repeat:
#   - Calculate the gradient based on one small batch:
#     Direction that will lead to decreasing loss
#   - Update weights according to the gradient


# Where the idea here is instead of lookng at all of the data, versus just a single point, we instead divide or
#data set up into small batches.

# Groups of data points where we can decide how big a particular batch is.

# But in short we're just going to look at a small number of points at any given time. hopefully getting a more
#accurate estimate of the gradient, but also not requiring all of the computational efffort needed to look at
#every single one of the data points.

# So gradient descent then, is this technique that we can use in order to train these neural networks, in order
#to figure out what the setting of all these weights should be, if we want some way to try and get an accurate
#notion of how it is that this function should work.

# Some way, modeling how to transform the inputs into particular outputs.

# Now so far, the networks that we've taken a look at have all been structured similar to this.


# O
#  \
# O -- O
#  /
# O


# We have some number of inputs, maybe 2, or 3, or 5, or more, and then we have one output, that is just predicting
#like rain or no rain, or just predicting one particular value.

# But often in machine learning problems we don't jsut care about one output, we might care about an output that
#has multiple different values associated with it.

# So, in the same way that we can take a neural network and add units to the input layer, we can likewise add
#outputs to the output layer as well.

# Instead of just one output, we can imagine we have two outputs.

# Or maybe even four outputs, for example.

# Where in each case as we add more inputs or add more outputs, if we want to keep the network fully connected
#between the two layers, we just need to add more weights.

# Each of the inputs nodes has weights associated with each of the outputs, meaning that each input node is 
#connected by a weighted edge to each output.

# So as we add nodes, we add more weights in order to make sure each of the inputs can somehow be connected to
#each of the outputs, so that each output value can be calculated based on what the value of the input happens to
#be.

# So what might a case be where we want multiple different outputs values?

# Well we might consider that in the case of weather predicting for example, we might not just care whether it's
#raining or not raining, there might be multiple different categories of weather that we would like to categorize
#the weather into.

# With only a single outptut variable, we can do a binary classification, like rain or no rain, for example, 1 or
#0.

# But it doesn't allow us to do much more that that.

# With multiple output variables, we might be able to use each one to predict something different.

# Maybe we want to categorize the weather into one of four different categories.

# Something like rainy, or sunny, or cloudy, or snowing.

# And we now have four output variables that can be used to represent maybe the probabilty that it is rainy,
#as opposed to sunny, as opposed to cloudy, as opposed to snowing.

# How then would this neural network work?

# Well we have some input variables that represent some data that we have collected about the weather.

# Each of the inputs gets multiplied by each of the various different weights.

# We have more multiplications to do, but these are fairly quick mathematical operations to perform.

# And then what we get after passing them through some sort of activation function in the outputs, we end up getting
#some sort number.

# Where that number we might imagine we could interpret as like a probability.

# Like a probability that it is one category as opposed to another category.

# So imagine that we are implying that based on the inputs, we think there is a 10 percent chance of it raining,
#a 60 percent chance it's sunny, a 20 percent chance it's cloudy, and a 10 percent chance that it's snowy.

# And given that output, if these represent a probability distribution, well then we could just pick whichever
#one has the highest value, in this case sunny, and say well most likely we think this categorization, the inputs
#mean that the output should be sunny, and that is what we would expect the weather to be in this particular case.

# And so this allows us to do these sort of multiclass classification.

# Where instead of just having a binary classification, 1 or 0, we can have as many different categories as we 
#want, and we can have our neural network output these probabilities over which categories are more likely than
#other categories, and using that data, we're able to draw some sort of inference on what it is that we should do.

# So this was sort of the idea of supervised machine learning.

# We can give our neural network a whole bunch of data, a whole bunch of input data corresponding to some labels,
#some output data, like, we know that it was raining on this day, we know that it was sunny on that day, and using
#all of that data, the algortithm can use gradient descent to figure out what all of the weights should be in 
#order to create some sort of model that hopefully allows us a way to predict what we think the weather is going
#to be.

# But neural networks have a lot of other applications as well.

# We could imagine applying this same sort of idea to a reinforcement learning sort of example as well.

# Where we remember that in reinforcement learning, what we wanted to do, is train some sort of agent to learn
#what action to take depending on what state that it currently happen to be in.

# So depending on the current state of the world, we wanted the agent to pick from one of the available actions
#that is available to them.

# And we might model that but having each of the input variables represent some information about this state.

# Some data, about what state our agent is currently in, and the output, for example, could be each of the 
#various different actions that our agent can take.

# For example, Actions 1, 2, 3, and 4.

# And we might imagine that this network would work in the same way.

# That based on the particular inputs, we go ahead and calculate values for for each of the outputs, and those 
#outputs could model which action is better than other actions, and we could just choose based on looking at those
#outputs, which action should we take.

# And so these neural networks are very broadly applicable.

# All they are really doing is modeling some mathematical function.

# So anything that we can frame as a mathematical function, something like classifying inputs into various different 
#categories, or figuring out based on some input state what action we should take, these are all mathematical functions
#that we could attempt to model by taking advantage of this neural network structure, and in particular, taking 
#advantage of this technique, gradient descent, that we can use in order to figure out what the weights should be
#in order to do this sort of calculation.

# Now how is it that we would go about training a neural network so it could have multiple outputs instead of just 1?

# Well with just a single output we could say what the value of that output should be, and then we update all of the
#weights that corresponded to it.

# And when we have multiple outputs, we can really think of them as four separate neural networks.

# So really we have output 1, connected to each of the three input neurons, by three weights, for example, as its 
#own neural network, and so and so forth for the other 3 outputs.

# Each output neuron would be connected to the same inputs, but all be connected by its own set of weights.

# And so if we wanted to train a neural network that has four outputs instead of just one, in this case where the 
#inputs are directed connected to the outputs, we could really think of this as training four independent neural
#networks.

# We know what the outputs for each of the four should be, based on our input data, and using that data, we can begin 
#to figure out what all of the individual weights should be.

# And maybe there's an additional step at the end, to make sure that we turn all of the output values into a 
#probability distribution, such that we can interpret which one is better than another, or more likely than
#another, as a category, or something like that. 

# So this then seems like it does a pretty good job, of taking inputs and trying to predict what outputs should 
#be.

# But it is important to think about what the limitations of this sort of approach is.

# Just taking some linear combination of inputs, and passing it in to some sort of activation function.

# And it turns out that when we do this in the case of binary classification, of trying to predict something like
#does it belong to one category or another, we can only predict things that are linear separable.

# Because we're taking a linear combination of inputs, and using that to define some decisioon boundary, or
#threshold.

# And what we get is a situation where if we have some linear set of data, we can predict a line that separates 
#linearly, one category of data points from another.

# But a single unit that is making a binary classification, otherwise known as a perceptron, can't deal with a 
#situation that is non linearly separable.

# Where there is no straight line that will go straight through the data that will divide one category away
#from another, it's a more complex decision boundary.

# The decision boundary somehow needs to capture things inside of each category separately, and there isn't 
#really a line that will allow us to deal with that.

# So this is the limitation of the perceptron.

# These units that just make these binary decisions based on their inputs.

#   Perceptron -
# - Only capable of learning linearly separable decision boundary.


# All it can do is define a line based on a linear decision boundary.

# And so this doean't seem like it's going to generalize well for situations where real world data is involved,
#because real world data often isn't linear separable.

# It often isn't the case that we can just draw a line through the data, and be able to divide it up into 
#multiple groups.

# So what then is the solution to this?

# What was proposed was the idea of a mutilayer neural network

#   Multilayer Neural Network -
# - Artificial neural network with an input layer, an output layer, and at least one hidden layer.

# So far all of the neural networks we seen have had a set of inputs and a set of outputs.

# And the inputs are connected to those outputs. (By way of weighted edges)

# But in a multilayered neural network, this is going to be an artificial neural network that has an input layer
#still, it has an output layer, but also has one or more hidden layers in between.

# Other layers of artificial neurons, or units, that are going to calculate their own value as well.

# So instead of a neural network that looks like this, with three inputs and one output.


# O
#  \
# O -- O
#  /
# O


# We might imagine in the middle of this, injecting a hidden layer.

# Something like this.



#      O
# O
#      O
# O         O
#      O
# O
#      O


# This is a hidden layer that has four nodes.

# And we can choose how many nodes or units go into the hidden layer, and we can have multiple hidden 
#layers as well.

# And so now each of the inputs isn't directly connected to the output.

# Each of the inputs is connected to the hidden layer, and then all of the nodes in the hidden layer, those are
#connected to the output.

# And so this is just another step that we could take towards calculating more complex functions.

# Each of the hidden units will calculate its output value, otherwise known as its activation, based on a linear
#combination of all the inputs. 

# And once we have values for all of the nodes in the hidden layer, we do the same thing again, calculate
#the output for the actual output neuron, based on multiplying each of the values for the hidden layer units
#by their weights as well.

# So in effect, the way this works is we start with inputs, they get multiplied by weights in order to calculate
#values for the hidden layer nodes, then those get multiplied by weights in order to figure out what the 
#ultimate output is going to be.

# The advantage of layering things like this is it gives us an ability to model more complex functions.

# Instead of just having a single decision boundary, a single line, dividing our data sets from one another,
#each of the hidden nodes can learn a different decision boundary, and we can combine those decision boundaries
#to figure out what the ultimate output is going to be.

# And as we begin to imagine more complex situations, we could imagine each of the hidden layer nodes learning
#some useful property, or learning some useful feature, of all of the  inputs, and us somehow learning how to
#combine those features together in order to get the output that we actually want.

# Now the natural question would be, how do we train a neural network that has hidden layers inside of it.

# And this turns out to be an initially tricky question because the input data that we're given is values for all
#the inputs, and we're given what the value of the output should be, what the category is for example, but the 
#input data doesn't tell us what the values for the hidden layer nodes should be.

# So we don't know how far off the hidden layer nodes actually are, because we are only given data for the inputs
#and the output.

# The reason that it's called the hidden layer is because the data that is made available to us doesn't tell us 
#what the values for the hidden layer nodes should  actually be.

# And so the strategy that people came up with was to say, that if we know what the error, or the loss is for the 
#output node, well then based on what the weights are, we can calculate an estimate for how much the error for 
#each hidden layer nodes weight.

# In effect saying that based on the error from the output, we can backpropagate the error and figure out and 
#estimate for what the error is for each of the hidden layer nodes.

# And the idea for this algorithm is known as backpropagation.

#   Backpropagation -
# - Algorithm for training neural networks with hidden layers.

# And the idea for this, the psuedo code for it, will again be if we want to run gradient descent with
#backpropagation, we'll start with a random choice of weights, as we did before.

# And now we'll go ahead and repeat the training process, again and again.

# But what we're going to do each time, we're goning to calculate the error for the output layer first.

# We know the ouput and what it should be, and we know what we calculated, so we can figure out what the error
#there is.

# But then, we're going to repeat for every layer, starting with the output layer, moving back into the hidden layer,
#then the hidden layer before that, if there are multiple hidden layers, going back all the way to the very first
#hidden layer, assuming there are multiple, we're going to propagate the error back one layer.

# Whatever the error was from the output, figure out what the error should be one layer before that, based on
#what the values of those weights are.

# And then we can update those weights.


#   Backpropagation 

# Start with a random choice of weights
# Repeat:
#   :Calculate error for output layer.
#   :For each layer, starting with ouput layer, and moving inwards towards earliest hidden layers:
#       :Propagate error back one layer.
#       :Update weights


# So graphically the way we might think about this, is we first start with the output.

# We know what the output should be, we know what output we calculated, and based on that we can figure out
#how we need to update the weights backpropagating the error to our hidden layer nodes, and using that we can
#figure out how we should update the input weights.

# And we might imagine if there are multiple layers we can repeat this process again and again to begin to figure
#out how all of the weights sholud be updated.

# And this backpropagtion algorithm is really the key algorithm that makes neural networks possible.

# Makes it possible to take these multi level structures, and be able to train those structures 
#depending on what the values of those weights are in order to figure out how it is we should go about updating
#those weights in order to create some function that is able to minimize the total amount of loss.

# Figure out some good setting of the weights that will take the inputs, and translate it into the output we
#expect.

# And as we stated, this works not just for a single hidden layer, but for multiple hidden layers.
