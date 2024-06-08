# Learning - This is where we take a look at Machine Learning, or Learning more generally, and looking at how, when
#we have access to data, our computers can be programmed to be quite intelligent by learning from data and learning
#from experience, being able to perform a task better and better based on greater access to data.
 

# One of the most popular ideas behind machince learning is the concept of supervised learning. 

#Supervised learning is a particular type of task that refers to a task that we give to a data set, where
#that data set consists of input-output pairs, and what we would like the computer to do is, we would like 
#our AI to able to figure out some function that maps inputs to outputs.


#           Supervised Learning -
# - Given a data set of input-output pairs, learn a function to map inputs to outputs.

# So we have a whole bunch of data that generally consists of some kind of inputs.
# Some evidence of information the computer will have access to, and what we would like the computer to do is, 
#based on that input information, is predict what some output is going to be.

# And we'll give it some data for the computer to train its model on, and begin to understand how it is this
#information works and how it is the inputs and outputs relate to each other.

# But ultimately we hope that our computer will be able to figure out some function, that given those inputs, is
#able to get those outputs.

# There are a couple of different task within supervised learning, but the one we'll focuse on and start with
#is known as classification.


#           Classification -
# - Supervised Learning task of learning a function, mapping an input point to a discrete category

# And classification is the problem, where if we are given a whole bunch of inputs, we need to figure out some 
#way to map those inputs into discrete categories, where we can decide what those categories are, and it's the
#job of the computer to predict what those categories are going to be.

# So that might be for example, we are given information about a bank note, like a U.S dollar, and we are asked to
#predict, does it belong to the category of authentic bank note, or does it belong to the category of 
#counterfeit bank notes.

# We need to categorize the input, and we want to train the computer to figure out some function to do that
#calculation.

# Another example might be the case weather. 
# Where we would like to predict, on a given day, is it going to be rain on that day, or cloudy on that day.

# And before we see how we can do this, if we really give the computer all the exact probablities for,
#if these are the conditions, what is the probability of rain?
# Often times we don't have access to that information though, but we do have access to is a whole bunch of 
#data.

# So if we wanted to predict something like is it going to rain, or be cloudy, we would give the computer
#historical information about days when it was raining, and days when it was not raining, and ask the computer
#to look for patterns in that data. 

# So what might that data look like?

# We could structure that data in a table like this, where for any particular day, going back, we have 
#information about that days Humidity, that days air pressure, and importantly we have a label, something
#where we (the Human) has said, on this particular day, it was raining, or it was not raining.



#       Date        |       Humidity        |       Pressure        |       Rain
#                   |   (relatve humdity)   |    (sea level, mb)    |
#_____________________________________________________________________________________
# January 1         |       93%             |     999.7             |   Rain
#-------------------------------------------------------------------------------------
# January 2         |       49%             |     1015.5            |   No Rain
#-------------------------------------------------------------------------------------
# January 3         |       79%             |     1031.1            |   No Rain
#-------------------------------------------------------------------------------------
# January 4         |       65%             |     984.9             |   Rain
#-------------------------------------------------------------------------------------
# January 5         |       90%             |     975.2             |   Rain
#-------------------------------------------------------------------------------------

# What makes this what we would call a supervised learning exercise, is that a Human (us) has gone in
#and labeled each of these data points.

# And what we would like the computer to be able to do, is to figure out, given these inputs, what label
#should be associated with that day.

# Does that day look like it's going to be rain, or does it look more like a day where it is not going to rain.

# Put a little more mathematically, we can think of this as a function, that takes two inputs.
# The inputs being the data points that our computer will have access to, like the Humidity and the pressure,
#so we can start to write a function, where f takes in two parameters, f(Humidity,pressure), and the output
#is going to be what categories we would descibe to these particular input points.
# What label we would associate with that input.
# We see a couple example data points below, where we are given two values, one for Humidity and one for pressure,
#and we predict, is it going to rain, or not going to rain.

# And this is just information we gathered from the world. We measured, on various different days, what the
#Humidity and pressure were, we observed whether or not there was rain, or no rain on that particular day.


# f(Humidity,pressure)
#      f(93,999.3) = Rain
#      f(49,1015.5) = No Rain
#      f(79,1031.1) = No Rain

# This function f is what we want to approximate. 

# Us Humans and computers don't know exactly how this function f works, it's probably a complex function.

# So what we are going to do instead, is attempt to estimate it. 

# We would like to come up with a hypothesis function.
# h(Humidity, pressure)

# Which is going to try to approximate what f does.

# We want to come up with a function h that will also take the same inputs, and will also produce an output,
#rain, or no rain.

# And ideally we would like these two functions to agree as much as possible. 

# So the goal then of this supervised learning classification task, is going to be to figure out, what does 
#that function h look like.

# How can we begin to estimate, given all this information and data, what category or label should be assigned
#to what particular data point.

# So where can we begin doing this?

# Well the reasonably thing to do, is to try to plot this on a graph that has two axis points, x axis, and y axis.

# And in this case we are just going to be using two numerical values as inputs. 

# But these same types of ideas scale as we add more and more input as well.
# We'll be plotting things in 2 dimensions, that as we'll soon see, we can add more inputs.


# f(Humidity,pressure)
#      f(93,999.3) = Rain
#      f(49,1015.5) = No Rain
#      f(79,1031.1) = No Rain
# h(Humidity, pressure)


# Our x axis will represent Humidity, and our y axis will represent pressure.

# And what we might do is say, let's take all of the days that were raining, and try to plot them on this graph
#and see where they fall.

# Here are all the rainy days represented by the lowercase letter o.
# Each rainy day corresponds to a particular value for Pressure and a particulat value for Humidity.

# And then we'll do the samething for the days where there was no rain.

# So take all the not raining days and figure out what there values were for each of these two inputs
#(humidity,pressure), and go ahead and plot them this plot as well.

# They will be reprsented by x (not to be confused with the axis)

# This will be the input that our computer has access to. 

# And what we would like for the computer to be able to do, is to train a model, such that if we are ever 
#presented with a new input, that doesn't have a label associated with it, something the like y dot in out graph,
#(not to be confused with the y axis).

# We would like to know if we should classify this y as a rainy day(o), or a not rainy day(x).

# Based on visuals alone, we would probably say that it belongs in the rainy day category, based on its
#placement on the graph.

# It might not be totally accurate, but it is apretty good guess.

# And this type of algorithm is a pretty popular one in machine learning. The nearest neighbor claasification.

#       Nearest-Neighbor Classification -
# - Algorithm that, given an input, chooses the class of the nearest data point to that input.

# By class, we just mean category, like rain, or not rain, counterfeit, or not counterfeit etc etc.

# And we choose the category, or class, based on the nearest data point. 

# So given all that data we just looked at, we'll just check to see if the nearest data point is an x 
#or an o.

# And depending on the answer to that question, we were able to make some sort of judgement.
# We were able to say something like, we think it's going to be an o, or we think it's going to be an x.

# So likewise we can apply this to other data points we encounter as well.

# If suddenly we get a new data point represented by the z on our graph, we can see that its nearest data
#point is x, so we would go ahead and classify that as an x day.

# Things get a little bit tricky though, when we look at a point like the t on our graph.

# We will ask the same sort of questions, should it belong to the x's, or o's?

# The nearest neighbor classification would say that the way we solve this problem is by looking at which
#point is nearest to that point. 

# We can see that the nearest point is an x, which is a not rainy day, and therefore, according to the nearest
#neighbor classification, we would say that the t is a not rainy day.

# Our first judgement is to say that t is a not rainy day, because of the n-n c, but what if we took a look
#at the bigger picture?

# Yes it is true that the nearest point to it is an x, but it's surrounded by a whole bunch of o days.

# So looking at the bigger picture, there's potential an argument to be made that our t should in fact
#be in the o category.

# And with only this data we actually don't know for sure. We are given some input, something we are 
#trying to predict, and we don't necessarily know what the output is going to be, so in this case,
#which one is correct is difficult to say, but often times considering more than just a single neighbor,
#considering multiple neighbors can give us a better result.



# x axis (Pressure)
#|                          x            o
#|                       x             xt
#|                  z          x      o
#|                    x              x      o
#|                     x                     o
#|                               x    o        o
#|                           x       o
#|                                             o    o
#|                 x                x      o
#|                                             o
#|                       x         o       o      o
#|                                            y
#|                     x          x
#__________________________________________________________
#                                    y axis (Humidity)


# There is a variant on the nearest neighbor classification algorithm which is known as the 
#k-nearest-neighbor classification.

#       k-Nearest-Neighbor-classification -
# - Algorithm that, given an input, chooses the most commom class out of the k nearest 
# - data points to that input.

# Where k is a parameter, some number that we choose, or how many neighbors are we going to look at.

# So one nearest neighbor classificaton is what we saw before, where we just pick the one nearest neighbor,
#and use that category.

# But with k-nearest-neighbor classification, where k might be three or five or seven, we say look at the
#three or five or seven closest neighbors, closest data points to that point, it works a little bit differently.

# The k algortithm, when given an input, should chosse the most common class out of the k nearest data points to
#that input.

# So we look at the five nearest points, and if three of them say it's raining, and two of them say it's not,
#we'll go with the three instead of the two.
# Because each one effectively gets one vote towards what they believe the category should be.

# And ultimately we choose the category which has the most votes. 

# And it turns out that this can work really well for solving a variety of different types of classification
#problems.

# But not every model is going to work under every situation, and so one of the things we'll take a look at, 
#espeacially in the context of machine learning, is the number of different approaches to machine learning,
#a number of different algorithms that we can apply, all solving the same type of problem, all solving
#some classification problem, where we want to take inputs, and organize them into different categories.

# And no one algorithm is necessarily going to be better than some other algorithm, they each have their
#trade offs depending on the data.

# And so this is what a lot of machine learning research ends up being about.
# When we are trying to apply machince learning techniques, we are often not looking at just one particular
#algortithm, but trying multiple different algorithms.

# Trying to see what is going to get us the best results for trying to predict some function that maps inputs
#to outputs.

# So what then are the draw backs of k-nearest-neighbor classification?

# There are a couple.
# One might be in the naive approach at least. It could be fairly slow to have to go through and have to measure
#the distance between a point for every single one of the points in our graph.  

# Now there are ways of trying to get around that. 
# There are data structures that could help to make it more quickly to be able to find these neighbors.

# There are also techniques that we could use to try and prune some of this data, remove some of the data points,
#so that we are only left with the relevent data points, just to make it a little more easier.

# But ultimately, what we might like to do, is come up with another way of trying to do this classification.

# One way of trying to do the classifiction, was looking at what are the neighboring points.

# But another way might be to try to look at all of the data, and see if we can come up with some decision
#boundry. Some boundry that will separate, the rainy days from the not rainy days.

# And in the case of 2 dimensions, we can do that by drawing a line, for example.

# So what we might want to try to do, is just find some line, some separator, that divides the rainy days,
#from the not rainy days.

# We are now trying a different approach, in contrast with the nearest neighbor approach, which just looked
#at local data around the input data point we cared about.

# Now what we are doing, is trying to use a technique known as linear regression, to find some sort of line,
#that will separate the rainy days from the not rainy days.


# x axis (Pressure)
#|                          x              /o
#|                       x             x /    o
#|                  x          x       /     o
#|                    x             x/     o
#|                     x           /       o
#|                        x      /     o        o
#|                   x         /     o
#|                           /                   o    o
#|      x        x         /       o
#|           x           /     o       o      o
#|                     /                       o
#| x          x      /
#|__________________________________________________________
#                                    y axis (Humidity)

# Sometimes it will actually be possible to come up with some line that perfectly separates all the rainy days
#from the not rainy days.

# Realisticly though, many data strutures won't have a perfectly straight line. Often times, data is messy.
# There are outliers, there's random noise that happens inside of a particular system, and what we would like 
#to do is still be able to figure out, what a line might look like.

# So in practice, the data will not always be linearly separable.

# Where linearly separable, refers to some data where we could just draw a line to seaprate the two halfs of 
#perfectly.

# Instead, we might have a situation like this, where there are some rainy days mixed in with some not rainy days.

# And there may not be a line that perfectly separates the two halves.

# We can still say that line still does a pretty good job. And we'll try to formalize a little bit later what 
#we mean we say something like this line does a pretty good job of trying to make that prediction.

# But for now, let's just say we are looking for a line, that does as good as job as we can at trying to separate 
#one category of things, from another category of things. 


# x axis (Pressure)
#|                          x       /     o
#|                       x       xo/
#|                  x          x  /    o
#|                    x          /x      o
#|                     x        /             o
#|                            x/    o        o
#|                            /x       o
#|                           /                  o    o
#|                 x     x o/
#|                         /                    o
#|                       x/ o       o      o
#|                       /                     o
#|              x      x/
#|__________________________________________________________
#                                    y axis (Humidity)


# So now, let's try to formalize this a little more mathematically.

# We want to come up with some sort of function.

# Someway we can define this line.

# And our inputs are things like humidity, and pressure, in this case.

# So our inputs we might have are x1, which represents humidity.
# And x2, which will represnt our pressure.

# These are inputs that we are going to provide to our machine learning algorithm.

# And given those inputs, we would like for our model to predict some sort of output.

# And we are going to predict that using the hypothesis function, which we called (h)


# h(x1,x2) =

# Our hypothesis function is going to take as inputs x1 and x2, humidity and pressure in this case.

# And we can imagine that if we didn't just have two inputs, we had three or four or five, we could have 
#this hyothesis function take all of those as inputs.

# And we'll see examples of that in just a minute.

# And now the question is, what does this hypothesis function do?

# Well, it really just needs to measure, is this data point on one side of the boundary, or is it on the 
#other side of the boundary?

# And how do we formalize that boundary?

# Well, the boundary is generally going to be a linear combination of these input variables, at least in
#this particular case.

# So what we are trying to do when we say linear combination is take each of our inputs and multiply them
# by some number that we are going to have to figure out.

# We'll generally call that number a weight, for how important should these variables be in trying to determine
#the answer.

# So we'll weight each of our variables with some weight, and we might add a constant to it just to try and 
#make the function a little bit different.

# And the result, we just need to compare.

# Is it greater than zero, or less than zero.
# Basically, does it belong on one side of the line, or the other?


# And so what that mathematical expression might look like is this.

# Rain if w0 + w1x1 + w2x2 > 0

# We would take each of the variables, x1 and x2, multiply them by some weight.

# We don't yet know what that weight is, but it's going to be some number, weight 1 and weight 2.

# And maybe we just want to add some other weight 0 to it, because the function might require us to shift 
#the entire value up or down by a certain amount.

# And then we just compare.

# If we do all the math, is it greater than or equal to zero?

# If so, we might categorize that data point as a rainy day.

# Otherwise, we might say, no rain.

# So the key here, then, is that this expression is how we are going to calculate if it is a rainy day or not.

# We're going to do a bunch of math where we take each of the variables, multiply them by a weight, maybe add
#an extra weight to it, and see if the result is greater than or equal to zero.

# And using that result of that expression, we're able to determine whether it's raining or not raining.

# This expression here, in this case is just going to refer to some line.
#                           |
#                           | 
#              Rain if w0 + w1x1 + w2x2 > 0
# h(x1, x2) ========= |     |
#              NO Rain Otherwise   

# If we were to plot that graphically, it would just be some line.

# And what the line actually looks like depends upon these weights.

# x1 and x2 are the inputs, but the weights are really what determiine the shape and or slope of that line,
#and what that line actually looks like.

# So we then would like to figure out what these weights should be.

# We can choose whatever weights we want, but we want to choose weights in such a way that if we pass in 
#a rainy day's humidity and pressure, then we end up with a result that is greater than or equal to 0.

# And we would like it such that if we passed into our hypothesis function, a not rainy day, then the output that we get 
#should be not raining. 

# So before we get there, let's try to formalize this a little bit more mathematically just to get a sense 
#for how it is that we'll often see this if we ever go further into supervised machince learning and explore
#this idea.

# One thing generally for these categories, we'll sometimes just use the names of the categories, like rain,
#or not rain.

# Often mathematically if we're trying to do comparisons between these things, it's easier to just deal in the 
#world of numbers, so we could just use the express below, for example.

#              1 if w0 + w1x1 + w2x2 > 0
# h(x1, x2) =========      
#              0 Otherwise


# So we do all this math, and if the result is greater than or equal to zero, we'll go ahead and say our
#hypothesis function output is 1, meaning raining.

# Otherwise it outputs zero, meaning not raining.

# And oftentimes, this type of expression, we'll instead express using vector mathematics.

# And all a vector is, is a sequence of numerical values. 

# We could represent that in python using something like a list of numerical values, or a tuple with numerical
#values.

# Here, we have a couple of sequences of numerical values.

# One of our vectors, are all of our individual weights. 

# So we could construct what we'll call a weight vector.

#   Weight Vector w: (w0, w1, w2)

# And we'll see why this useful n a moment.

# We'll call it (w), generally using a bold face w, and it's just a sequence of our 3 weights.

# And to be able to calculate based off of these weights whether we think a day is rainy, or not rainy,
#we're going to multiple each of those weights by one of our input variables.

# For example w2 is going to be multiplied by x2, w1 is going to be multiplied by x1, and w0, well, that's
#not being multiplied by anything, but just to make sure the vectors are the same length, we'll say that
#w0 is being multiplied by 1.

# So in addition to the weight vector w, we'll also have an input vector x, that has 3 values.

#   Input Vector x: (1, x1, x2)

# So here we have reprresented two distinct vectors.

# A vector of weights, that we need to somehow learn.
# The goal of our machine learning algorithm is to learn what this weight vector is supposed to be.
# We could choose any set of numbers and it would produce a function that tries to predict rain, or not rain.
# But it probably wouldn't be any good.

# What we want to do, is come up with a good choice of these weights, so that we're able to do the accurate
#predictions.

# And then our input vector represents a particular input to the function.
# A data point, for which we would like to estimate, is that day a rainy day, or is that day not raining.

# And that's going to vary depending on what input is provided to our function.
# What it is that we are trying to estimate.

# And then to do the calculation, we want calculate this expression here, 1 if w0 + w1x1 + w2x2 > 0
#                                                                         0 Otherwise

# And it turns out that expression is what we would call the dot product of these two vectors.
# The dot product of two vectors, just means taking each of the terms in the vectors, and multiplying 
#them together. (w0 multiplied by 1, w1 multiplied x1, w2 multiplied by x2)
# That's why our vectors need to be the same length.

# And then we just add all of the results together.

#  So the dot product of w and x, our weight vector and input vector is just going to be w0 + w1x1 + w2x2

# W . X: w0 + w1x1 + w2x2

# So we have our weight vector we need to figure out. We need our machine learning algorithm to figure
#out what the weights should be.

# We have the input vector, representing the data point that we are trying to predict a category for,
#or predict a label for.

# And we're able to do that calculation, by taking this dot product, which we'll often see represented
#in vector form, and then seeing if the result is greater than, or equal to zero.

# This expression here W . X: w0 + w1x1 + w2x2, is identical to the expression that we are calculating to 
#see whether or not that answer is greater or equal to zero.

# For that reason, we'll often see the hypothesis function written as something like this.

#        1 if W . X > 0
#hw(X) = 
#        0 otherwise


# A simple representation where the hypothesis takes as input, some input vector X, some humdity and pressure
#for some day.

# And we want to predict an output, like rain or not rain, or 1 or 0, however we choose to represent things
#numerically.

# And the way we do that is by taking the dot product of the weights, and our input.
# If it's greater than or equal to zero, we'll go ahead and say the output is 1, otherwise the output is zero.

# And the hypothesis, we say is paramaterised by the weights.
# Depending on the weights we choose, we'll end up getting a different hypothesis.

# If we choose the weights randomly, we're probably not going to get a very good hypothesis function.

# We'll get a 1 or a 0, but it's probably not going to accurately reflect whether we think the is going 
#to be raining or not raining.

# But if we choose the weights right, we can often do a pretty good job of trying to estimate whether we think the 
#output function is going to be a 1 or a 0.

# And so the question then, is how to figure out what these weights should be.

# How to be able to tune those parameters.

# And there are a number of ways that we can do that.

# One of the most common, is known as the perceptron learniing rule.
# The idea of the perceptron learning rule is to say that given some data point that we would like to learn from,
#some data point that has an input (x) and an output (y), where y is like 1 for rain or 0 for no rain, then we
#are going to update the weights.

# But the big picture idea, is that we can start with random weights.
# But then learn from the data. 
# Take the data points one at a time.
# And for each one of the data points, figure out what parameters do we need to change inside of the weights
#in order to better match that input point. 

# So that is the value of having access to a lot of data in a supervised machine learning algorithm.

# We can take each of the data points and look at them mltiple times and constantly try and figure out if we 
#need to shift our weights in order to better create some weight vector that is able to correctly or more
#accurately try to estimate what the output should be. Whether we think it's going to be raining or not raining.

# So what does that weight update look like?

# We are going to update each of the weights, to be the result of the original weight plus some additional
#expression.

# And to understand this expression, (y) is what the actual output is.
# And the hypothesis of X the input, (hw(x)), is going to represent what we thought the input was. 


#       Perceptron Learning Rule -
# - Given data point (x,y), update each weight according to: wi = wi + a(y - hw(X)) * xi 


# If we wanted to, we could represent the the original equation like this
#   a(actual value - estimate)
# And based on the difference between the actual value and the estimated value, we might want to change our
#hypothesis. 
# Change the way we do that estimation.

# If the actual value and the estimate were the samething, meaning we were correctly able to predict what 
#category this data point belonged to, well actual value minus estimate is going to be zero.

# Which means the entire term on the right hand side is going to be zero, and the weight doesn't change.

# Weight i, where (i) is like w1 or w2 or w0, weight i just stays as weight i, and none of the weights change.

# That is if, we were able to corectly predict what category the input belonged to.

# But if our hypothesis didn't correctly predict what category input belonged to, well maybe then, we need to
#make some changes.
# Adjust the weights so that we are better able to predict this kind of data point in the future.

# And what is th way we might do that?

# Well if the actual value was bigger than the estimate, then that means we need to increase the weight 
#in order to make it such that the output is bigger, and therefore we are more likely to get to the right
#actual value.

# And so if the actual value is bigger than the estimate, then actual value minus estimate will return a 
#positive number.

# Then we'll just be adding some positive number to the weight, to increase it, ever so slightly.

# And likewise the inverse is true.

# That if the actual value is less than the estimate, then we want to decrease the value of the weight,
#because then in that case we want to try and lower the total value of computing that dot product, in order
#to make it less likely that we would predict that it would actually be raining.