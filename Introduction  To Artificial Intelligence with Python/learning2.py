# The general idea is that every time we encounter some data point, we adjust the weights accordingly to try and make 
#the weights better line up with the actual data that we have access to.

# And we can repeat this process with data point after data point until eventually, hopefully, our algorithm
#converges to some set of weights that do a pretty good job of trying to figure out whether a day is going to be
#raining or not raining.

# And just as a final point about this particular equation, (wi + a(y - hw(X)) * xi), this value alpha here (a)
#is generally what we'll call the learning rate.
# It's just some parameter, some number we choose for how quickly we're actually going to be updating these
#weight values.
# So that if alpha is bigger, then we're going to update these weight values by a lot.
# And if alpha is smaller, then we're going to update the weight values by less.
# And we can choose the value of alpha. 
# Depending on the problem, different values might suit the situation better or worse than others.


# So after all of that, after we've done this training process of take all this data and using the learaniing rule,
#look at all the pieces of data and use each piece of data as an indication to us of do the weights stay the same,
#do we increase the weights, do we decrease the weights, and if so, by how much?

# What we end up with is effectively a threshold function.

# And we can look at what the threshold function looks like, like this.


# y-axis
#   |
# 1 |
#   |
#   |                                     _____________________________________
# O |                                     |
# U |                                     |
# T |                                     |
# P |                                     |
# U |                                     |
# T |                                     |
#   |                                     |
#   |                                     |
#   |                                     | 
#   |                                     |
#   |                                     |
# 0 |_____________________________________|
#_________________________________________________________________________________________________
#                                        W.X                                    x-axis


# On the x axis we have the output of that function, taking the weights, taking the dot product of it with the 
#input.

# And on the y axis, we have what the output is going to be, 0, which in this case represents not raining,
#and 1, which in this case repreents raining.

# And the way that our hypothesis function works is it calculates this value. (W.X)
# And if it's greater than 0, or greater than some threshold value, then we declare that it is a rainy day.

# And otherwise, we declare that it's not a rainy day.

# And this then graphically is what that function looks like, that initially when the value of this product
#is small, it's not raining, but as soon as it crosses that threshold, we suddenly say ok, now it's raining.

# And the way to interpret this kind of representation is that anything on the right side of the line, that 
#would be the category of data points where we say, yes, it's raining.

# Anything that falls on the left side of the line are the data points where we would say, it's not raining.

# And again, we want to choose some value for the weights that results in a function that doees a pretty good job 
#of trying to do this estimation.

# But one tricky thing with this type of hard threshold, is that it only leaves two possible outcomes.

# We plug in some data as input.

# And the output we get is raining or not raining.

# And there's no room for anywhere in between.

# And maybe that's what we want sometimes.

# Maybe all we want is given some data point, we would like to be able to classify it into one or two or more
#of these various different categories.

# But it might also be the case that we care about knowing how strong that prediction is, for example.

# So if we go back to this instance here, where we have rainy days on the right side of the line, and not 
#rainy days on the left side of the line.

# Let's imagine that we have two white dots and we'll look at those two data points now.



# x axis (Pressure)
#|                          x       /     o
#|                       x       xo/
#|                  x          x  /    o
#|                    x          /x      o
#|                     x        /             o
#|                            x/    o        o
#|                            /       o
#|                           /                  o    o
#|                 x     x o/                   w
#|                         /                    o
#|                       x/ o       o      o
#|                       /x                    o
#|              x      x/w
#|__________________________________________________________
#                                    y axis (Humidity)


# We would like to predict a label or category for these two data points (white dots)

# It seems likely that we could pretty confidently say that the data point to the far right is going to be a rainy
#day.
# It seems close to the other rainy days if we're going by the nearest neighbor strategy.

# And if we are using the line strategy of just which side of the line does it fall on, which side of this decision
#boudary, we would also say that the data point to the left is also going to be a rainy day, because of its position
#on the decision line. 
# But it's likely that even in this case, we don't feel nearly as confident about this data point
#on the left, compared to the data point on the right.

# For the data point on the right, we can feel very confident that yes, it's a rainy day.

# The data point to the left is pretty close to the line if we're judging just by distance.

# And so we might be less sure.

# But our threshold function doesn't allow for a notion of less sure or more sure about something.
# It's what we would call a hard threshold.
# Once we cross that threshold line, then immediately we say yes, this is going to be a rainy day.
# Anywhere before we cross the threshold line, we are going to say that is not a rainy day.

# That may not be helpful in a number of cases.

# One, this is not a particularly easy function to deal with.

# As we get deeper into the world of machine learning and are trying to do things like taking derivatives of these
#curves with this type of function makes things challenging.

# But the other challenge is that we don't really have any notion of gradation between things.
# We don't have a notion of yes, this is a very strong belief that it is going to be raining as opposed to it's
#likely than not that it's going to be raining, but maybe we're not totally sure about that either.

# So what we can do by taking advantage of a technique known as logistic regression, is instead of using the 
#hard threshold type of function, we can can use instead a logistic function, something we miht call a soft threshold.

# And that's going to transform our hard threshold into something that looks more like this, something that more
#nicely curves.


# y-axis
#   |       Soft Threshold
# 1 |
#   |
#   |                                                            ________________________________
# O |                                                          _/
# U |                                                        _/
# T |                                                      _/
# P |                                                    _/
# U |                                                  _/
# T |                                                _/
#   |                                              _/
#   |                                            _/
#   |                                          _/
#   |                                        _/
#   |                                      _/
# 0 |_____________________________________/
#_________________________________________________________________________________________________
#                                        W.X                                    x-axis


# And as a result, the possible output values are no longer just 0 and 1.
# 0 for not raining, and 1 for raining.

# But we can actually get any real numbered value between 0 and 1.

# But if we're all the way to the far left, we still get the value of 0.
# Meaning that we're pretty sure that it's not going to be raining.

# Likewise, if we are to the far right, we still get a value of 1.
# Meaning that we're pretty sure that it is going to be raining.

# But in bewtween, we could get some real numbered value, where a value like 0.7 might mean we think it's 
#going to rain.
# It's more probable that it's going to rain than not, based on the data.

# But we're not as confident as some of the other data points might be.

# So one of the advantages of a soft threshold is that it allows us to have an output that could be some real
#number that potentially reflects some sort of probablilty. 
# The likelihood that we think that this particular data point belongs to that particular category.

# And there are some other nice mathematical properties of that as well.


# So that then is two differebt approaches to trying to solve this type of classification problem.

# One is this nearest neighbor type of approach, where we just take a data point and look at the data points
#that are nearby to try and estimate what category we think it belongs to.

# And the other approach is trying to use linear regression, figure out what the weights should be, adjust the
#weights in order to figure out what line or decision boundary is going to best separate the two categories.

# It turns out that another popular approach, a very popular approach if we just have a data set and we want to start
#trying to do some learning on it, is what we call the support vector machine.

# And we're not going to go too deep into the mathematics of the support vector machine, but we'll at least 
#explore it graphically to see what it is that it looks like.

# And the idea, the motivation behind the support vector machine, is the idea that there are actually a lot of 
#different lines that we can draw, a lot of different decision boundaries that we could draw to separate the two
#groups.

# So for example we have the x's data points to the left, and the o's data points to the right.

# One possible line we could draw is a line like this.


# x axis (Pressure)
#     x                        |                          o
#|      x             x        |                      o
#|     x          x            |                       o
#|       x             x       |                     o
#|             x               |                      o
#|           x                 |                 o        o
#|        x                    |                      o
#|                             |                        o    o
#|    x        x               |                          o
#|         x                   |                      o       o      o
#|                             |                          o
#| x          x                |
#|_____________________________|_____________________________
#                                    y axis (Humidity)

# This line would separate our x data points from our o data points.
# And it does so perfectly.

# All the x data points are on one side of the line.

# All of the o data points are on the other side of the line.

# But this should probably make us a little bit nervous.

# If we come up with a model and the model comes up with a line like this.

# And the reason why is that we worry about how well it's going to generalize to other data points that are
#not necessarily in the data set that we have access to.

# For example, if there was a data point that fell right here, for example (see e dot on graph), on the right 
#side of the line, well, then based on that, we might want to guess that it is, in fact, an x data point, 
#but it falls on the side of the line where instead we would estimate that it's an o data point instead.


# x axis (Pressure)
#     x                        |e                          o
#|      x             x        |                      o
#|     x          x            |                       o
#|       x             x       |                     o
#|             x               |                      o
#|           x                 |                 o        o
#|        x                    |                      o
#|                             |                        o    o
#|    x        x               |                          o
#|         x                   |                      o       o      o
#|                             |                          o
#| x          x                |
#|_____________________________|_____________________________
#                                    y axis (Humidity)

# And so based on that, this line is prpbably not going to be a great choice, just because it is so close to 
#these various data points.

# We might prefer instead a diagonal line that goes diagonally through the data set like we've seen before.

# But there too, there's a lot of diagonal lines that we can draw as well.

# For example we could draw a diagonal line here, which also successfully separates all of the x data points
#from the o data points.


# x axis (Pressure)
#|                 x        /               o
#|         x             x /                  o
#|         x          x   /                o
#|         x            x/                    o
#|           x          /                       o
#|             x       /                   o        o
#|          x         /                        o
#|                   /                   o    o
#|      x        x  /                     o
#|           x     /                  o       o      o
#|                /                         o
#| x          x  /
#|__________________________________________________________
#                                    y axis (Humidity)

# From the perspective of something like just trying to figure out some setting of weights that allows us to predict
#the correct output, this line will predict the correct output for this particular set of data every single time
#because the x data points are on one side, and the o data points are on the other side.

# But yet again, we should probably be a little nervous because this line is so close to the x data points,
#even though we are able to correctly predict on the input data, if there was a point that fell in this general area,
#(see e dot on graph) our algorithm, this model, would say that we think this is an o data point, when actuality
#it might belong to the x data point category instead, just because it looks like it is close to the other x 
#data points.


# x axis (Pressure)
#|                 x        /               o
#|         x             x /                  o
#|         x          x   /                o
#|         x            x/                    o
#|           x          /e                       o
#|             x       /                   o        o
#|          x         /                        o
#|                   /                   o    o
#|      x        x  /                     o
#|           x     /                  o       o      o
#|                /                         o
#| x          x  /
#|__________________________________________________________
#                                    y axis (Humidity)


# What we really want to be able to say, given this data, how can we generalize this as best as possible, is to 
#come up with a line like this that seems like the intuitive line to draw.

# And the reason that it is intuitive is because it seems to be as far as possible from the x data points and the
#o data points.

# So that if we generalize a little bit and assume that maybe we have some points that are different from the 
#input but still slightly further away, we can still say that something on left side, probably x data point,
#something on the right side, probably o data point.

# And we can make those judgements that way.


# x axis (Pressure)
#|                          x             /    o
#|                       x           x   /    o
#|                  x          x        /     o
#|                    x             x  /     o
#|                     x              /       o
#|                        x          /     o        o
#|                   x              /     o
#|                                 /                   o    o
#|      x        x                /       o
#|           x                   /     o       o      o
#|                              /                       o
#| x          x                /
#|__________________________________________________________
#                                    y axis (Humidity)


# And that is what support vector machines are designed to do.

# They're designed to try and find what we call the maximum margin separator, where the maximum margin separator 
#is just some boundary that maximizes the distance between the groups of points, rather than come up with some
#boundary that's very close to one set or the other, where in the case before, we wouldn't have cared.


#   Maximum Margin Separator -
# - Boundary that maximizes the distance between any of the data points.

# As long as we're categorizing the input well, that seems all we need to do.

# The support vector machine will try and find this maximum margin separator, some way of trying to maximize that
#particular distance.

# And it does so by finding what we call the support vectors, which are the vectors that are closest to the line,
#and trying to maximize the distance between the line and those particular points.

# And it works that way in two dimensions.

# It also works in higher dimensions, where we're not looking for some line that separates the two data points,
#but instead looking for what we generally call a hyperplane, some decision boundary, effectively, that 
#separates one set of data from the other set of data.

# And this ability of support vector machines to work in higher dimensions actually has a number of other
#applications as well.

# But one is that it helpfully deals with cases where data may not be linearly separable.

# So we talked about linear separability before, this idea that we can take data and just draw a line or some
#linear combination of the inputs that allows us to perfectly separate the two sets from each other.

# There are some data sets that are not linearly separable.

# And some were even two.

# We would not be able to find a good line at all that would try to do that kind of separation.

# Something like this for example


#                o
#               o   o o o   o   o   o   o   o   o
#                 o                            o   o
#               o                                   o
#             o                                    o
#           o             x x    x  x                   o
#           o            x     x   x  x                o
#                        xx      x  x   x                 o
#           o            xx    x     x  x               o
#                         x x     x  x  x                 o
#            o             x   xx    x  x                  o
#                          xx    x   x  x                o
#               o           x  x   x  x                o o
#               o                                  o o
#                   o                             o o
#                   o     o                    o   o


# Imagine that our x data points are surrounded by our o data points 

# If we try to find a line that divides the x data points from the o data points, it's actually going to difficult,
#if not impossible.

# Anywhere we draw a line, there's going to be a lot of errors and mistakes, a lot of what we'll soon call loss
#to that line that we draw, a lot of points that will end up getting categorized incorrectly.

# What we really want is to be able to find a better decision boundary that may not be a straight line through
#this two dimensional space.

# And what support vector machines can do is they can begin to operate in higher dimensions and be able to find
#some other decision boundary, like a circle around our x data points, for example, that actually is able to
#separate one of these sets of data from the other set of data a lot better.


#                o
#               o   o o o   o   o   o   o   o   o
#                 o                            o   o
#                   o                           o
#                      o()()()()()()() ()()    o
#                     o()  x x    x  x    ()      o
#                   o  () x     x   x  x    ()    o
#                      () xx      x  x   x   ()  o
#                    o () xx    x     x  x   ()    o
#                      ()  x x     x  x  x   ()   o
#                    o ()  x   xx    x  x    ()   o
#                      ()  xx    x   x  x    ()  o
#               o     O ()   x  x   x  x    ()  o o
#                    o   ()()()()()()()()()   o o
#                   o                        o o
#                   o     o              o   o


# So often times in data sets where the data is not linearly seperable, support vector machines by working in
#higgher dimensions can actually figure out a way to solve that kind of problem effectively.

# So that then, three different approaches to trying these sorts of problems.

# We've seen support vector machines.

# We've seen trying to use linear regression and the perceptron learning rule to be able to figure out how to
#categorize inputs and outputs.  

# We've seen the nearest neighbor approach.

# No one necessarily better than the other.

# It's going to depend on the data set, the information we have access to.

# It's going to depend on what the function looks like that we're ultimately trying to predict.

# And this is where a lot of research and experimentation can be involved in trying to figure out how it
#is to best perform that kind of estimation.

# But classification is only one of the tasks that we might encounter in supervised machine learning.

# Because in classification, what we're trying to predict is some discrete category.

# We're trying to predict o data or x data, raining or not raining, authentic or counterfeit.

# But sometimes what we want to predict is a real numbered value.

# And for that, we have a related problem, not classification, but instead known as regression.

#   Regression -
# - Supervised learning tasks of learning a function mapping an input point to a continuous value.

# Basically, regression is the supervised learning problem, where we try and learn a function mapping
#inputs to outputs same as before.
# But instead of the outputs being dicrete categories, things like rain or not rain, in a regression problem,
#the outputs are generally continuous values, some real number that we would like to predict.

# This happens all the time as well.

# We might imagine that a company might take this approach if it's trying to figure out, for instance, what
#the effect of its advertising is.

# Like, how do advertising dollars spent translate into sales for the company's product, for example.

# And so they might like to try to predict some function that takes as input the amount of money spent on
#advertising.

# f(advertising)

# Here we are just going to use one input.
# But again we could scale things up to many more inputs as well, if we have a lot of different kinds of data
#we have access to.

# And the goal is to learn a function, that given this amount of spending on advertising, we're going to get
#this amount in sales.

# f(advertising)
#       f(1200) = 5800
#       f(2800) = 13400
#       f(1800) = 8400

# And we might judge, based on having access to a whole bunch of data, like every past month, here is how
#much we spent on advertising, and here is what sales were.

# And we would like to predict some hypothesis function, that again, given the amount spent on advertising,
#we can predict, in this case, some real number, some number estimate of how much sales we expect that company
#to do in this month, or quarter, or whatever unit of time we are choosing to measure things in.

# f(advertising)
#       f(1200) = 5800
#       f(2800) = 13400
#       f(1800) = 8400

# h(advertising)

# And so again, the approach to solving this type of problem, we can try using a linear regression type of 
#approach, where we take this data and we just plot it.

# On the x axis, we have advertising dollars spent, and on the y axis we have sales.

# And we might want to try and just draw a line that does a prettty good job at trying to estimate this 
#relationship between advertising and sales.


#  y axis     
#    |               /              
#    |              /  o
# s  |        o    /          
# a  |            /     o
# l  |      o    /       
# e  |          /  o
# s  |     o   /     o
#    |        /  o
#    |  o    /  
#    |    o /   
#    |   o /  
#    |  o /  o
#    |   /   o
#    |o /     
#    | /   o
#    |/_____________________________________________________
#                        advertising                  x axis

# And in this case, unlike before, we're not trying to separate the data points into discrete categories,
#but instead in this case we're just trying to find a line that approximates this relationship between
#advertising and sales, so that if we want to firgure out what the estimated sales are for a particular
#advertising budget, we could just look it up in this line and figure out for x amount advertising,
#we would have y amout of sales.

# We could just try to make the estimate that way.

# And we could try and come up with a line, figuring out how to modify the weights using various techniques
#to try and make it so that this line fits as well as possible.

# So with all of these approaches, to try and solve machine learning style problems, the question becomes, 
#how do we evaluate these approaches.

# How do we evaluate the various different hypotheses that we could come up with.

# Because each of these algorithms will give us some sort of hypothesis, some function that maps inputs to
#outputs.

# And we want to know how well does that function work.

#   Evaluating Hypothesis

# And we can think of evaluating these hypotheses and trying to get a better hypothesis, as kinda like an optimization
#problem. 

# In the optimization problems we can recall from before, we would either try to maximize some objective function,
#by trying to find a global maximum.

# Or we were tying to minimize some cost function, by trying to find some global minimum.

# And in the case of evaluating these hypotheses, one thing we might say, is that this cost function, the thing 
#we are trying to minimize, we might be trying to minimize what we call a loss function.

#   Loss Function -
# - Function that expresses how poorly our hypothesis performs.

# More formally, it's like a loss of utility, by whenever we predict something that is wrong, that is a loss
#of utility that's going to add to the output of our loss function

# And we could come up with any loss function that we want.

# It's just a mathematical way of estimating. 

# Given each of these data points, given what the actual output is, and given what our projected output is, our 
#estimate, we could calculate some sort of numerical loss for it.

# But there are a couple of popular loss functions that are worth discussing, just as we've seen them before.

# When it comes to discrete categories, things like rain or not rain, and counterfeit or not counterfeit,
#one approach is the zero one loss function.

#   0-1 Loss Function -

# L(actual,predicted) =
#      0 if actual = predicted
#      1 otherwise

# The way this works is, for each of the data points our loss function takes as input, what the actual output
#is, like whether it was actually raining or not, and takes some prediction into account.
# Did we predict, given this data point, whether it was raining or not.

# And if the actual value does equal the prediction, well then the zero one loss function will just say 0.

# There was no loss of utility, because we were able to predict correctly.

# And otherwise, if the value is not the same thing as what we predicted, well in that case, our loss is one.

# We loss something. 
# We loss some utility. Because what we predicted the output of the function was, was not what it actually was.

# And the goal in a situation like this would be tto come up with some hypothesis that minimizes the total
#imperical loss, the total amount that we loss if we add up for all these data points, what the actual output is
#and what our hypothesis would have predicted.

# So in this case for example, if we go back to classifying days as raining or not raining, and we came up with
#this decision boundary, how would we evaluate this decision boundary.

# How much better is it than drawing a line to the far left or the far right?


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


# Well we could take each of these input data points, and each input data point has a label, whether it was raining,
#orr whether it was not raining, and we could compare it to the prediction.

# Whether we predicted it would be raining or not raining, and assign it a numerical value as a result.

# So for example, our o data points, they  were all rainy days, and we predicted they would be rainy, because they
#they fall on the right side of our line.

# So they have a loss of zero.
# Nothing loss from thos situations.

# And likewise the same is true for most of the x data points, where it was not raining, and we predicted that it 
#would not be raining.

# Where we do have loss, are the o data points that fall on the left side of the line, and the x data points that
#fall on the right side of the line.

# Where we predicted that it would not be raining, but our x data point fell on the right side, and likewise,
#where we predicted that it would be raining but the o data point fell on the left side.

# So as a result we miscategorized these data points that we were trying to train on, and as a result, there is 
#some loss.

# Each data point that was miscategorized will count as a loss of utility.

# That will give us a total loss of 4 in this case.

# That might be how we would estimate how we would say that this line is better than a line that goes to the far 
#left, or the far right, because this line might minimize the loss.

# There is no way to do better than just those 4 points of loss if we're just drawing a straight 
#line through our space.

# So the zero one loss function checks, did we get it right, did we get it wrong. 

# If we got it right the loss is zero. Nothing lost.

# If we got it wrong, then our loss function for that data point says one, and we add up all those loses across
#all of our data points to get some sort of imperical loss.

# How much have we lost across all the original data points that our algorithm had access to.
