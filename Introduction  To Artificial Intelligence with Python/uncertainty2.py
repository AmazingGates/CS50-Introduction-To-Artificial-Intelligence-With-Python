from pomegranate import *

# Here we will be talking about different models we can create. The first is bayesian network

# Bayesian Network -
#                  - Data structure that represents the dependence among random variables.
#                  - Directed Graph
#                  - Each node represents a random variable
#                  - Arrow from X to Y means X is a parent of Y
#                  - Each node X has probability distribution P(X|Parents(X))

# The odds are most random variables in this world are not independent from each other, but there's some
#relationship between things that are happening that we care about. If it is rainy today, that might increase the
#likelihood that my flight or my train gets delayed, for example. There are some dependence between these random
#variables, and a bayesian network is going to be able to capture those dependencies. 
# So what is a bayesian network? What is its actual structure, and how does it work?
# Well, a Bayesian newtork is going to be a directed graph. We've seen a directed graph before. They are 
#individual nodes with arrows or edges that connect one node to another node pointing in a particular direction.
# And so this directed graph is going to have nodes as well, where each node in this directed graph is going to
#represent a random variable, something like the weather, or something like whether my train was on time or delayed.
# And we're going to have an arrow from node x to a node y to mean that x is a parent of y. 
# If there's an arrow from x to y, x is going to be considered a parent of y.
# And the reason that's important is because each of these nodes is going to have a probability distribution that 
#we're going to store along with it, which is the distribution of x given some evidence, given the parents of x.
# So the way to more intuitively think about this is the parents seem to be thought of as sort of causes for some
#effect that we're going to observe. 
# Let's take a look at an actual example of a Bayesian network and think about the types of logic that might be
#involved in reasoning about that network.
# Let's imagine for a moment that we have an appointment out of town, and we need to take a train in order to get to 
#that appointment. So what are the things we might care about?
# Well, we care about getting to appointment on time. And we can use a Bayesian network to represent some different
#scenarios that might affect the outcome of us attending our appointment, and getting there on time.
# We'll use four nodes to draw our graph of probabilities.

#           Rain
#     {none,light,heavy}
#       |             |
#       |             |
#   Maintanace        |
#    {yes,no}         |
#         |           |
#         |   Train   |
#        {on time, delayed}  
#                 |
#                 |
#                 |
#            Appointment   
#          {attend, missed}

# Here we have four nodes representing four random variables that we would like to keep track of.
# These are our possible values. 
# Here are our four nodes, which each represents a random variable, each of which has a domain of possible values 
#that it can take on. And the arrows, the edges pointing from one node to another, encode some notion of
#dependence inside of this graph, that whether we make it to our appointment or not is dependent upon whether the 
#train is on time or delayed. And whetehr the train is on time or delayed is dependent on two things given by 
#the two arrows pointing at this node. It is dependent on whether or not there was maintenance on the train track.
# And it is also dependent upon whether or not it was raining. 
# Let's also say that whether or not there is maintenance on the track, the "Rain" node might be influenced.
# That if there is heavier rain, well, maybe it's less likely that it's going to be maintenance on the tarck that day
#because they're more likely to want to do maintenance on the track on days when it's not raining, for example.
# And so these nodes might have different reltionships between them. But the idea is that we can come up with
#a probability distribution for any of these nodes based only upon its parents.
# And so, let's look node by node at what this probability distribution might actually look like.
# We'll go ahead and begin with the root node, our rain node, which is at the top, and has no arrows pointing
#into it, which means its probability distribution is not going to be a conditional distribution.
# It's not based on anything. We just have some probability distribution over possible values for the rain random
#variable. And that distribution might look a little somethibg like this.

#           Rain               none | light | heavy
#     {none,light,heavy}        0.7    0.2     0.1

# None, light and heavy each have a possible value. Here we are saying the likelihood of no rain is 0.7, of light rain
#is 0.2, of heavy rain is 0.1, for example.
# This is a probability distribution for the root node in our Bayesian network.
# Let's now consider the next node in our network, Maintenance. Track maintenance yes or no. 
# And the general idea of what this probability distribution is going to encode, at least in this story, is the
#idea that the heavier the rain is, the less likely that there's going to be track maintenance. 
# What might track probability distribution look like?
# 

#   Maintanace     R   |  yes  |  no   
#    {yes,no}     none |  0.4  |  0.6
#                light |  0.2  |  0.8
#                heavy |  0.1  |  0.9

# Each of these rows is going to sum up to 1. Because each of these represent different values that that 
#random variable can take on. And each is associated with its own probability distribution that is ultimately
#all going to add up to the number 1. This is a distribution for our random variable called maintenance.

# Now we will look at our next Variable. That will our a node inside our Bayesian network called Train that 
#has two possible values, on time and delayed. And this node is going to be dependent upon the two nodes that 
#are pointing towards it, that whether or not the train is on time or delayed depends on whether or not
#there is track maintenance. And it depends on whether or not there is rain, that heavier rain probably means
#more likely that my train is delayed. 
# And so we could construct a larger probability distribution, a conditional probabilty distribution, that 
#instead of conditioning on just one variable, as was the case with the Rain variable, is now conditioning
#on two variables, conditioning both on rain represented by r and on maintenance represented by yes.
# Again each of these rows has two values that sum up to the number 1, one for whether the train is on time,
#one for whether the train is delayed.
# And here we can say something like, all right, if we know there was light rain and track maintenance, well,
#ok, that would be r is light and m is yes. For example. Well, then there is a probability of 0.6 that my train
#is on time, and a probability of 0.4 the train is delayed.

#              Train                    R  |  M  |  on time  |  delayed
#        {on time, delayed}           none   yes     0.8          0.2
#                                     none    no     0.9          0.1
#                                    light   yes     0.6          0.4
#                                    light    no     0.7          0.3
#                                    heavy   yes     0.4          0.6
#                                    heavy    no     0.5          0.5

# And the last thing we care about is whether or not we make it to our appointment. 
# So did we attend or miss the appointment?
# And ultimately, whether I attend or miss the appointment, it is influenced by track maintenance, because
#it's indirectly this idea that, all right, if there is track maintenance, well, then our train might likely
#be delayed. And if our train is more likely to be delayed, then we are more likely to miss our appointment.
# But what we encode in this bayesian network are just what we might consider to be more direct relationships.
# So the train as a direct influence on the appointment. And given that we know the train is on time or delayed,
#knowing whether there is track maintenance isn't going to give us any additional information that we didn't
#already have. That if we know train, these other nodes that are up above isn't really going to influence the
#result. 
# And so here we might represent it using another conditional probability that looks like this.
# That train can take on two possible values. It's on time or it's delayed.
# And for each of those possible values we have a ditribution for what are the odds that we attend the appointment
#and what are the odds that we miss the appointment.

#            Appointment           T  |  attend  |  missed
#          {attend, missed}   on time     0.9         0.1
#                             delayed     0.6         0.4

# All of these nodes put together represent our Bayesian network, this network of random variables I ultimately
#care about, and that have some sort of relationship between them, some sort of dependence where these arrows
#from one node to another indicate some dependence, that I can calculate the probability of some node given the
#parents that happen to exist there. So now that we've been able to describe the structure of this bayesian network
#and the relationships between each of these nodes by associating each of the nodes in the network with probability
#distribution, whether that's an unconditional probability distribution in the case of our root node, like rain,
#and a conditional probability distribution in the case of all of the other nodes whose probabilities are 
#dependent upon the values of their parents, we can begin to do some computation and calculating using the
#information inside of that table. 

# Computing Joint Probabilities

# So let's imagine, for example, that I just wanted to compute something simple like the probability of light rain.
# How would we get the probability of light rain? Well, light rain, rain is a root node.
# And so if we wanted to calculate that probability, we could just look at the probability distribution for rain
#and extract from it the probability of light rains, just a single value that we already have access to.
# But we could also imagine wanting to compute more complex joint probabilities, like the probability that there 
#is light rain and also and no track maintenance. This is a joint probabiltity of two values, light rain and no track
#maintenace. And the way we might do that is first by starting by saying, all right, let's get the probability of 
#lightt rain. But now we also want the probability of no track maintenance. But of course, this node is dependent
#upon the value of rain. So what we really want is the probability of no track maintenance, given that we know that
#there was light rain. And so the expression for calculating this idea that the probability of light rain and no
#track maintenance is really just the probability of light rain and the probability that there is no track
#maintenance, given that we know that there already is light rain. So we take the unconditional probability of
#light rain, multiply it by the conditional probability of no track maintenance, given that we know there is 
#light rain. And we can continue to do this again and again for every variable that we want to add into this joint 
#probability that we might want to calculate. If we wanted to know the probability of light rain and no track
#maintenance and a delayed train, well, that's going to be the probability of light rain, multiplied by the 
#probability of no track maintenance, given light rain, multiplied by the probability of a delayed train, given 
#light rain and no track maintenance. Because whether the train is on time or delayed is dependent upon both 
#of these variables. And so we have two pieces of evidence that go into the calculation of that conditional
#probability. And each of these three values is just a value that we can look up by looking at one of these
#individual probability distributions that is encoded into our bayesian network.
#           P(light, no, delayed) 
# P(light) P(no|light) P(delayed|light,no)

# And if we wanted a joint probability over all four of the variables, something like the probability of light rain
#and no track maintenance, and a delayed train and we miss our appointment, well, that's going to be multiplying
#four different variables, one from each of these individual nodes. It's going to be the probability of light rain,
#then of no track maintenance given light rain, then of a delayed train, given light rain and no track maintenance.
# And then finally, for this node here, for whether we make it to our appointment or not, it's not dependent upon
#these two variables, given that we know whether or not the train is on time. We only need to care about the 
#conditional probability that we miss our train, or that we miss my appointment, given that the train happens 
#to be delayed. And so that's represented here by four probabilities, each of which is located inside of one of
#these probability distributions for each of the nodes, all multiplied together.  

#            P(light, no, delayed, missed) 
# P(light) P(no|light) P(delayed|ligth, no) P(missed|delayed)

# And so we can take a variable like that and figure out what the joint probability is by mulitplying a whole
#bunch of these individual probabilities from the bayesian network. But of course, just as with last time, 
#where what we really wanted to do was to be able to get new pieces of information, here, too, this is what we're
#going to want to do with our Bayesian network.

# Inference 

#In the context of knowledge, we talked about the problem of inference. Given things that we know to be true, 
#can we draw conclusions, make deductions about other facts about the world that we also know to be true?
# And what we are going to do now is apply the same sort of idea to probability. Using information about which
#we have some knowledge, whether some evidence or some probabilities, can we figure out not other variables
#for certain, but can we figure out the probabilities of other variables taking on particular values?
# And so here, we introduce the problem of inference in a probabilistic setting, in a setting where variables
#might not necessarily be true for sure, but they might be random variables that take on different values 
#with some probability.
# So how do we formulate define what exactly this inference problem actually is? Well, the inference problem 
#has a couple of parts to it.

# Inference -
#           - Query X: variable for which to compute distribution
#           - Evidence Variables E: odserved variables for event e
#           - Hidden Variables Y: non - evidence, non - query variable

#           - Goal: Calulate P(X|e)

# It turns out we can do this calculation using a lot of the probability rules that we've already seen.
# Let's imagine for example that we want to compute the probability distribution of the appointment random
#variable, given some evidence, given that we know there was light rain and no track maintenance.
# P(Appointment|light,no)
# So there's our evidence, the two variables that we observed the values of. We observed the value of 
#light rain, and we observed the value of no track maintenance. And what we care about knowing, our query,
#is the distribution of our random variable appointment. Like what is the chance that we are able to attend
#our appointment, what is the chance we missed our appointment, given our evidence. 
# And the hidden variable, the information that we don't have access to, is the variable train. The information
#that is not part of the evidence that we see, not something that we observe. But it is also not the query that
#we want to know.

# And so, what might this inference procedure look like? 
# If we recall back from when we were defining conditional probabilities, doing math with conditional probabilities.
# We know that conditional probability is proportional to the joint probability. 
# And we remember this by recalling the probability of A given B, is just some constant factor alpha, multiplied
#by the probability of A and B.
# The important thing is its just some constant multiplied by the joint distribution. The probability that all
#of these individual things happened.

# So in this case, we can take the probability of the appointment random variable, given light rain and no
#track maintenance, and say that is just going to be proportional, some constant alpha, multiplied by the joint
#probability, the probability of a particular value for the appointment random variable, and light rain, and no
#track maintenance
# = a P(Appointment|light,no). How do we calculate this?
# We can use marginalization. There are only two ways that we can get any confiruration on an appointment, light
#rain and no track maintenance. Either the particular set of events happens and the train is on time, or, the
#particular set of events happens and the train is delayed. Those are two possible cases that we would want to 
#consider. And if we add those two cases up, well then we get the result, just by adding up all of the possibilities
#for the hidden variable, or variables, if there are multiples. 
# All we have to do is iterate over all possible values for that hidden variable train, and add up there 
#probabilities.

# So this probability expression here = a P(Appointment|light,no), becomes probability distribution over appointment,
#light rain, no track maintenance, train on time and the probability distribution over apponitment, light rain,
#no track maintenance, train delayed.
# = a[P(Appointment|light,no, on time)
#   + P(Appointment|light,no, delayed)]
# So we take both possible values for train and them up.

# The formula for how we do this is a process called inference by numeration, and looks like this.

# P(X|e) = a P(X,e) = a sum over all y P(X,e,y)

# X is the query variable
# e is the evidence
# y ranges over values of hidden variables
# a normalizes the result

# Let's start with what we care about knowing. Which is the probability of X, our query variable, given some
#sort of evidence, some alpha normalizing constant, multiplied by the joint probability X and evidence.
# And how do we calculate that? We are going to marginalize over all of the hidden variables, all of the
#variables that we don't directly observe the values for. We are going to basically iterate over all of 
#the possibilities that it could happen, and sum them all up. 
# We can translate that into a sum over all y, which is all the possible hidden variables and the values they can 
#take on, and adds up all of those possible individual probabilities.
# And that is going to allow us to use process of inference by numeration.

# Here we will be using the pomegranate library
# In python we will create nodes for each of the bayesian nodes we have in the network we created.

# Rain node has no parents
#rain = Node(DiscreteDistribution({
 #   "none": 0.7,
  #  "light": 0.2,
   # "heavy": 0.1
#}), name = "rain")

# Track maintenance node is conditional on rain
#maintenance = Node(ConditionalProbabilityTable([
 #   ["none", "yes", 0.4],
  #  ["none", "no", 0.6],
   # ["light", "yes", 0.2],
    #["light", "no", 0.8],
  #  ["heavy", "yes", 0.1],
   # ["heavy", "no", 0.9]
#], [rain.distribution]), name="maintenance")

# Train node is conditional on rain and maintenance
#train = Node(ConditionalProbabilityTable([
 #   ["none", "yes", "on time", 0.8],
  #  ["none", "yes", "delayed", 0.2],
   # ["none", "no", "on time", 0.9],
   # ["none", "no", "delayed", 0.1],
   # ["light", "yes", "on time", 0.6],
   # ["light", "yes", "delayed", 0.4],
   # ["light", "no", "on time", 0.7],
   # ["light", "no", "delayed", 0.3],
   # ["heavy", "yes", "on time", 0.4],
   # ["heavy", "yes", "delayed", 0.6],
   # ["heavy", "no", "on time", 0.5],
   # ["heavy", "no", "delayed", 0.5],
#], [rain.distribution, maintenance.distribution]), name="train")

# Appointment node is conditional on train
#appointment = Node(ConditionalProbabilityTable([
 #   ["on time", "attend", 0.9],
  #  ["on time", "missed", 0.1],
   # ["delayed", "attend", 0.6],
   # ["delayed", "missed", 0.4]
#], [train.distribution]), name="appointment")

# Create a Bayesian Network and add states
#model = BayesianNetwork()
#model.add_states(rain, maintenance, train, appointment)

# Add edges connecting nodes
#model.add_edge(rain, maintenance)
#model.add_edge(rain, train)
#model.add_edge(maintenance, train)
#model.add_edge(train, appointment)

# Finalize model
#model.bake()

# The key idea here is that someone can design a library for a general Bayesian Network, that has nodes,
#that are based upon its parents, and then all a programmer has to do, using one of these libraries, is
#define what those nodes and probabilities are, and we can begin to do some interesting logic based on that.


# Here we will be talking about Approximate Inference. 
# Approximate Inference is used when we don't know the exact probability, but we have a general sense of the
#probability.
# How can we use approximate inference inside of our bayesian network. One method we can use is called sampling.

# Sampling - In the process of sampling, we are going to take a sample of all the nodes inside our bayesian network.
# What exactly are we going to sample?
# We are going to sample one value from each of our nodes according to their probability distribution.
# So how might we take a sample of a value?
# First we will start with the rain node. Using something like a random number generator we will pick one of
#rains three values. For example let's say that we picked R = none.
# Then we will do the samething for the other variables. When we get to the maintenance variable, we only have
#to randomly pick from its none distribution because we know that information from our rain variable. Now 
#let's say that we randomly picked M = yes.
# Next in our train variable, we will sample from its first distribution because we know that there is no rain,
#and we know that there is track maintenance from our previous information. Let's say that we randomly picked
#T = on time.
# And finally we will do the same steps for our appointment variable. Based on the on-time distribution we 
#learned from our previous information, let's say we randomly picke A = attend
# So by going through these nodes we can very quickly do some sampling and get a sample of the possible values
#that could come up from going through our entire bayesian network. According to those probability distributions

# R =  none, M = yes, T = on time, A = attend

# And what makes this so powerful is, if we do this over and over we can generate a much larger list of samples
#to work with, all using this distribution. 
# We can get a value for each of the possible variables we can come up with.
# This is useful in the case that we are ever faced with a question like, what is the probability that the 
#train is on time? P(Train = on time)?

# Here is a list of all the possible samples we can get from our bayesian network

# R = light     R = light       R = none        R = none
# M = no        M = yes         M = no          M = yes
# T = on time   T = delayed     T = on time     T = on time
# A = missed    A = attend       A = attend      A = attend

# R = none      R = none        R = heavy       R = light
# M = yes       M = yes         M = no          M = no
# T = on time   T = on time     T = delayed     T = on time
# A = attend    A = attend      A = miss        A = attend

# Here we only focus on the samples where the train is on time and ignore the samples where the train is delayed
# So in this case, there are 6 out of the 8 samples that have our train arriving on time.
# In this case we can say something like in 6 out of 8 cases, our train is on time.
# With only 8 samples that might not be such a great example, but if we had thousands of samples, it would
#make for a better inference procedure to be able to these kinds of calculations.
# This is a direct sampling method, to just do a bunch of samples and then fgure out what the probability 
#of some event is.

# Sometimes what we want to figure out is a conditional probability. Something like what is the probability
#that there is light rain, given that the train is on time? P(light|train = on time)
# To do that kind of calculation, what we might do is look at the two cases where the train is delayed, and 
#ignore, or reject them. Exclude them from the possible samples we are considering. 
# And now we want to look at the remaining cases, where the train is on time, and there is light rain.
# Now that leaves us with 2 out of the 6 possible cases that can give us an approximation for the
#probability of light rain, given the fact that we know the train is on time.
# And we do that in almost exactly the same way, just by adding an additional step. That additional
#step is us rejecting all of the samples that doesn't match our evidence,, and only accept the samples
#that do match the evidence that we have. This process is called rejection sampling. This will allow us 
#to figure out a probability not by direct inference, but instead by sampling. 
# It turns out that there a number of other sampling methods that we could use. 
# One problem with rejection sampling is that if the evidence that we are looking for is a fairly unlikely
#event, we are going to be rejecting a lot of samples. For example, if we are looking for the probability
#x given some evidence e, if e is very unlikely to occur, then we will limit ourselves to the samples
#we take in and analize. That would be inefficient because we end up throwing away a lot of samples
#and it takes computational effort to generate those samples, so we would like to avoid that if there
#are better methods to use. 
# There are other sampling methods we can use to address that issue. One is called likelihood weighting.

# Likelihood Weighting, we follow a slightly different procedure, and the goal is avoiding having to throw out
#samples that didn't match the evidence.

# Likelihood Weighting -
#                      - Start by fixing the values for evidence variables
#                      - Sample the non-evidence variables using conditional probabilities in the Bayesian Network
#                      - Weight each sample by its likelihhod: the probability of all of the evidence

# What would this look like?
# If we ask the same question, what is the probabilty of light rain given that the train is on time.
# P(light|train = on time)

# When we start the process we are going to start by fixing the evidence variable. We are already going to
#have in our sample that the train is on time. That way, we don't have to throw out anything, and only sample 
#things that we know, the value of the variables that are our evidence are what we expect them to be.

# R = light / randomly picked
# M = yes / randomly picked
# T = on time / fixed evidence variable
# A = attend / randomly picked

# So now we have generated a sample by fixing an evidence variable and sampling the other three.
# And the last step is now weighting the sample by finding out how much weight it should have.
# And the weight is based on how probable is it that the train is actually on time, given the values of the
#other variables. 
# To do that, we can just go back to the train variable. we see that the likelihood that our train is on time
#plus like rain and track maintenance is 0.6.
# That tells us that this particular sample has a weight of 0.6

# We can perform this sampling process repeatedly. Each time every sample will be given a weight according
#to the probability of the evidence that we see associated with it.
# There are other sampling methods as well, but all of them are just designed to try and get at the same
#idea. To approximate the inference procedure of figuring out the value of a variable.

# Here we will be discussing the topic of uncertainty over time.
# This will explore what we can do to deal with uncertainty over a period of time. We will also be dealing
#with the topic of what we can do when values change over time. 

# This can come up in the context weather. Where we have sunny days and cloudy days. We would like to know 
#things like not just the probability that it's raining now, but what is the probability that it rains
#tomorrow, or the next day, or the day after that etc.

# To do that we are going to produce a slightly different type of model.
# Here we are going to have a random variable. Not just one for the weather, but for every possible time step.
# We can define time step however we like, so we'll use days as our time step for this example.
# So we can define a variable called X sub t, which is going to be the weather at time t.
# Xt: Weather at time t.

# When we are trying to do this inference inside of a computer, when we are trying to reasonably do this
#sort of analysis, it's helpful to make some simplifying assumptions. Some assumptions about the problem that 
#we assume are true. To make our lives a little bit easier. Even if they're not totally accurate assumptions.
# If they're close to accurate, or approximate, they're usually pretty good.
# And the assumption we are going to make is called the markov assumption.

#   Markov Assumption -
#                     - The assumption that the current state depends on only a finite fixed number of 
#previous states.

# So the current days weather depends not on all the previous days weather for all of history, but the 
#current days weather we can predict just based on yesterdays weather. Or just based on the last two
#days weather. How many number of days we specify. But usually, we are just going to deal with just one
#previous state, which helps to predict this current state.

# And by putting a bunch of these random variables together using this markov assumption, we can create 
#what is called a markov chain.

#   Markov Chain -
#                - A sequence of random variables where the distribution of each variable follows the
#markov assumption.

# We'll do an example of this markov assumption where we can predict the weather, is it sunny or
#raining. And we'll just consider those two possibilties for now, even though there are many more
#other types of weather.
# We can predict each days weather, just on the prior days weather. Using todays weather, we can
#come up with a probability distribution for tomorrows weather.
# And here's what that weather might look like. It's formatted in terms of a matrix, with rows and columns
#of values. Where on the left hand side we have todays weather. Represented by the variable X sub t 
#Today (Xt). And in our columns we have tomorrows weateher. Represented by the variable X sub t plus 1
#Tomorrow (Xt+1)

#               Tomorrow (Xt+1)
#             |       | Sunny | Rainy |
#  Today (Xt) | Sunny | 0.8   | 0.2   |
#             | Rainy | 0.3   | 0.7   |

# What this matrix is saying, is that if today is sunny, it's more likely than not that tomorrow will be sunny.
# We have a 0.8 chance of a Sunny day. With a 0.2 probability of rain.
# Like-wise, if today is raininig, than it's more likely that tomorrow will be raining also.
# We have a 0.3 probability of a Sunny day. With a 0.7 probability of another rainy day.

# So this matrix, this description of how it is that we transition from one state to the next state, is what
#we are going to call the transition model. And using the transition model, we can begin to construct this
#markov chain, by just predicting given todays weather, whats the likelihood of tomorrows weather happenning.

# And we can imagine doing a similar sampling procedure, where we take this information, and we sample what
#tomorrows weather is going to be, and using that, we sample the next days weather, and the result of that 
#forms our markov chain.

# Sunny --> Sunny  -->  Rainy -->  Rainy -->  Rainy
#  X0        X1          X2          X3        X4

# The pattern of our markov chain follows, given the distribution that we have access to, our transition model,
#is that when it's sunny, it tends to stay sunny for the next couple of days.
# And when it's raining, it tends to stay raining for the next couple of days.
# And once we have our markov chain, we can perform analysis on it. We can say given that today is raining,
#what is the probability that tomorrow will be raining. Or we can begin to ask probability questions, like
#what is the probability of this sequence of five values of sun, sun, rain, rain, rain, and answer those
#sorts of questions too.

# Here we will go over an example of how we would implement a markov assumption inside our code.

# Define starting probabilities
start = DiscreteDistribution({
    "sun": 0.5,
    "rain": 0.5
})

# Define Transition Model
transitions = ConditionalProbabilityTable([
    ["sun", "sun", 0.8],
    ["sun", "rain", 0.2],
    ["rain", "sun", 0.3],
    ["rain", "rain", 0.7]
], [start])

# Create Markov Chain
model = MarkovChain([start, transitions])

# Sample 50 states from chain
print(model.sample(50))

# This alone defines our Markov Model. We can begin to answer questions using this model.
# We will sample from our Markov Chain using the built in functions from the Markov Chain Library
# This allows us to sample 50 states from the chain, basically just simulating 50 instances of weather.

# When we run this (in live code), it is going to sample from our Markov Chain 50 states. 50 days worth
#of weather, that it's just going to randomly sample. And we can imagine sampling many times to be able
#to get more data, in order to be able to do more analysis.
# Ultimately we get a model that follows the distribution that we originally described. The distribution
#that says sunny days tend to lead to more sunny days, and rainy days tend to lead to more rainy days.
# And this, is a Markov Model
# And Markov Models rely on us knowing the values of these individual states. We know that today is sunny
#or that today is raining, and using that information, we can draw some sort of inference about what
#tomorrow would be like.

# Here we will be looking the idea of Sensor Models. 
# With Sensor Models we are describing how it is we translate what the hidden state is with what the observation
#is, what it is the AI knows, or what it is that the AI has access to.

# For example, a hidden state might be a robots position. If a robot is exploring new and uncharted territory,
#the robot likely doesn't know exactly where it is, but it does have an observation, it has robot sensor data,
#where it can sense how far away are possible obstacles around it, and using that information, using the 
#observed information that it has, it can infere something about the hidden state. Because what the true
#hidden state is, influences those observations.
# Whatever robots true position is effects the sensor data that the robot is able to collect. Even if the 
#robot doesn't know for certain what its true position is.

# Like-wise if we think about voice recognition or speech recognition programs that listen to us and is able
#to respond to us, we might imagine that the hidden state, the underlying state is what words are actually spoken,
#the true nature of the world contains us saying a particular sequence of words, but our phone, or smart device
#doesn't know for sure exactly what words we said, the only observation that the AI has access to is some 
#audio wave form. And those audio wave forms are of course dependent on this hidden state, and we can infere
#based on those audio wave forms what words were likely spoken. But we might not know with 100 percent 
#accuracy what that hidden state actually is.
# We are tasked with trying to predict, given our observation, given these audio wave forms, can we figure
#out what the actual words spoken were.
# Like-wise we might imagine on a website, true user engagement might be information we don't directly have
#access to, but we can observe data, like website or app analytics that tell us how often a button was pushed,
#or how often people are interacting with the page in a particular way. And we can use that to infere things
#about our users as well.

# This is our Sensor Model

#     Hidden State   ||   Observation
#-------------------------------------------
#   robot's position ||  robots sensor data
#     words spoken   ||  audio waveforms
#    user enagement  ||  website or app analytics

# This kind of problem comes up all the time when dealing with AI and trying to infere things about the world.
# Often AI doesn't really know the hidden true state of the world. All the AI has access to is some observation
#that is related to the hidden state and true state, but it's not direct. It might be some noise there. The
#audio waveform might have some additional noise that may be difficult to parse. Sensor data might not be 
#exactly correct, there's some noise that might not allow us to conclude with certainty what the hidden state
#is. But can allow us to infere what it might be.

# A simple example we'll take a look at is, imagining that the hidden state is weather, whether it's sunny
#or rainy. And imagine that we are programming an AI inside of a building that has access to a camera inside 
#the building, and all we have access to is an observation, as to whether or not the employees are bringing
#an umbrella into the building or not. And using that information, we can predict whether it's sunny or not.
# Even if we don't know what the underlying weather is.

# Basically, anytime we observe something, it's based on some underlying hidden state.

# This is our Sensor Model

#     Hidden State   ||   Observation
#-------------------------------------------
#   robot's position ||  robots sensor data
#     words spoken   ||  audio waveforms
#    user enagement  ||  website or app analytics
#     weather        ||  umbrella
