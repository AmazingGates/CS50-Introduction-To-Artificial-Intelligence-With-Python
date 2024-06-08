# Uncertainty - This is where we talk about the ideas of what happens if a computer isn't sure about a fact, 
#but maybe is only sure with a certain probability. We will discuss some of th ideas behind probability, and 
#how computers can begin to deal with uncertain events in order to be a little bit more intelligent in that 
#sense as well.

# Here we will look at the idea of Probability theory.

# Probability - The idea that there are possible worlds. And the idea of a possible world is that when we
#roll a die, there are six possible worlds that could result from it. And each of those possible worlds has
#some probability of being true. And we represent that probability like this, using the capital letter P and
#then in parentheses (w), what it is that we want the probability of (w). So this right here would be the probability
#of some possible world as represented by the little letter omega (w).

# Now, there are a couple of basic axioms of probability that become relevant as we consider how we deal with
#probability and how we think about it. 

# First and foremost, every probability value must range between 0 and 1 inclusive. So the smallest value any
#probability can have is the number 0, which is an impossible event. And on the other end of the spectrum,
#probability can range all the way up to the positive number 1, meaning an event is certain to happen.

# And then they can range through any real number in between these two values. We're generally speaking,
#a higher value for the probability means an event is more likely to take place, and a lower value for the
#probability means the event is less likely to take place.

# And the other key rule for probability looks a little like this,
# Sigma - refers to summation, the idea that we're going to be adding up a whole sequence of values.
# What this notation means, is that if we sum up all the possible worlds omega that are big omega, which
#represents the set of all the possible worlds, meaning we take for all the worlds in the set of possible
#worlds and add up all of their probabilities, what we ultimately get is number 1. Sigma omega big-omega P(w) = 1
# So if we take all the possible worlds, add up what each of their probabilities is, we should get the number
#1 at the end, meaning all probabilities just need to sum 1.
# So for example, if we take dice, and imagine we have a fair die with numbers 1 through 6 and we roll the die,
#each one of these rolls has an equal probability of taking place. And the probability is 1 over 6, for example.
# S each of the probabilities is between 0 and 1, 0 meaning impossible and 1 meaning for certain. And if we add
#up for all of the possible worlds, we get the number 1.
# And we can represent any one of those possibilities like this. P(die)= 1 over 6
# The probability that we roll the number 2, for example, is just 1 over 6.
# Every six times we roll the die, we'd expect that one time, for instance, the die might come up as a 2.
# Its probability is not certain, but it's a little more than nothing, for instance.


#           Possible Worlds
#--------------------------------
#               P(w) = Some probability of possible worlds
#           0 < P(w) < 1 
#          Sigma omega big-omega P(w) = 1
#               P(die)= 1 over 6


# Things get more interesting as our models of the world get a little more complex. 
# Let's imagine now that we're not just dealing with a single die, but we have two dice, for example.
# We have a red die and a blue die, and we care not just about what the individual roll is, but we care 
#about the sum of the two rolls. And the sum of the two rolls is the nnumber 3.
# How do we go about finding out what the probability will look like, if instead of having one die, we now have two
#dice?
# What we might imagine is that we could first consider what are all of the possible worlds. And in this case,
#all of the possible worlds are just every combination of the red and blue die that we could come up with.
# For the red die, it could be a 1, or 2, or 3, ... up until 6. And for each of those possibilities, the blue die,
#likewise, could also be either 1, or 2, or 3, ... up until 6. And it just so happens that in this particular case,
#each of possible combinations is equally likely. Equally likely are all of these possible worlds. That's not
#always going to be the case though.
# If we imagine more complex models that we could try to build and things that we could try to represent in the real
#world, it's probably not going to be the case that every single possible world is always equally likely.
# But since we are using dice, we can consider all of these possible worlds to be equally likely. but even though
#all of the possible worlds are equally likely, that doesn't necessarily mean that their sums are equally likely.
# So if we consider what the sum is of all of these two, so 1 plus 1, is a 2. 2 plus 1 is a 3. And consider for
#each of these possible pairs of numbers what their sum ultimately is, we can notice that there are some patterns
#here, where it's not entirely the case that every number comes up equally likely.
# If we consider 7, for example, what's the probability that when we roll two dice, their sum is 7?
# There are several ways this can happen. There six possible worlds where the sum is 7. It could be a 
#1 and a 6, or a 2 and a 5, or a 3 and a 4, or a 4 and a 3, and so forth. 
# But if we instead consider what's the probability that we roll two dice, and the sum of those two dice is 12,
#for example, we notice that there's only one possible world in which that can happen. And that's the possible
#world where both dice are six. 
# That tells us that the probability that the sum is 7 is greater than the probability that the sum is 12.
# And we can represent that even more formally by saying the probability that we sum to 12 is 1 out of 36
#P(sum to 12)= 1 over 36.
# Out of the 36 equally likely possible worlds, six squared because we have six options for the red die and 
#six options for the blue die, out of those 36 options, only one of them sums to 12.
# On the other hand, the probability that if we take two dice rolls and they sum to the number 7, out of those 
#36 possible worlds, there were six worlds where the sum was 7. And so we get 6 over 36, which we can simplify
#as a fraction to just 1 over 6.
#P(sum to 7)= 6 over 36 = 1 over 6
# These are known as unconditional probabilities. Some degree of belief in a proposition, some fact about the world 
#in the absence of any other evidence.

# Unconditional Probability -
#                           - Degree of belief in a proposition in the absence of any other evidence.

# Usually when we're thinking about probability, especially when we're training AI to intelligently be able 
#to know something about the world and make predictions based on that information, it's not unconditional
#probability that our AI is dealing with, but rather conditional probability, probability where rather than
#having no original knowledge, we have some initial knowledge about the world and how the world actually
#works.

# Condiditional Probability -
#                         - Degree of belief in a proposition given some evidence that has already been 
#revealed.

# What does this look like? It looks like this in terms of notation. P(a|b)
# We're going to represent conditional probability as a probability of A vertical bar B.
# The way to read this is the thing on the left-hand side of the vertical bar is what we want the probability 
#of. We want the probability that A is true, that it is a real world, that it is the event that actually
#does take place. And then on the right-hand side of the vertical bar is our evidence, the information
#that we already know for certain about the world. For example, that B is true. 
# So the proper way to read P(a|b) is, what is the probability of A given B, the probability that A is true,
#given that we already know that B is true. And this type of judgement, conditional probability, the probability
#of one thing given some fact, comes up quite a lot when we think about the types of calculations we might want
#our AI to be able to do. 
# For example, we might care about the probability of rain today given that we know that it rained yesterday.
# P(rain today | rain yesterday) 
# Another example P(route change | traffic conditions)
# Another example P(disease | test results)
# This notion of conditional probability comes up everywhere. So we begin to think about what we would like
#to reason about, and begin to reason more intelligently by taking into account evidence that we already have.

# Now that we have this idea of what conditional probability is, the next question we have to ask is, how
#do we calculate conditional probability? 
# How do we figure out mathematically, if we have an expression like this, how do we get a number from that?
# What does conditional probability actually mean?
# Well the formula for conditional probability looks like this. P(a|b)=P(a and b) over P(b).
# Translated = The probability of a given b, the probability that a is true, given that we know b is true,
#is equal to this fraction, the probability that a and b are true, divided by just the probability that just 
#b is true.
# And the way to intuitively try to think about this is that if we want to know the probabbility that a is
#true, given that b is true, we want to consider all the ways they could both be true out of the only worlds
#where b is already true. We can ignore all the cases where b isn't true, because those aren't relevant to my
#ultimate computation. They're not relevant to what it is that we want to get information about. 
# Let's take a look at an example.
# Let's go back to the example of rolling two dice and the idea that those two dice might sum up to the number 12.
# Let's say that we want to know what is the probability that the two dice sum to 12, given that we know that the
#red die was a 6. We already have some evidence, we know that the red die is a 6. We don't kknow what the blue die
#is. That information isn't given to us in this expression. But given the fact that we know that the red die is a 6, 
#what is the probability that we sum to 12? 
# We can begin to do the math using that expression from before. P(red)=1 over 6
# So now, in addition to the fact that the red die rolled as a 6 and the probability of that, the other piece
#of information we need to know in order to calculate this conditional probability is the probability that both
#of our variables, A and B, are true. The probability that both the red die is a 6, and they all sum to 12.
# So what is the probability that both of these things happen? 
# Well, it only happens in one possible case in 1 out of these 36 cases, and it's the case where both the red
#and the blue die are equal to 6. This is a piece of information that we already knew.
# And so this probability is equal to 1 over 6. P(sum 12 and red die)=1 over 36
# To get the conditional probability that the sum is 12, given that we know that the red die is equal to 6,
#well, we just divide these two values together, and 1 over 36 divided by 1 over 6 gives us this probability
#of 1 over 6. P(sum 12 and red die)=1 over 36 divided by P(red die)= 1 over 6 equals to P(sum 12|red die)=1 over 6.
# The probability that the sum of the two dice is 12 is aslo 1 over 6.
# So in this case, the conditional probability seems fairly straightforward. But this idea of calculating a 
#conditional probability by looking at the probability that both of these events take place is an idea that's
#going to come up again and again. This is the definition of conditional probability. And we're going to use
#that definition as we think about probability more generally to be able to draw conclusions about the world.
# This is that formula. P(a|b) = P(a and b) over P(b)
# The probability of A given B is equal to the probability that A and B take place divided by the probability of B.
# We might also see this formula written in different ways. P(a and b) = P(b)P(a|b). 
# Translated - The probability of A and B is equal to the probability of B times the probability of A given B.

# Here we will talk about random variable.

# Random Variable -
#                 - A variable in probability theory with a domain of possible values it can take on.

# What this means is, we might have a random variable that is just called roll, for example, that has 
#six possible values. 

# Random Variable -
#                   Roll
#               {1,2,3,4,5,6}

# Roll is my variable, and the possible values, the domain of values that it can take on are 1,2,3,4,5, and 6.
# And we might like to know the probability of each. In this case, they happen to all be the same. But in other
#random variables, that might not be the caes. For example, I might have a random variable to represent the 
#weather, where the domain of values it could take on are things like sun or cloudy or rainy or windy or 
#snowny. 

# Random Variable - 
#                    Weather
#           {sun, cloud, rain, wind, snow}

# Each of these might have a different probability. And we care about knowing what is the probability that
#the weather equals sun or that the weather equals clouds, for instance. And we might like to do some
#mathematical calculations based on that information.

# Other random variables might be something like traffic. 

# Random Variable -
#                 - Traffic
#              {none, light, heavy}

# What are the odds that there is no traffic or light traffic or heavy traffic?
# Traffic in this case is our random variable. And the values that that random variable can take on are here.
# It's either none, or light, or heavy.

# Here is one more example. This is flight. And often we want to kknow something about the probability 
#that our random variable takes on each of those possible values.

# Random Variable - 
#                 - Flight
#         {on time, delayed, cancelled}

# And this is what we call a probability distribution. A probability distribution takes a random variable
#and gives us the probability for each of the possible values in its domain.
# So in case of fligth, for example, our probability distribution might look somethong like this -

# Probability Distribution -
#                          - P(Flight=on time)=0.6
#                          - P(Flight=delayed)=0.3
#                          - P(Flight=canceled)=0.1

# Our probability distribution says the probability that the random variable flight is equal to the value on time
#is 0.6. Or otherwise, the likelyhood that our Flight is on time is 60 percent, for exampe.
# And in this case, the probability that our flight is delayed is 30 percent.
# The probability that our fight is canceled is 10 percent.
# And if we sum up all of these possible values, the sum is going to be 1.
# If we take all of the possible worlds, add them all up together, the result needs to be the number 1, per
#that axiom of probability theory that we've discussed before.
# This is one way of representing this probability distribution for the random variable flight.
# Sometimes we'll see it represented a little bit more concisely that this is pretty verbose for really
#just trying to express three possible values.
# So often, we'll instead see the same notation representing using a vector. All a vector is is a
#sequence of values. As opposed to just a single value, we might have multiple values. And so we can extend
#instead, represent this idea this way. 

# Probability Distribution -
#                          - P(Flight)=<0.6,0.3,0.1>
# Large P, generally meaning the probability distribution of this variable flight is equal to this vector
#represented in angle brackets. The probability distribution is 0.6, 0.3, and 0.1.
# And we just have to know that this probability distribution is in order of on time or delayed and canceled
#to know how to interpet this vector. To mean the first value in the vector is the probability that our flight 
#is on time. The second value in the vector is the probability that our flight is delayed. And the third value
#in the vector is the probability that our flight is canceled.
# This is just an alternate way of representing the flight formula more verbosely. 
# But oftentimes, we'll see that we just talk about a probability distribution over a random variable.
# And whenever we talk about that, what we're really doing is trying to figure out the probabilities of 
#each of the possible values that that random variable can take on. But this notation is just a little bit more
#succinct, even though it can sometimes be a little confusing, depending on the context in which we see it.
# So we'll start to look at examples where we use this sort of notation to describe events that might take place.

# A couple of other important notes to remember about probbility theory.
# One is this idea of independence.

# Independence - The knowledge that one event occurs does not affect the probability of the other event.
# For example, in the context of our two dice rolls, where we had the red die, the probability that we
#roll the red die and the blue die, those two events are independent of each other. Knowing the result
#of the red die doesn't change the probabilities for the blue die. It doesn't give any additional information
#about what the value of the blue die is ultimately going to be.

# So basically, independence refers to the idea that one event doesn't influence the other.
# And if they're not independent, then there might be some relationship.
# So mathematically, formally, what does independence actually mean?
# Recall this formula from before, that the probability of A and B is the probability of A times the probability
#of B given A. P(a and b)=P(a)P(b|a). The more intuitive way to think about this is that to know how likely
#it is that A and B happen, we'll first figure out the likelyhood that A happens. And then given that we know
#that A happens, let's figure out the likelyhood that B happens and multiply those two things together. 
# But if A and B were independent, meaning knowing A doesn't change anything about the likelyhood that B
#is true, well, then the probability of B given A, meaning the probability that B is true, given that we 
#know that A is true, well, that we know A is true shouldn't really make a difference if these two things
#are independent, that A shouldn't influence B at all. So the probability of B given A is really just the 
#probability of B. If it is true that A and B are independent. And this right here is one example of a 
#definition for what it means for A and B to be independent. 
# P(a and b)=P(a)P(b). The probability of A and B equals the probability of A times the probability of B.
# Anytime we find two events A and B where this relationship holds, then we can say that A and B are 
#independent. 


# Here we will be looking at Bayes' Rule. This is a very important rule when it comes to probability theory.
# Let's take a look at a previous equation to be able to derive Bayes' rule ourselves.
# P(a and b) = P(b) P(a|b) is the same as P(a and b) = P(a) P(b|a). This is a sort of symmetric 
#relationship where it doesn't matter the order, A and B and B and A mean the same thing. And so in these
#equations, we can just swap out A and B to be able to represent the same idea.
# So we know that those two equations are already true. That allows us to do a little bit of algebraic
#manipulation. Both of the expressions on the rigth-hand side are equal to the probability of A and B.
# So what we can do is take those two expressions on the right-hand side and just set them equal to each other.
# P(a) P(b|a) = P(b) P(a|b). If they're both equal to the probability of A and B, then they both must be equal
#to each other. So the probability of A times probability of B given A is equal to the probability B times
#the probability of A given B. Now we can do some division. We can divide both sides by P of A. That gives us
#Bayes' Rule.
# P(b|a) = P(b) P(a|b) divided by P(a). 
# Sometimes in Bayes' Rule, we'll see the order of those two arguments switched. So instead of B times A given B,
#it'll be A given B times B. That ultimately doesn't matter because in multiplication, we can switch the order
#of the two things we're multiplying, and it doesn't change the result. 
# But the first version is the most common formulation of Bayes' Rule. The probability of B given A is equal
#to the probability of B times the probability of A given B divided by the probability of A. 
# This rule is important when it comes to trying to infer things about the world, because it means we can 
#express one conditional probability, the conditional probability of B given A, using knowledge about the
#probability of A given B, using the reverse of that conditional probability. 

# Now we go through an example of this just to see how we might use it, and then explore what this means a little
#more generally. 
# We will construct a situation where we have some information. 
# There are two events that we care about, the idea that it is cloudy in the morning and the idea that it is
#rainy in the afternoon. 
# Those are two different possible events that could take place, cloudy in the morning, rainy in the evening.
# And what we care about is, given clouds in the morning, what is the probability of rain in the afternoon?
# We can use data to try to figure this out.
# So let's imagine that we have some access to some pieces of information. 
# We have access to the idea that 80 percent of rainy afternoons start out with a cloudy morning.
# We also know that 40 percent of days have cloudy mornings.
# And we also know that 10 percent of days have rainy afternoons.
# Now using this information, we would like to figure out, given clouds in the morning, what is the probability
#that it rains in the afternoon.
# We want to know the probability of afternoon rain given morning clouds.
# We can do that using this fact. If we know that 80 percent of rainy afternoons start with cloudy mornings,
#then we know the probability of cloudy mornings given rainy afternoons.
# So using sort of the reverse conditional probability, we can figure that out.
# Expressed in terms of Bayes' Rule, this is what that would look like.

# Bayes' Rule Version
# P(rain|clouds) = P(clouds|rain)P(rain) divided by P(clouds).
# Now we can just do the math. We have that information.

# We know that 80 percent of the time, if it was raining, then there were clouds in the morning. S0 (.8).
# Probability of rain is (.1), because 10 percent of days were rainy, and (.4) because 40 percent of days
#were cloudy. 
# P(rain|clouds) = (.8)(.1) divided by (.4)
# When we do the math, we can figure out the answer is (.2). 
# So the probability that it rains in the afternoon, given that it was cloudy in the morning, is (.2) in 
#this case.
# This is an application of Bayes' Rule, the idea that using one conditional probability, we can get the
#reverse probability. 
# This is often useful when one of the conditional probabilities might be easier for us to have data about.
# And using that information, we acn calculate the other conditional probability. So what does this look like?
# Well, it means that knowing the probability of cloudy mornings given rainy afternoons, we can calculate
#the probability of rainy afternoons given cloudy mornings.
# Use P(cloudy morning|rainy afternoon) to calculate P(rainy afternoon|cloudy morning)
# Or, for example, more generally, if we know the probability of some visible effect, some effect that we can
#see and observe, given some unknown cause that we're not sure about, well, then we can calculate the 
#probability of that unknown cause given the visible effect. 
# Use P(visible effect|unknown cause ) to calculate P(unknown cause|visible effect).
# So what might that look like?
# Well, in the context of medicine, for example, we might know the probability of some medical test result 
#given a disease. Like, we know that if someone has a disease, then x percent of the time the medical test 
#result will show up as this, for instance. And using that information, then we can calculate, all right,
#what is the probability that given we know the medical test result, what is the likelyhood that someone
#has the disease?
# Use P(medical test result|disease) to calculate P(disease|medical test result).
# This is the piece of information that is usaully easier to know P(medical test result|disease), easier
#to immediately have access to data for.
# And this is the information that we actually want to calculate. P(disease|medical test result).
# Another example, we might want to know, for example, if we know that some probability of counterfeit bills
#have a blurry text around the edges, because counterfeit printers aren't nearly as good at printing text 
#precisely. So we have some information about, given that something is a counterfeit bill, like x percent
#of counterfeit bills have blurry text, for example. And using that information, then we can calculate some
#piece of information that we might want to know, like, given that we know there's blurry text on a bill,
#what is the probability that that bil is counterfeit?
# Use P(blurry text|counterfeit bill) to calculate P(counterfeit bill|blurry text).
# So given one conditional probability, we can calculate the other conditional probability as well.

# Now we have taken a look at a couple of different types of probability. We've looked at unconditional
#probability, where we just look at what is the probability of this event occuring, given no additional evidence
#that we might have access to. And we've also looked at conditional probability, where we have some sort of
#evidence, and we would like to, using that evidence, be able to calculate some other probability as well.

# Another type of probability we are going to look at is joint probability. This is when we are considering
#the likelyhood of multiple different events simultaneously.
# For example we might have probability distributions that look like this. Like if we want to know the 
#probability distribution of clouds in the morning. 40 percent of the time, C, which is our random variable here,
#is equal to it's cloudy. And 60 percent of the time, it's not cloudy. This is just a simple probability 
#distribution that is effectively telling us that 40 percent of the time, it's cloudy. 

#         AM
# C = cloud  C = -cloud
#   0.4        0.6

# We might also have a probability distribution for rain in the afternoon, where 10 percent of the time,
#it is raining in the afternoon. And with probability 0.9, or ninety percent, it is not raining in the afternoon.

#          PM
# R = rain  R = -rain
#    0.1      0.9

# And using just these two pieces of information, we don't actually have a lot of information about how
#these two variables relate to each other. 
# But we could if we had access to their joint probability, meaning for every combination of these two things, 
#meaning morning cloudy and afternoon rain, morning cloudy and afternoon not rain, morning not cloudy and 
#afternoon rain, and morning not cloudy and afternoon not raining, if we had access to values for each of those four,
#we'd have more information.
# Information that'd be organized in a table like this, and this, rather than just a probability distribution,
#is a joint probability distribution.
# It tells us the probability distribution of each of the possible combinations of values that these random
#variables can take on.

#               AM  PM
#          R = rain  R = -rain
#C = cloud   0.08       0.32
#C = -cloud   0.02       0.58

# So if we want to know what is the probability that on any given day it is both cloudy and rainy, well, we
#would say, all right, we look cases where it is cloudy and cases where it is raining.
# And the intersection of those two, that row in that column, is 0.08. So that is the probability that it is
#both cloudy and rainy using that information. 
# And using this joint probability table, we can begin to draw other pieces of information about things like
#conditional probability. So we might ask questions like, what is the probability distribution of clouds
#given that we know that it is raining? P(C|rain)
# Meaning we know for sure that it is raining. Tell us the probability distribution over whether it's cloudy or 
#not, given that we know already that it is, in fact, raining. And we use C to stand for that random variable.
# We are looking for distribution, meaning the answer to this is not going to be a single value. It's going
#to be two values, a vector of two values, where the first value is probability of clouds, the second value
#is probability that it is not cloudy, but the sum of those values is going to be 1.
# Because when we add up the probabilities of all of the possible worlds, the result that we get must be
#the number 1. 
# And what do we know about how to calculate a conditional probability?
# Well, we know that the probability of A given B is the probability of A and B divided by the probability 
#of B. So what does this mean?
# It means we can calculate the probability of clouds given that it is raining as the probability of clouds
#and raining divided by the probability of rain.

# P(C|rain) = P(C, rain) divided by P(rain). And the comma for probability distribution stands in for the
#word and. We'll see both used in different situations.
# Instead of expressing this as this joint probability divided by the probability of rain, sometimes we'll
#just represent it as alpha times the numerator, the probability of C, the variable, and that we know that
#it is raining, for instance. 

# P(C|rain) = P(C, rain) divided by P(rain) = aP(C|rain)
# So all we've done here is said this value of 1 over the probability of rain, that's just a constant we're
#going to divide by or equivalently multipl by the inverse of at the end. We'll just call it alpha for now
#and deal with it a little bit later.
# But the key idea here now, and this is an idea that's going to come up again, is that the conditional 
#distribution C given rain is proportional to, meaning just some factor multiplied by the joint probability
#of C and rain being true.
# How do we figure this out?
# Well, this is going to be the probability that it is cloudy given that it's raining, which is 0.08,
#and the probability that it's not cloudy given that it's raining, which is 0.02.
# And s we get alpha times here now is that probability of distribution. 0.08 is clouds and rain.
# 0.02 is not cloudy and rain. =a<0.08, 0.02>
# But of course, 0.08 and 0.02 don't sum up to the number 1. And we know that in a probability distribution,
#if we consider all of the possible values, they must sum up to a probability of 1.
# And so we know that we need to figure out some constant to normalize, so to speak, these values, something
#we can multiply or divide by to get it so that all these probabilities sum up to 1, and it turns out that 
#if we multiply both numbers by 10, then we can the result of 0.8 and 0.2. The proportions are still equivalent,
#but now 0.8 and 0.2, those sum up to the number 1.

# Here we will be going a couple important probability rules.
# One of the simplest rules is just this negation rule.

#   Negation - 
#            - What is the probability of not event A? P(not a) = 1 - P(a)
# So A is an event that has some probability, and we would like to know what is the probability that A
#does not occur. 
# And it turns out it's just 1 minus P of A, which makes sense. Because if those are the two possible
#cases, either A happens or A doesn't happen, then when we add up those two cases, we must get 1, which
#means that P of not A just be 1 minus P of A. Because P of A and P of not A must sum up to the number 1.
# They must include all of the possible cases. P(not a) = 1 - P(a)

# Here we will look at the Inclusion-Exclusion

#   Inclusion-Exclusion-
#                      - Used for The actual expression for calculating the probability of A or B. We take 
#the probability of A, add it to the probability of B. Then we need to exclude the cases that we've double
#counted. So we subtract from that the probability of A and B. And that gets us the result for A and B.
# We consider all the cases where A is true and all the cases where B is true. And if we imagine this is like
#a Venn diagram of cases where A is true, cases where B is true, we just need to subtract out the middle to
#get rid of the cases that we have overcounted by double counting them inside both of these expressions.
# P(a or b) = P(a) + P(b) - P(a and b)

# One other rule that is going to helpful is a rule called marginalization.

#   Marginalization -
#                   - Marginalization is answering the question of how do we figure out the probability of A using
#some other variable that we have access to, like B? Even if we don't know information about it, we know that B,
#some event, can have two possible states, either B happens or B doesn't happen, assuming it's a boolean, true or 
#false. And what that means is that for us to be able to calculate the probability of A, there are only two cases.
# Either A happens and B happens, or A happens and B doesn't happen. These are two disjoint, meaning they can't 
#both happen together. Either B happens or B doesn't happen. They are separate cases.
# And so we can figure out the probability of A just by adding up those two cases. The probability that A is true
#is the probability that A and B is true, plus the probability that A is true and B isn't true. So by marginalizing,
# we've looked at the two possible cases that might take place, either B happens or B doesn't happen. And in either
#of those two cases, we look at what's the probability that A happens. And if we add those together, then we get 
#the probability that A happens as a whole. P(a) = P(a and b) + P(a and not b)

# Lastly we will be looking at something called conditioning.

#   Conditioning -
#                - Conditioning says that if we have two events, a and b, but instead of having access to their joint
#probabilities, we have access to their conditional probabilities, how they relate to each other. Again, if we want
#to know the probability that a happens, and we know that there's some other variable b, either b happens or
#b doesn't happen, and so we can say that the probability of a is the probability of a given b times the probability
#of b, meaning b happened. And given that we know b happened, what's the likelihood that a happened? And then we consider
#the other case, that b didn't happen. And just as in the case of marginalization, where there was an equivalent rule for
#random variables that could take on multiple possible values in a domain of possible values, here, too, conditioning
#has the same equivalent rule. 
# P(a) = P(a|b) P(b) + P(a|not b) P(not b)