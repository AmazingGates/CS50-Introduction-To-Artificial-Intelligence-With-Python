# Here we are going to discuss a Hidden Markov Model

#  Hidden Markov Model -
#                      - A Markov model for a system with hidden states that generate some observed event.

# So in addition to our transition model, that we still need, that says given the uderlying state of the world,
#if it's sunny or rainy, what's the probability of tomorrows weather, we also need another model, that given
#some state, is going to give us an observation, of something like green yes, someone brings an umbrella in
#the office, or red no, nobody brings umbrellas into the office.
# So the observation is going to be that if it is sunny, chances are that nobody brought umbrellas into the office.
# And if it's raining, then there's a much higher probability that people bring umbrellas into the office.

# So using the observation, we can begin to prredict with reasonably likeihood what the underlying state is.
# Even if don't actually get to observe the underlying state. If we don't get to see what the hidden state is.

# This is what we often call a Sensor Model. It is also sometimes called an emittion probability, because the 
#underlying state emits some sort of emittion that we can observe.


#                             Observation(Et)
#-------------------------------------------------------------
#                    || Umbrella ||  No Umbrella ||
#           | Sunny  ||    0.2   ||     0.8      ||
# State(Xt)---------------------------------------------------
#           | Rainy  ||    0.9   ||     0.1      ||
#-------------------------------------------------------------


# Senor Markov Assumption -
#                         - The assumption that the evidence variable depends only on the corresponding state.

# Meaning it can predict whether or not people will bring umbrellas or not, entirely depending on whether or not
#it is sunny or rainy today.

# So what our Hidden Markov Model ends up looking like is this

#   Hidden Markov Model
# X0 --> X1 -->  X2 -->  X3 -->  X4 -- This top line represents our underlying state of the world
# Sun    Sun    Rain    Rain    Rain
#  |      |       |       |       | # Each of these states produces an emittion, an observation that we see
#  |      |       |       |       |
# No --> Yes --> Yes --> Yes --> Yes -- This bottom line represents our emittions
# E0      E1      E2      E3      E4

# So this two is way we can try to represent this idea. 
# What we want to think about is that the underlying state is the true nature of the world.
# The robots position as it moves over time. And that produces some sort of senor data that might be observable,
#or what people are actually saying. And using the emittion data of what audio waveform we detect in order to
#process that data ad try and figure it out.

# And there a number of possible task that we might want to do, given this kind of information.
# And one of the simplest is trying to infere something about the future or past, or about these sort of 
#hidden states that might exist.

# So the task we'll often see are all based on the same idea of conditional probabilities and using the
#probability distributions we have to draw these sorts of conclusions. 

# One task is called filtering, which is given observations from the start until now, calulate distribution 
#for a current state. Meaning, given information about the beginning of time until now, on which days did
#people bring an umbrella or not bring an umbrella, can we calculate the probability of the current state
#of the day. Is it sunny or is it raining.

# Another task that may be possibleis predictions. Which is given observations from the start until now, 
#can we calulate distribution for tomorrow. Is it sunny or is it rainy.

# And we can also go backwards as well, by a smoothing. Which is given observations from the start until now,
#can we calulate distribution for some past state. Like we know that today people brought umbrellas, and 
#tomorrow people will bring umbrellas. And so given two days worth of data of people bringing umbrellas,
#What's the probability that yesterday it rained. And that we know that people brought umbrellas today,
#that might inform that decision as well, and likely influence these probabilities.

# And there's also a most likely explanation task, in addition to other task that might exist as well,
#which is a combination of some these task.
# Which is given observations from the start until now, figuring out the most likely sequence of states.
# So this is what we are going to take a look at now. This idea that if we have these obervations, umbrella,
#no umbrella, can we calculate the most likely states of sun or rain, that actually represented the true
#weather that would produce these observations.



#   Task     ||                              Definition                                              ||
#------------------------------------------------------------------------------------------------------
# Filtering  || given observations from the start until now, calulate distribution for a current state.
# Prediction || given observations from the start until now, calulate distribution for a future state.
# Smoothing  || given observations from the start until now, calulate distribution for a past state.
# Most Likely Explanantion || given observations from the start until now, calulate most likely sequence of states.

# This is common when we are trying to do somehing like voice recognition for example. 
# That we have these emittions in audio waveform, and we would like to calculate based on all of the 
#observations we have, what is the most likely sequence of actually words or syllables or sounds that the
#user actually made when they were speaking to this particular device. 
# Or other task that might come up in that context as well.

# Here, we will write out the code that we might try out in real time, using  a Markov Model in Python.

# Observation Model For each state.
sun = DiscreteDisribution({
    "umbrella": 0.2,
    "no umbrella": 0.8
})

rain = DiscreteDisribution({
    "umbrella": 0.9,
    "no umbrella": 0.1
})

states = [sun, rain]

# Transition Model
transtion = numpy.array([
    [0.8, 0.2], # Tomorrow's prediction if today = sun
    [0.3, 0.7]  # Tomorrow's prediction if today = rain
])

# Starting Probabilities
starts = numpy.array([0.5, 0.5])

# Create The Model
model = HiddenMarkovModel.from_matrix(
    transitions, states, starts,
    state_names = ["sun", "rain"]
)

model.bake()

# Once we have our model, we can begin to do some inference (Usually in a separate file).

# Observed Data
observations = [
    "umbrella", 
    "umbrella",
    "no umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "umbrella",
    "no umbrella",
    "no umbrella"
]

# Predict underlying states
predictions = model.predict(observations)
for prediction in predictions:
    print(model.states[prediction].name)

# If we ran this prediction model in real time Python, we would most likely see this output
rain
rain
sun
rain
rain
rain
rain
sun
sun


# Now we have seen a couple of ways that AI can begin to deal with uncertainty. We've taken a look at
#probability, and how we can use probability to describe numerically, things that are likely, or more likely,
#or less likely to happen in other events or other variables, and using that information we can begin to
#construct the standard types of models. Things like Bayesian Networks, and Markov Chains, and Hidden Markov
#Models, that all allow us to be able to describe how particular events relate to other events, or how the
#values of particular variables relate to other variables. Not for certain, but with some sort of
#probability distribution.

# And by formulating things in terms of these models that already exist, we can take advantage of Python
#libraries that implement these sort of models already, and allow us to be able to use them to produce
#some sort of resulting effect.

# All of this then allows our AI to deal with these sort of uncertain problems, so that our AI doesn't
#need to know things for certain, but can infere, based on information it doesn't know.