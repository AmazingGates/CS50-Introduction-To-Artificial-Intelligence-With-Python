# Knowledge - Ideally, we want our AI to be able to know information, to be able to represent that information, and 
#more importantly, to be able to draw inferences from that information, to be able to use the information it knows 
#and draw additional conclusions. 

# Knowledge-Based-Agents - Agents that reason by operating on internal representations of knowledge.
# Example -
#   Initial Information
#       1. If it didn't rain, Harry visited Hagrid today
#       2. Harry visited Hagrid or Dumbledore today, but not both
#       3. Harry visited Dumbledore today
#   What We Can Conclude
#       1. Harry did not visit Hagrid today
#       2. It must have been raining today
# It's this kind of logic reasoning, where we use logic based on the information that we know in order to take 
#information and reach conclusions.

# This is where we will be going over Logics
# Sentence - An assertion about the world in a knowledge representation language. Some way of representing 
#knowledge inside of our computers
# Propositional Logic - Based on a logic of propositions, or just statements about the world.
# Proposition Symbols - P  Q  R
# Each of these symbols represent some fact or sentance about the world.
#   For example, "P", might represent the fact that it is raining.
#                "Q", might represent Harry visited Dumbledore today.
#                "R", might represent Harry didn't visit Hagrid today
# In addition to just having individual facts about the world, we want some way to connect these propositional
#symbols together in order to reason more complexly about other facts that might exist inside the world in 
#which we're reasoning. In order to that we need to, we'll need to introduce some additional symbols that 
#are known as logical connectives. Logical Connectives have their own symbols but can also be identified by
#their key words.
# Logical Connectives -  Not, And, Or, Implication, Biconditional
# What we're going to show for each of these logical connectives is what we call a truth table, a table
#that demonstrates what the "L C" word means when we attach it to a propositional symbol or any sentence 
#inside of our logical language.
#       Not - Placing a Not symbol in front of some sentence of propositional logic tells us to do the 
#             opposite of what comes after our Not symbol.
#                                     (Inputs)  P || -P  (Results with Not Symbol)
#                                           false || true
#                                            true || false

#       And - As opposed to just taking a single argument the way "Not" does, "And" is going to combine two
#             different sentences in propositional logic together. The general logic for what P and Q
#             means is that both of its operands are true. P is true and also Q is true. Here is what the 
#             truth table looks like. 
#                                (Inputs P, Q)  P || Q || P And Q  (Results With P And Q)
#                                          false || false || false
#                                          false || true || false
#                                          true || false || false
#                                          true || true || true

#      Or - Is true if either of its arguments are true. As long as P is true or Q is true, then P "OR" Q
#           is going to be true. Which means the only time that P "Or" Q is false is if both operands are
#           false.
#                                 (Inputs P, Q)  P || Q || P Or Q  (Results With P Or Q)
#                                          false || false || false
#                                          false || true || true
#                                          true || false || true
#                                          true || true || true   

#      Implication - What "Implies" means that if one is true, then the other must be true. So we might say
#                    something like, if it is raining, then I will be indoors. Meaning, it is raining
#                    I wil be indoors, as the logical sentence that we're saying there. The truth table
#                    can be tricky sometimes when dealing with "Implication". P implies Q just means that 
#                    if P is true, Q must be true. But if P is not true, then we make no claim about whether
#                    or not Q is true at all. So, in either case, if P is false, it doesn't matter what Q is.
#                    We can still evaluate the implication to be true. The only way that the implication is 
#                    is ever false is if our premise, P, is true, but the conculsion that we're drawing Q
#                    happens to be false. So in that case, we would say P does not imply Q in that case.
#                                 (Inputs P, Q)  P || Q || P Implies Q  (Results With P Implies Q)
#                                          false || false || true
#                                          false || true || true
#                                          true || false || false
#                                          true || true || true

#       Bi-conditional - We can think of a Bi-conditional as a condition that goes in both directions.
#                        Bi-conditional can be read as an if and only if. So we can say, I will be indoors if
#                        and only if it is raining, meaning if it is raining, then I will be indoors. And if
#                        I'm indoors, it's reasonable to conclude that it is also raining. So this Bi-conditional
#                        is only true when P and Q are the same.
#                                  (Inputs P, Q)  P || Q || P And Q  (Results With P Bi-conditional Q)
#                                          false || false || true
#                                          false || true || false
#                                          true || false || false
#                                          true || true || true
# These five basic logical connectives are going to form the core of the language of propositional logic, 
#the language that we're going to use to describe ideas, and the language that we're going to use in order
#to draw conclusions. 

# Now we will take a look at some of the additional terms that we will need to know about in order to go
#about trying to form this language of propositional logic and writing AI that's actually able to understand
# This sort of logic.

# Here we will be going over the notion of Models. A Model assigns a truth value, where a truth value is either true
#or false, to every propositional symbol. In other words, it's creating what we might call a possible world.
# Model - Assignment of a Truth value to every propositional symbol (a "possible world")
# Example of a Model 
#                   P: It is raining. - Propositional Symbol
#                   Q: It is a Tuesday. - Propositional Symbol

#                   {P = true, Q = false} - Model
# In this model, in other words, in this possible world, it is possible that P is true and Q is false. But
#there other possible worlds or other models as well. There are some models where both of these variables
#are true, and some model where both of these variables are false. 
# In fact, if there are n variables that are propositional symbols like this that are either true or false,
#then the number of possible models is 2 to the n, because each of these possible variables within our model
#could be set to either true or false, if we don't know any information about it.

# Here we will go over a Knowledge Base.
# Knowledge Base - A set of sentences known by a knowledge-based agent. Some set of sentences in propositional
#logic that are things that our AI knows about the world. Our AI can use the information stored in its
#knowledge base to able to draw conclusions about the rest of the world. 
# To understand these conclusions, we will need to introduce a new symbol. That is the notion of entailment.

# Entailment - In every model in which sentence is "a" is true, sentence "b" is also true. 
#                        "a" = "b" (alpha entails beta)
# Alpha and Beta are just representations of sentences in propositional logic. 
# What we mean when we say that alpha entails beta means that in every model, in other words, in every possible
#world in which sentence alpha is true, then sentence beta is also true.
# So if something entails something else, if alpha entails beta, it means that if we know alpha to be true,
#then beta must therefore also be true.
# Example, if alpha = "a tuesday in Janurary", then b = "it is Janurary"
# That means that we can reasonably use deduction based on that first sentence to figure out that the second
#sentence is, in fact, true as well.
# Ultimately, it's this idea of entailment that we're going to try and encode into our computer. We want our
#AI agent to be able to figure out what the possible entailments are. This process is known as inference.

# Inference - The process of deriving new sentences from old ones.
#Example of inference -
#                      P: It is a Tuesday.
#                      Q: It is raining.
#                      R: Harry will go for a run.

#                      KB (Knowledge Base): (P and not Q) implies R.    
#            
# The way we would translate the above sentence as a human would be, 
#"It is a Tuesday and it is not raining, so Harry will go for a run."
# Now we will imagine that our KB has two pieces of information as well. (P = true and Q = false.) 
# Using this information, we should be able to draw some inferences. 
# Because of the two new pieces of information, we can infer that R is true, and that Harry will 
#go for a run. 
# This ultimately is the beginning of what we might consider to be some sort of inference algorithm,
#some process that we can use to try and figure out whether or not we can draw some conclusion.
# What these inference algorithms are going to answer is the central question about entailment.
# The question -
#               Does KB = alpha ?
# In other words, using only the information we know inside of our knowledge base, the knowledge
#that we have access to, can we conclude that this sentence alpha is true? That is the goal.
# To help us acheive this, we can write an algorithm that can look at our KB and figure out
#whether or not this query alpha is actually true?
# There a many algorithms that can helps us do this. One of the simplest, perhaps, is known as
#model checking

# Model checking - 
#                - To determine if KB = alpha:
#                    - Enumerate all possible models. In other words, consider all possible values
#                         of true and false for our variables, all possible states in which our world
#                         can be in.
#                    - If in every model where KB is true, a is true, then KB entails alpha.
# This is going to form the foundation of our model checking algorithm. We are going to enumerate all
#the possible worlds and ask ourselves whenever the KB is true, is alpha true? And if that's the case
#then we know alpha to be true. If it is not, there can be no entailment. Our KB does not entail alpha.
# Example -
#           P: It is Tuesday.   Q: It is raining   R: Harry will go for a run
#   KB: (P and not Q) implies R       P      not Q
#   Query: R
#           P   ||  Q   ||  R   ||  KB
#         false || false||false ||
#         false || false|| true ||
#         false || true ||false ||
#         false || true || true ||
#          true || false||false ||
#          true || false|| true ||
#          true || true ||false ||
#          true || true || true ||

# We have three propositional symbols here, P, Q, and R, which means we have 2 to the third power, or 8
#possible models. 8 possible ways we could assign true and false to all of these models.
# We might ask in each one of them, is the knowledge base true?
# In which of these worlds could this knowledge base possibly apply to?
# In which world is this knowledge base true?
# Given the information we have, plus our KB, we can conclude that there is only one possible world 
#that our KB is true.
#                   P   ||  Q   ||  R   ||  KB
#                  true ||false || true || True
# This is would be the only possible world/model for our KB to exist.

# There can be cases of multiple worlds/models where the KB is True.

