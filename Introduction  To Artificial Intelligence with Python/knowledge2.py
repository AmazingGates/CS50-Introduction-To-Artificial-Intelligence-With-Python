# This is where we will practice writing actual code for propsitional symbols and logic, and these connectives like 
#"and", "or", "not", and "implication" etc...

# Here, we will go over the process of storing information about our worlds, models.
# In the practice library, capitalized Symbol is used to create symbols.

# Symbol("rain") - It is raining
# Symbol("hagrid") - Harry visited Hagdrid
# Symbol("dumbledore") - Harry visited Dumbledore
# We will save our Symbols to variables so that we can use them later.

# rain = Symbol("rain")
# hagdrid = Symbol("hadrid")
# dumbledore = Symbol("dumbledore")

# Now that we have these logical symbols, we can use logical connectives to combine them together.
# Example -
#           Sentence = And(rain, hagdrid)
#           print(sentence.formula()) # This is a practice function that takes a sentence in propositional logic
#           and prints it out so that we, the programmers, can now see this in order to get an understanding
#           for how it actually works.
# When run, we should see this sentence in propositional logic (rain and hagdrid)
# This is the logical representation of what we have here in our Pyhthon program of saying And(rain, hagdrid).
# This is quite common in Python object-oriented programming, where we have a number of different classes,
#and we pass arguments into them in order to create a new "and" object, for example, in order to represent
#this idea. 

# What we would like to do now is somehow encode the knowledge that we have about the world in order to solve
#that problem from the beginning of class, where we talked about trying to figure out who Harry visited,
#and trying to figure out if its raining, or if it's not raining. 
# What knowledge do we have?
#                           knowledge = Implication(Not(rain), hagdrid)
# So we're saying implication, the premise is that it's not raining. And if it is not raining, then Harry
#visited Hagdrid.
# To see the logical formula of our sentence, we can run 
#                                                  print(knowledge.formula())
#which will give us the equivalent to that same idea of our sentence.
# As a result, we will get a text-based version of what we were looking at before,
#                                                                   (not rain) => hagdrid
# which is saying, that if it is not raining, then Harry visited Hagdrid.

# Since we know multiple things, or have multiple pieces of knowledge, we can wrap our all of our
#knowledge in an and.
#                       knowledge = And(
#                                       Implication(Not(rain), hagdrid),
#                                       Or(hagdrid, dumbledore)
#                                       )
# So we're saying knowledge is an And of multiple different sentences. Or, we knkow multiple sentences 
#to be true.
# One sentence that we know to be true is the Implication, that if it is not raining, then Harry
#visited Hagdrid. 
# Another sentence that we know to be true is or Hagdrid Dumbledore. In other words, Hagdrid or dumbledore 
#is true, because we know that Harry visited either Hagdrid or Dumbledore.
# We actually know even more than that. The initial sentence from previously said that Harry visited 
#Hagdrid or Dumbledore, but not both. 
# So know we want a sentence that will encode the idea taht Harry didn't visit both and Hagdrid
#and Dumbledore.
# The notion of Harry visiting Hagdrid and Dumbledore would look like this
#                                                    And(hagdrid, dumbledore)
# And if that is not true, if we want to say not that, then we can just wrap the whole thing inside of 
#a not. 
#                                                   Not(And(hagdrid, dumbledore)),
# This is what we have so far
#                               knowledge = And(
#                                               Implication(Not(rain), hagdrid),
#                                               Or(hagdrid, dumbledore),
#                                               Not(And(hagdrid, dumbledore)),
#                                               )
# So now these three lines inside the And says that 
#                                                  if it is not raining, then Harry Visited Hagdrid
#                                                  Harry visted Hagdrid or Dumbledore
#                                                  Harry didn't visit both Hagdrid and Dumbledore

# Finally we can add the last piece of information that we knew, which was that Harry visited Dumbledore
#                                                     dumbledore

# So this is the final formula for our KB with all of the information that we know

#                                knowledge = And(
#                                               Implication(Not(rain), hagdrid),
#                                               Or(hagdrid, dumbledore),
#                                               Not(And(hagdrid, dumbledore)),
#                                               dumbledore
#                                               )
# After print, this is what a logical representation of our KB would look like.

#         ((not rain) => hagdrid) And (hagdrid, dumbledore) And (Not(hagdrid, dumbledore)) And dumbledore

# Now that we have our KB, we would like to use Model Checking to ask a query, based on this information,
#do we know whether or not it's raining?

# The idea is, in order to do Model Checking, we need to enumerate all of the possible Models.
# And for each of the possible Models, we need to ask ourselves, is the KB true? And, is the
#query true.

# The first thing we need to do is enumerate all of the possible Models, meaning for all possible symbols 
#that exist, we need to assign true and false to each one of them and see whether or not it's still
#true.
# Here is the way we are going to that.
# First, we're going to start by getting all of the symbols in both the knowledge and the query, by
#figuring out what symbols we are dealing with. 
# In this case, the symbols that we are dealing with are 
#           rain and Hagdrid and Dumbledore

# This is how we Model check based on our knowledge.

# model_check(knowledge, rain)
# We provided 2 arguments to our model_check() what the query is. The first argument is knowledge,
#because that's where our information is, and the second argument is what we are querying, or what 
#the thing I want to ask is.
# And what we want to ask in this case is, is it raining?


#                           knowledge = And(
#                                               Implication(Not(rain), hagdrid),
#                                               Or(hagdrid, dumbledore),
#                                               Not(And(hagdrid, dumbledore)),
#                                               dumbledore
#                                           )

#                           print(model_check(knowledge, rain))

# If we were to run this print, based on everything we have gone over, we should have a return 
#of True. Meaning that based on this information, we can conclusively say that it is raining, because
#using this Model Checking algorithm, we were able to check that in every world where this knowledge
#is true, it is raining.
# In other words, there is no world where this knowledge is true, and it is not raining.
# So we can conclude that it is, in fact, raining.

# Here we will be going over Knowledge Engineering. 
# Knowledge Engineering - This is when SWE and AI will take a problem 
#and try to fiugre out how to distill it down into knowledge that is representable by a computer.
# We will take a look at a few example of Knowledge Engineering.

# Here we will code the logic for a classic Clue game.
# We will try to formalize it and see if we could train a computer to be able to play this game 
#by reasoning through it logically.

# In the game of Clue, there's a number of different factors that are going on. But the basic 
#premise of the game, is that there a number of different people. And three of these, one person,
#one room, and one weapon, is the solution to the mystery, the murderer and what room they were in
#and what weapon they used.

# What happens at the beginning of the game is that all the cards are randomly shuffled together.
# And three of them, one person, one room, and one weapon, are placed into a sealed envelope that
#we don't know. And we would like to figure out, using some sort of logical process, what's inside
#the envelope, which person, which room, and which weapon. And we do so by looking at some, but all, 
#of the cards we have, to try to figure out what happened.

# In order to this, we'll begin by thinking about what propositional symbols we're ultimately going to need.
# Remember, propositional symbols are just some symbol, some variable, that can be either true or false
#in the world/model.
# In this case, the propositional symbols are really just going to correspond to each of the possible
#things that could be inside the envelope.

#                 Propositional Symbols

#           People         Rooms           Weapons

#        Col. Mustard     Ballroom          Knife
#        Prof. Plum       kitchen           Revolver
#        Ms. Scarlet      Library           Wrench

# Using these propositional symbols, we can begin to create logical sentences, create knowledge
#that we know about the world.
# For example, we know someone is the murderer, that one of the three people is, in fact, the murderer.
# This is how we could possibly encode the people symbols
#               (mustard or plum or scarlet)
# This piece of knowledge encodes that one of these three people isthe murderer. We don't know which,
#but one of these things must be true.
# What other information do we know? Well, we know for example, one of the rooms must have been the room
#in the envelope. 
#               (ballroom or kitchen or library)
# The crime was committed in either the ballroom or the kitchen or the library.
# And likewise, we can say the same thing about the weapon, that it was either the knife, or the revolver
#or the wrench, that one of these weapons must have been the weapon of choice
#               (knife or revolver or wrench)
# This is the knowledge we have so far

#                       Clue
#               (mustard or plum or scarlet)
#               (ballroom or kitchen or library)
#               (knife or revolver or wrench)

# Using these cards we can deduce information. That if someone gives us a card, for example, 
#we have the Prof. Plum card in our hand, then we know that Prof. Plum card can't be inside the envelope.
# We know that Prof. Plum is not the criminal, so we know a piece of information like not Plum, for example.
# We know that Prof. Plum has to be false. This propositional symbol is not true.

# Now lets create our KB with the information from our initial knowledge. And we can see what that knowledge 
#looks like by printing out knowledge.formula.

# Knowledge = And(
#                 Or(mustard, plum, scarlett),
#                 Or(ballroom, kitchen, library),
#                 Or(knife, revolver, wrench),
#                 )
# print(knowledge.formula())

# After printing we get back a formula with our information in logical format.

# (ColMustrad or ProfPlum or MsScarlet) and (ballroom or kitchen or library) and (knife or revolver or wrench)

# We will next use addition information to try and help ourselves logically reason our way through this
#process. 
# And we're just going to provide the information. Our AI is going to take care of doing the inference
#and figuring out what conclusions it's able to draw. 
# So for example, if we have the ColMustard card, we knkow that the mustard symbol must be false.
# In other words, mustard is not the one in the envelope, is not the criminal.
# So we can add to our knowledge by using something called ".add", which is a way of adding knowledge/
#logical sentence to an And clause. So we can say 
#               knowledge.add(Not(mustard))
# We happen to know this because we have the ColMustard card, which eliminates him as a suspect.
# And then we will use a check function to check our knowledge.

# Knowledge = And(
#                 Or(mustard, plum, scarlett),
#                 Or(ballroom, kitchen, library),
#                 Or(knife, revolver, wrench),
#                 )
# Knowledge.add(Not(mustard))

# check_knowledge(knowledge)

# Not let's add a few more cards that we have as examples.
#           knowledge(Not(kitchen))
#           knowledge(Not(revolver))

# So know we have three cards that we can draw information from.

# Knowledge = And(
#                 Or(mustard, plum, scarlett),
#                 Or(ballroom, kitchen, library),
#                 Or(knife, revolver, wrench),
#                 )
# Knowledge.add(Not(mustard))
# knowledge.add(Not(kitchen))
# knowledge.add(Not(revolver))

# check_knowledge(knowledge)

# These are the things we know to be true.

# With this new information, when we run the program again, we will have eliminated some possibilities.

# Let's say that another card was revealed that added to our knowledge.
# Because we are not sure which card was revealed, we can add the knowledge like this, since we are not 
#sure which card is not in the envelope, we just know one of them is not in the envelope.
#               knowledge.add(Or(
#                                Not(scarlett), Not(library), Not(wrench))
#                               )
# So, at least one of these needs to be false, we don't know which one, but know that one is false.
# It is possible that it could be multiple, but we still don't know.
# This is what our new KB looks like with the new information.

# Knowledge = And(
#                 Or(mustard, plum, scarlett),
#                 Or(ballroom, kitchen, library),
#                 Or(knife, revolver, wrench),
#                 )
# Knowledge.add(Not(mustard))
# knowledge.add(Not(kitchen))
# knowledge.add(Not(revolver))

# knowledge.add(Or(
#                  Not(scarlett), Not(library), Not(wrench))
#                 )

# check_knowledge(knowledge)
# If we run the program now we won't see a change from earlier. So we'll need a little more information.

# Next, let's say that someone showed us the Prof. Plum card. 
#           knowledge.add(Not(plum))
# Finally, when we run the program now, we will have draw some new conclusions. Now we are able to eliminate
#Prof. Plum as a suspect, because he couldn't be in the envelope.

# Knowledge = And(
#                 Or(mustard, plum, scarlett),
#                 Or(ballroom, kitchen, library),
#                 Or(knife, revolver, wrench),
#                 )
# Knowledge.add(Not(mustard))
# knowledge.add(Not(kitchen))
# knowledge.add(Not(revolver))

# knowledge.add(Or(
#                  Not(scarlett), Not(library), Not(wrench))
#                 )

# knowledge.add(Not(plum))
# check_knowledge(knowledge)

# With all of the knowledge that we have obtained so far, we able to conclude that it was Ms. Scarlet.

# We still have to solve the mystery of where she did it and what weapon she used. 
# To help us finally get all the answers, let's say that we got one more piece of information.
# Let's say that we know that it's not in the ballroom. Let's say that someone has shown us the 
#ballroom card, so we know that for sure it's not the ballroom. Which means that we finally
#conclude that it's the library. 
#               knowledge.add(Not(ballroom))
# After running the final version of our KB we should have all the answers we need to solve 
#our mystery.

# Knowledge = And(
#                 Or(mustard, plum, scarlett),
#                 Or(ballroom, kitchen, library),
#                 Or(knife, revolver, wrench),
#                 )
# Knowledge.add(Not(mustard))
# knowledge.add(Not(kitchen))
# knowledge.add(Not(revolver))

# knowledge.add(Or(
#                  Not(scarlett), Not(library), Not(wrench))
#                 )

# knowledge.add(Not(plum))
# knowledge.add(Not(ballroom))
# check_knowledge(knowledge)

# Now we take a look at an example of Logical Puzzles. This is when we puzzle our way through trying
#to figure something out. 
# This is what a classical logical puzzle might look like.

# Logical Puzzle -
#   - Gilderoy, Minerva, Pomona and Horace each belong to a different one of the four houses: 
#       Gryffindor, Hufflepuff, Ravenclaw, and Slytherin House.
#   - Gilderoy belongs to Gryffindor or Ravenclaw.
#   - Pomona does not belong in Slytherin.
#   - Minerva belongs to Gryffindor.

# Using this information, we need to be able to draw some conclusions about which person should be
#assigned to which house. 
# And again, we acn use the same exact idea to try and implement this notion.
# So we need some propositional symbols.
# This time, our propositional symbols will be a little more complex.

# Logical Puzzles
#         Propositional Symbols

#   GilderoyGryffindor        MinervaGryffindor
#   GilderoyHufflepuff        MinervaHufflepuff
#   GilderoyRavenclaw         MinervaRavenclaw
#   GilderoySlytherin         MinervaSlytherin

#   PomonaGryffindor          HoraceGryffindor
#   PomonaHufflepuff          HoraceHufflepuff
#   PomonaRavenclaw           HoraceRavenclaw
#   PomonaSlytherin           HoraceSlytherin

# These are our propositional symbols, one for each person and house.
# Remember, every propositional symbol is either true or false. For example Gilderoy in Gryffindor
#is either true or false, and so on for the rest of the propositional symbols.
# Using this type of knowledge we can then begin to think about what types of logical sentences
#we can say about the puzzle. Before we even think about the information we were given, we can think about
#the premise of the problem, that every person is in a different house.
# That tells us sentences like this.
#       (PomonaSlytherin implies Not(PomonaHufflepuff))
# This is saying, that if Pomona is in Slytherin, then we know she is not in Hufflepuff.
# And we can apply that logic to all the other propositional symbols. Meaning that no matter
#what person we pick, if they're in one house, then they're not in some other house.
# So we'll propably end up with a bunch of knowledge base statements that are written in the form
#of our first sentence.
# We were also given the information that each person is in a different house, which means that
#we will also have pieces of information that look like this.
#       (MinervaRavenclaw implies Not(GilderoyRavenclaw))
# We can conclude this because we know from prior information that all of the students are in different
#houses, so no two can be in the same house at once. This information will give us a lot of similar
#sentences to the one we just wrote, expressing the idea for different people and other houses as well
# And so in addition to sentences of these form, We also have the knowledge that was given to us.
# Information like Gilderoy was in Gryffindor or in Ravenclaw that would be represented like this
#       (GilderoyGryffindor Or GilderoyRavenclaw)
# Which is saying that Gilderoy is in Gryffindor or Ravenclaw.
# By using these types of sentences, we can begin to draw some conclusions about the world.

#           Logical Puzzzle

#   (PomonaSlytherin implies Not(PomonaHufflepuff))
#   (MinervaRavenclaw implies Not(GilderoyRavenclaw))
#   (GilderoyGryffindor Or GilderoyRavenclaw)

# Let's see an example of this. We'll go ahead and actually try and implement this logical
#puzzle to see if we can figure out what the answer is. 

# This is where we will implement an example logic puzzle with code., to see if we can figure out what 
#the answer is.

# people = ["Gilderoy", "Pomona", "Minerva", "Horace"] # This is where we created a list of people and houses
# houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# symbols = []

# knowledge = And()

# for person in people: # This is where we created one symbol for every person in every house
#     for house in houses:
#         symbols.append(Symbol(f"{person}{house}"))

# # This is where we added some information
# #Each person belongs to a house
# for person in people:
#     knowledge.add(Or(
#         symbol(f"{person}Gryffindor"),
#         symbol(f"{person}Hufflepuff"),
#         symbol(f"{person}Ravenclaw"),
#         symbol(f"{person}Slytherin")
#))

# # This is other information that we know
# # Only one house per person.
# for person in people:
#     for h1 in houses:
#         for h2 in houses:
#             if h1 != h2:
#                knowledge.add(
#                     Implication(Symbol(f"{person}{h1}"), Not(Symbol(f"{person}{h2}")))
# )

# # This is more information that we know
# # Only one person per house.
# for house in houses:
#     for p1 in people:
#         for p2 in people:
#             if p1 != p2:
#                 knowledge.add(
#                     Implication(Symbol(f"{p1}{house}"), Not(Symbol(f"{p2}{house}")))
# )


# From information we have, we know that one of these is true.
# knowledge.add(
#      Or(Symbol("GilderoyGryffindor"), Symbol("GilderoyRavenclaw"))
#)

# This is also information that we know
#knowledge.add(Not(Symbol("PomonaSlytherin")))

# We also know this
# Knowledge.add(Symbol("MinervaGryffindor")))

# This loop here goes over all of our symbols and checks to see if our knowledge entails that
#symbol, and if it does, if we know the symbol is true, we print out the symbol
# for symbol in symbols:  
#     if model_check(knowledge, symbol):
#         print(symbol)

# Once we run our program with all of our knowledge, python should give us our answer.
# Here is our entire code.

# people = ["Gilderoy", "Pomona", "Minerva", "Horace"] # This is where we created a list of people and houses
# houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

# symbols = []

# knowledge = And()

# for person in people: # This is where we created one symbol for every person in every house
#     for house in houses:
#         symbols.append(Symbol(f"{person}{house}"))

# for person in people:
#     knowledge.add(Or(
#         symbol(f"{person}Gryffindor"),
#         symbol(f"{person}Hufflepuff"),
#         symbol(f"{person}Ravenclaw"),
#         symbol(f"{person}Slytherin")
#))

# for person in people:
#     for h1 in houses:
#         for h2 in houses:
#             if h1 != h2:
#                knowledge.add(
#                     Implication(Symbol(f"{person}{h1}"), Not(Symbol(f"{person}{h2}")))
# )

# for house in houses:
#     for p1 in people:
#         for p2 in people:
#             if p1 != p2:
#                 knowledge.add(
#                     Implication(Symbol(f"{p1}{house}"), Not(Symbol(f"{p2}{house}")))
# )

# knowledge.add(
#      Or(Symbol("GilderoyGryffindor"), Symbol("GilderoyRavenclaw"))
#)

# knowledge.add(Not(Symbol("PomonaSlytherin")))
# Knowledge.add(Symbol("MinervaGryffindor")))

# for symbol in symbols:  
#     if model_check(knowledge, symbol):
#         print(symbol)

# Conclusion
# GilderoyRavenclaw
# PomonaHufflepuff
# MinervaGryffindor
# HoraceSlytherin


#               Mastermind
# Here we will play a simplified version of the game Mastermind, where there are four colors, red blue, green,
#and yellow, and they're in some order, but we don't kknow what order. We just have to make a guess, and 
#we'll find out of red, blue, green, and yellow how many of the four we gotin the right position.

# So in a simplified version of this game, we might make a guess like red, blue, green, yellow, and we'll
#find out something like two of those four are in the correct position, but the other 2 are not.
# Then we could make a guess and say, all right, try this, blue, red, green yellow.
# Trying to switch 2 of them around, and this time maybe we'll find out that none of these is in the 
#correct position. The question then becomes, what is the correct order of these 4 colors?
# Because the first sequence has 2 correct, and the second sequence has none correct could imply
#that since we switched red and blue from the first sequence and that returned no correct positions
#for the second sequence, red and blue must have been in the correct positions in the first sequence,
#and that green, and yellow should be looked at. We can then move and try to encode this inside our
#computer. And it's going to be very similar to the logic puzzle that we just did a moment ago.

# colors = ["red", "blue", "green", "yellow"] # Here we have 4 different colors, and 4 positions those
# symbols = []                                #colors could be.
# for i in range(4):
#     for color in colors:
#         symbols.append(Symbol(f"{color}{i}))

# knowledge = And()

# # Each color has a position.
# Here we have some additional knowledge
# for color i colors:
#     knowledge.add(Or(
#         Symbol(f"{color}{0}")
#         Symbol(f"{color}{1}")
#         Symbol(f"{color}{2}")
#         Symbol(f"{color}{3}")
#))

# # Only one position per color.
# for color in colors:
#     for i in range(4):
#         for j in range(4):
#             if i !=j:
#                 knowledge.add(Implication(
#                     Symbol(f"{color}{i}"), Not(Symbol(f"{color}{j}")
#))

# knowledge.add(Symbol("red")) is in the 0 position 

# Knowledge.add(Symbol("blue")) is the 1 position 

# knoweldege.add(Not(Symbol("yellow"))) in the 3 position

# for symbol in symbols:  
#     if model_check(knowledge, symbol):
#         print(symbol)

# Conclusion
# red0
# blue1
# yellow2
# green3




# Here we will be going over Inference Rules.

#           Inference Rules
# Rules we can apply to take knowledge that already exists and translate it into new forms of knowledge.
# The general way we will structure Inference Rules is by having a horizontal line here.
# Anything above the line is goind to represent a premise, something that we know to be true. 
# And anything below the line will be the conclusion that we can arrive at after we apply the logic
#from the inference rule that we are going to demonstrate.


#      If it is raining, then Harry is inside == Things we know to be true
#     -----------------------------------------------------------------------
#            Harry is inside == conclusion

# We will do some of these IR's by demonstarting them in english first, then transalte them into the world 
#of propositional logic so we can what those IR's actually look like. 
# So for example, let's imagine that we have access to 2 pieces of information.
# We know, for example, that if it is raining, then Harry is inside, for example.
# And let's say that we also knkow that it is raining.
# We can reasonably look at this information and conclude that, Harry must be inside.
# This IR is known as Modus Ponens, and it's phrased more formally in logic as this.
# If we know that alpha implies beta, in other words, if alpha, then beta, and we also know that alpha is 
#true, then we should be able to conclude that beta is also true.


#            a -> B == if alpha then beta

#                       a
#    -----------------------------------------------
#                       B

# We can apply this IR to take these 2 pieces of information and generate that new information.

# Notice that this is a totally different approach from the model checking approach, where the approach
#was look at all of the possible worlds and see what's true in each of these worlds.
# Here, we're not dealing with any specific world. We're just dealing with the knowledge that we know
#and what conclusions we can arrive at based on that knowledge. That we know that A implies B, and
#the conclusion is B.
# This will be true for many, if not all of our  IR's that we take a look at.
# Modus Ponens basically means application of impication, that if we know that alpha implies beta, 
#then we can conclude beta.

# Now we will take a look at another example.
# Something like Harry is friends with Ron and Hermione. Based on that information, we can reasonably
#conclude that Harry is friends with Hermione. That must also be true.

#     Harry is friends with Ron and Hermione.
#    ---------------------------------------------
#     Harry is friends with Hermione.

# This IR is known as And Elimination. 
# And Elimination says that if we have a situation where alpha and beta are both true, we have information
#alpha and beta, then just alpha is true

#            a and B
#   ----------------------------------
#               a

# Basically, if we know that both parts are true, then one of those parts must also be true.

# In addition to that, let's take a look at another example of an IR, something like it is not true
#that Harry did not pass the test. Well if it is false that Harry did not pass the test, then the only
#reasonably conclusion is that Harry did pass the test.

#       It is not true that Harry did not pass the test.
#   ---------------------------------------------------------
#                   Harry did pass the test

# Instead of this being an And Elimination, this is called a Double Negation Elimination.
# Double Negation Elimination means that if we have 2 negatives inside of our premise, then we can just
#remove them all together. They cancel each other out. One turns false, and the other one turns false 
#back into true.

# Basically, if the premise is not alpha, then the conclusion we can draw is just alpha.

#               Not(Not a)
#   ------------------------------
#                    a

# We can say alpha is just true.