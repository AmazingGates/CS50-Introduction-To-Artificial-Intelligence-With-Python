# And one very common application of reinforcement learning is in game play.

# If we want to teach an agent how to play a game, we just let the agent play the game a whole bunch, and then the
#rewards signal happpens at the end of the game.

# When the game is over, if our AI won the game, it gets a reward of 1, for example.

# And if it lost the game, it gets rewarded negative 1.

# And from that it begins to learn what actions are good, and what actions are bad.

# We don't have to tell the AI what's good or what's bad.

# But the AI figures it out based on that reward.

# Winning the game is some signal.

# Losing the games is some signal.

# And based on all of that it begins to figure out what decisions it should actually make.

# One very simple game, which we may have played before, is a game called NIM.

# And in the game of NIM, we have a whole bunch of objects in a whole bunch of different piles.

# Each pile is an individual row.


#            o
#           ooo
#          ooooo
#         ooooooo


# And the game of NIM is a two player game.

# Where players take turns removing objects from piles.

# And the rule is, that on any given turn we are allowed to remove as many objects as we want from any one of
#the piles.

# We have to remove at least one object, but we can remove as many as we want from exactly one of the piles.

# And whoever removes the last object loses.

# So player one might remove four from the last row/pile.


#            o
#           ooo
#          ooooo
#         ooo


# Player two might remove four from the third row/pile


#            o
#           ooo
#          o
#         ooo


# Next, player onne might remove the entire second row.


#            o
#           
#          o
#         ooo


# Player two, if their being strategic, might remove two from the last row/pile.


#            o
#           
#          o
#         o


# Now we have three rows/piles left, each with one object left.


#            o         
#          o
#         o


# Player one might remove one from one pile.


#            o         
#          
#         o


# Player two removes one from another.


#            o         
#          
#         


# And now player one is going to be left with choosing the last object from the last pile. At which point player 
#one loses the game.


#            


# Fairly simple game.

# Piles of objects. Any turn we choose how many objects to remove from one pile.

# Whoever removes the last object loses.

# And this is the type of game we could incode into an AI fairly easily, because the states are really just four
#numbers.

# Every state is just how many objects in each of the four piles.

# And the actions are things like, how many things are we going to remove from each one of the individual piles.

# And the reward happens at the end.

# That if we are the player that has to remove the last object, then we get some sort of punishment.

# But if we are not, and the other player had to remove the last object, then, we get some sort of reward.

# So we can actually try to show a demostration of this.

# We are going to implement an AI to play the game of NIM.

import random

from nim import train, play


# Here we are going to create an AI as a result of training AI the AI on some number of games.
# The AI is going to play against itself with the idea that the AI is going to play games against itself, learn from
#each of those experiences, and learn what to do in the future.
# And then us Humans will play aginst the AI.

# So initially we'll say train zero times, meaning we're not going to let the AI play any practice games against
#itself in order to learn from those experiences.
# We're just going to see how well it plays.
ai = train(1)
#play(ai)

# This is how the first game against the AI played out

# This is our initial game board, and it is our turn first.

#            o
#           ooo
#          ooooo
#         ooooooo (We choose to take 5 objects from this pile)

# Now it is the AI's turn and this is the new game board.

#            o (AI chose to take 1 object from this pile.)
#           ooo
#          ooooo
#         oo

# It's our turn again and this is the new game board.

#
#           ooo
#          ooooo (We chose to take 5 objects from this pile)
#         oo

# Now it's the AI's turn and this is the new game board.

#            
#           ooo (AI chose to take 2 objects from this pile)
#          
#         oo

# Now it is our turn and this is the new game board.

#            
#           o
#          
#         oo (We chose to take 2 objects from this pile.)

# AI will make the last move, losing the game, and this is the new game board.

#            
#           o (The last move AI can make, Game Over)
#          
#         

# We were able to win easily because the AI was just playing randomly.
# It didn't have any prior experience that it was using in order to make these sorts of judgements.

# Now we will let the AI train itself on something like 10,000 games.

# Let the AI play 10,000 games of nim against itself.

# Everytime it wins or loses it will learn from that experience, and learn in the future what to do, and what not 
#to do.

# Once the AI has trained on its 10,000 games, it will once again challenge us, but this time, with a lot more
#experience.

# Since we we're unable to run the actually ai = train(10,000) program, we will recreate it.

# This is the first game we played against the AI after its training of 10,000 games agaoinst itself.

# This is the starting game board

#            o 
#           ooo (AI made the first move and chose to take one object from this pile.)
#          ooooo
#         ooooooo

# It is now our turn and this is the new game board

#            o
#           oo
#          ooooo
#         ooooooo (We chose to take one object from this pile)

# Now it is the AI's turn again and this is the new game board

#            o
#           oo
#          ooooo
#         oooooo (AI chose to take all the objects from this pile)

# It is our turn now and this is the new game board

#            o
#           oo
#          ooooo (We decided to take 3 objects from this pile)
#         

# It is now the AI's turn and this is the new board

#            o (AI decided to take 1 object from this pile)
#           oo
#          oo
#         

# It's our turn again and this is the game board

#            
#           oo (We decided to take 1 from this pile)
#          oo
#         

# Now it's AI's turn again and this is the game board

#            
#           o
#          oo (AI took 2 objects from this pile)
#         

# This is the final move and game losing move for us since we are making this move, and this is the board.

#            
#           o (This is the only option to take for us)
#          
#         

# AI has won this round.

# So it seems like, after playing 10,000 games against itself, the AI has learned something about what states and 
#actions tend to be good, and has begun to learn some sort of pattern for how to predict what actions are going to be
#good and what actions are going to be bad in any given state.

# So reinforcement learning can be a very powerful technique for achieving these sort of game playing agents.

# Agents that are able to play a game well just by learning from experience.

# Whether that's playing against other people, or by playing against itself, and learning from those experiences
#as well.

# Now nim is a bit of an easy game to use reinforcement learning for, because there are so few states.

# There are only states for how many different objects are in each of these various different piles. 

# We might image that it is going to be harder if we think of a game like chess, or games where there are many more
#states and many more actions we can imagine taking, where it's not going to be as easy to learn for every state and
#every action, what the value is going to be.

# So often times in that case, we can't neccessarily learn exactly what the value is for every state and every action.

# But we can approximate it.

# So much as we saw with minimax, where we could use a depth limiting approach to stop calculating at a certain
#point in time, we could do a similar type of approximation, known as function approximation, in a reinforcement
#learning context.


#   Function Approximation -
# - Approximating Q(s,a), often by a function combining various features, rather than storing one value for every
#state-action pair


# Where instead of learning a value of Q for every state and every action, we just have some function that estimates
#what the value is for taking this action in this particular state that might be based on various different features
#of the state of the enviornment that the agent happens to be in.

# Where we might have to choose what those features actually are, but we can begin to learn some patterns that 
#generalize beyond one specific state and one specific action, that we can begin to learn that certain features 
#tend to be good things or bad things.

# Reinforcement learning can allow us to generalize beyond one particular state and say, if this other state looks
#kinda like this state, then maybe the similar types of actions that worked in one state will also work in another
#state as well.

# So this type of approach can be quite helpful as we begin to deal with reinforcement learning that exist in 
#larger and larger state spaces, where it's just not feasible to explore all of the possible states that can
#actually exist.

# So there then are two of the main categories of reinforcement learning.

# Supervised Learning, where we have labeled input and output pairs.

# And Reinforcement Learning, where an agent learns from rewards or punishments that it recieves.

# The third major category of Machine Learning that we'll just touch on briefly, is known as unsupervised learning.

#   Unsupervised Learning -
# - Given input data without any additional feedback, learn patterns.

# And unsupervised learning happens when we have data without any additional feedback, without labels.

# In the supervised learning case, all of our data had labels.

# We labeled the data point with whether that was a rainy day, or a not rainy day, and using those labels we were 
#able to infer what the pattern was.

# Or we labeled data as a counterfeit bill, or an authentic bill, and using those labels, we were able to draw 
#inferences and patterns to figure out what does a bank note look like, and what it doesn't.

# In unsupervised learning, we don't have any access to any of those labels, but we would still like to learn some
#of those patterns.

# And one of the task that we might want to perform in unsupervised learning is something like clustering.

#   Clustering -
# - Organizing a set of objects into groups in such a way that similar objects tend to be in the same group.

# Where clustering is just a task, given some set of objects, organizing into distinct clusters.
# Groups of objects that are similar to one another.

# And there's lots of applications for clustering.

#   Some Clustering Applications -
#
# - Genetic Research
# - Image Segmentation
# - Market Research
# - Medical Imaging
# - Social Network Analysis

# One technique for clustering is an algorithm known as k-means clustering

#   k-Means Clustering -
# - Algorithm for clustering data based on repeatedly assigning points to clusters and updating those clusters'
#centers.

# And what k-means clustering is going to do is, it's going to divide all of our data points into k different
#clusters, and it's going to do so by repeating this process assigning ponts to clusters, and then moving around 
#those clusters' centers.

# We're going to define a cluster by its center, the middle of the cluster, and then assign points to that cluster
#based on which center is closest to that point.

# We'll go over an example of that now.


#                     o             r o
#              b o   o   o   o
#                 o      o                                           o
#                       o                                       o
#                                                                         o
#                                                              o    o
#                                                                        o
#                                                               o   o   o
#                                      g o
#                      o     o    o
#          o    o           o            o
#           o    o    o         o       o


# Here we have a whole bunch of unlabeled data.

# Just various data points that are in some sort of graphical space, and we would like to group them into various
#different clusters.

# But we don't know how to do that oiginally.

# Let's say we want to assign 3 clusters to this group.

# We have to choose how many clusters we want in k-means clustering, but we can try multiple and see how well those
#values perform.

# But we'll start just by randomly picking some places to put the centers ofthose clusters. 

# Maybe we a blue cluster, a red cluster, and a green cluster.

# And we're going to start with the centers of those clusters, just being in these three locations as specified.

# And what k-means clustering tells us to do, is once we have the centers of the clusters, assign every point to
#a cluster, based on which cluster center it is closest to.

# So we end up with something like this.


#                    b o                        r o (center)
#        (center)b o   b o  b o              ro
#                b o     b o                                          r o
#                      b o                                      r o
#                                                                        r o
#                                                             r o   r o
#                                                                       r o
#                                                              g o  g o  g o
#                                    g o (center)
#                     g o    g o   g o
#         g o   g o          g o           g o
#          g o   g o   g o        g o      g o

# Where all of the points labeled b are closest to the blue cluster center than any other cluster center, 
#all of the points labeled g are closest to the green cluster center than any other center, and all of the points
#labeled r are closest to the red cluster center than any other center.

# So here is one possible assignment of all these points, to three different clusters.

# But it's not great.

# It seems like some points are a little far apart from the cluster centers that they are labeled for.

# It might not be our ideal choice of how we would cluster these various different data points.

# But k-means clustering is an iterrative process, meaning that after we do this, there's a next step which
#is after we've assigned all the points to the cluster center that it's nearest to, we are going to re-center
#the clusters
# Meaning take the cluster centers, and move them to the middle of all the points that are in that cluster.

# So we'll take the blue center and move it to the center of all of the points that were assigned to the blue 
#cluster.

# We'll do the same thing for red. We'll move the cluster center to the middle of all the points that are assigned
#to the re cluster, weighted by how many points there are.
# This means that the red center will closer to the location where the majority of the points assigned to the red
#cluster are.

# We'll finally do the same for the green cluster center.


#                    b o                        r o 
#        b o   b o (center) b o              ro
#                b o     b o                                          r o
#                      b o                                      r o
#                                                             (center)    r o
#                                                             r o   r o
#                                                                       r o
#                                                              g o  g o  g o
#                                    g o 
#                     g o    g o   g o
#         g o   g o          g o  (center) g o
#          g o   g o   g o        g o      g o



# So we re-center all of the clusters, and then we repeat the process.

# We'll go ahead and now reassign all of the points to the cluster center that they are now closest to.

# And now that we've moved around the cluster centers, these cluster assignments might change.


#                    b o                   b o 
#        b o   b o (center) b o         b o
#                b o     b o                                          r o
#                      b o                                      r o
#                                                             (center)    r o
#                                                             r o   r o
#                                                                       r o
#                                                              r o  r o  r o
#                                    g o 
#                     g o    g o   g o
#         g o   g o          g o  (center) g o
#          g o   g o   g o        g o      g o


# So we can reassign which clusters each of these data points belong to, and then repeat the process again.

# Moving each of these cluster means to the middle of the cluster, (or mean, the average of all of the other points
#that happen to be there).


#                    b o                   b o 
#        b o   b o  b o   (center)      b o
#                b o     b o                                          r o
#                      b o                                      r o
#                                                                 r o
#                                                             r o   (center) r o
#                                                                       r o
#                                                              r o  r o  r o
#                                    g o 
#                     g o    g o   g o
#         g o   g o          g o        g o
#          g o   g o   g o    (center) g o  g o


# And repeat the process again.

# Go ahead and assign each of the points to the cluster that they are closest to.

# So once we reach a point where we've assigned all of the points to clusters that they are nearest to, and nothing
#changed, we've reached the equalibrium.

# No points are changing their assignment, and as a result we can declare that this algorithm is now over, and we now
#have some assignment of these points into three different clusters.

# And it looks like we did a pretty good job of trying to identify which points are more similar to one another,
#than they are to points in other groups.

# And we did so without any access to some labels to tell us what these various different clusters were.

# We just used an algorithm in an unsupervised sense, without any of those labels, to figure out which points 
#belonged to which category.

# And there are many more algorithms in each of these various different fields within machine learning,
#supervised learning, reinforcement learning, unsupervised learning, but those are many of the big picture 
#foundational ideas that underlie a lot of these techniques.

# That was a look at some of the principals that are at the foundation of modern machine learning, this ability 
#to take data, and learn from that data so that the computer can perform a task, even if they haven't been given
#explicit intructions in order to do so.