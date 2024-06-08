# Here we will take a look at a couple of more of these examples. If it is raining, then Harry is inside. How do
#we reframe this? If we know that it is raining, then Harry is inside, then we can conclude one of 2 things must 
#be true. Either it is not raining, or Harry is inside. 
# This example is a bit more tricky. So let's think about it a little more.
# The first premise, if it is raining, then Harry is inside, is saying that if I know that it is raining, then 
#Harry must be inside. So, what is the other possible case here? Well if Harry is not inside, then I know that
#it must not be raining. So one of those two situations must be true. Either it's not raining, or it is raining,
#in which case Harry is inside. So the conclusion that we can draw is either it is not raining, or it is raining,
#so therefore, Harry is inside. This is a way to translate if-then statements into or statements.
# This is known as Implication Elimination.


#       If it is raining, then Harry is inside.
#   -----------------------------------------------
#        It is raining or Harry is inside.

# This is similar to what we did in the beginning when we were first looking at those very first sentences
#about Harry and Hagrid and Dumbledore. pharsed a little bit more formally, this says that if we have the 
#implication, alpha implies beta, then we can draw the conclusion that either not alpha or beta, because
#there are only two possibilities. Either alpha is true or alpha is not true. So one of these possibilties
#is alpha is not true. But if alpha is true, then we can draw the conclusion that beta must be true. 
# So either alpha is not true or alpha is true, in which case beta is also true. This is one way to turn an
#implication into just a statement about or.

#           Implication Elimination

#               a implies B
#   ----------------------------------
#               Not a or B


# In addition to eliminating implications, we can also eliminate biconditionals as well. 
# Let's take a look at an english example. It is raining if and only if Harry is inside. This if and only if
#sounds like a biconditional, the double arrow sign that we saw in propositional logic not too long ago.
# What does this actually mean if we were to translate this? Well, this means that if it is raining, then
#Harry is inside. If it is raining, then Harry is inside, and if Harry is inside, then it is raining, meaning
#that this implication goes both ways. This is what we call biconditional elimination.


#                        It is raining if and only if Harry is inside.
#   ---------------------------------------------------------------------------------------------
#        If it is raining, then Harry is inside, and if Harry is inside, then it is raining.



# This is what we would call biconditional elimination, that we can take a biconditional, a if and only if b,
#and translate that into something like this, a implies b, and b implies a.

#                      Biconditional Elimination

#                              a <-> B
#   --------------------------------------------------------
#                       (a -> B) and (B -> a)

# Many of these Inference Rules are taking logic that uses certain symbols and turning them into different symbols,
#taking an implication and turning it into an or, or taking a biconditional and turning it into implication.

# Another example of it would look something like this.
# It is not true that both Harry and Ron passed the test. How can we translate this?
# Well, if it is true that both of them passed the test, then the reasonable conclusion we might draw is that
#at least one of them didn't pass the test. So the conclusion is either Harry did not pass the test or Ron 
#didn't pass the test, or both.
# If it is true that it is not true that both Harry and Ron passed the test, then either Harry didn't pass
#the test or Ron didn't pass the test. This type of law is one of De Morgan's Laws. This law is famous in logic
#where the idea is that we can turn an and into an or.


#       It is not true that both Harry and Ron passed the test.
#   ----------------------------------------------------------------------
#      Harry did not pass the test or Ron did not pass the test.

# This is how we can frame that more formally using logic.
# If it is not true that alpha and beta, then either not alpha or not beta.
# A way we can think about this is that if we have a negation in front of an and expression, we move
#the negation inwards, moving the negation into each of these individual sentences and then flip the and 
#into an or. So the negation moves inwards and the and flips into an or. So we go from not a annd b
#to not a or not b.

#           De Morgan's Law

#           Not(a and B)
#   -----------------------------
#           Not a and Not B


# There's also a reverse of De Morgan's Law that goes in the other direction and looks something like this.
# If we say it is not true that Harry or Ron passed the test, meaning neither of them passed the test, then 
#the conclusion we can draw is that Harry did not pass the test and Ron did not pass the test. In this case,
#instead of turning an and into an or, we're turning an or into an and. 


#          It is not true that Harry or Ron passed the test.
#   -------------------------------------------------------------
#       Harry did not pass the test and Ron did not pass the test.


# The idea is the same, and is anotherexample of De Morgan's Law. And the way that works is that if
# we have not a or b this time, the same logic is going to apply. We are going to move the negation inwards.
# We are going to flip the or into an and. 
# So if not a or b, meaning it is not true that a or b, then we can say not alpha and not beta, moving the 
#negation inwards in order to make that conclusion.


#             De Morgan's Law

#              Not(a or B)
#   --------------------------------------
#             Not a and Not B

# Those are De Morgan's Laws. 

# These are a couple other Inference Rules that worth taking a look at.

# One is the Distributive Law, that works like this.
# alpha and beta or gamma.Then much in the same way that we can use in math, using distributive laws to 
#distribute operands like addition and multiplication, we can do a similar thing here, where we have
#if alpha and beta or gamma, then we can say something like alpha and beta or alpha and gamma.


#       (a and (B or gamma))
#   -----------------------------------
#       (a and B) or (a and gamma)

# This is an exampe of the distributive property or the distributive law as applied to logic in much the same way
#that we would distribute a multiplication over addition of something, for example.

# This works in reverse also.
# So if, for example, we have alpha or beta and gamma, we can distribute the or throughout the expression.
# We can say alpha or beta and alpha or gamma.


#         (a or (B and gamma))
#   --------------------------------
#       (a or B) and (a or gamma)

# This is helpful if we want to take an or and move it into the expression.

# The question becomes, how can we use these inference rules to actually try and draw some conclusions, 
#to actually try and prove something about entailment, proving that given some initial knowledge base,
#we would like to find some way to prove that a query is true?

# Now that we have these inference rules that take some set of sentences in propositional logic, and get 
#us some new set of sentences in propositional logic, we can actually treat those sets of sentences
#as states inside of a search problem. 
# So if we want to prove that some query is true, or prove that some logical theorem is true, we can
#treat theorem proving as a form of a search problem.

# Theorem Proving -
#   - initial state: starting knowledge base
#   - actions: inference rules
#   - transition model: new knowledge base after inference
#   - goal test: check statement we're trying to prove
#   - path cost function: number of steps in proof

# Here, we have been able to apply the same types of ideas that we saw last time with search problems
#to something like trying to prove something about knowledge by taking our knowledge and framing it in terms 
#that we can understand as a search problem with an initial state, with actions, with a transition model.
# So this, yet again, is a second way, in addition to model checking, to try and prove certain statements
#are true. 

# It turns out there is another way that we can try and apply inference. It is known as resolution.
# Resolution is based on another inference rule that we'll take a look at now. An inference rule
#that will let us prove anything that can be proven about a knowledge base. And it is based on this
#common idea. 
# Let's say we know that either Ron is in the Great Hall, or Hermione is in the library. Let's also
#say we know that Ron is not in the Great Hall. Based on those two pieces of information, what can 
#we conclude? We can pretty easily conclude that Hermione is in the library. 
# We came to this conclusion because of the two statements above the line. These statements that we'll
#call complementary literals, literals that complement each other, they are opposites of each other,
#and seem to conflict each other. One sentence tells us that either Ron is in the Great Hall or Hermione
#is in the library. So if we know that Ron is not in the Great Hall, that conflicts with the first half
#of our first sentence, which means Hermione must be in the library.


#       (Ron is in the Great Hall) or (Hermione is in the library)
#                   (Ron is not in the Great Hall)
#   -------------------------------------------------------------------
#                    (Hermione is in the library) 

# This, we can frame as a more general rule, known as the unit resolution rule, a rule that says if we have 
# p or q, and we also know not p, then we can conclude q. That if p or q are true and we know that p is
#not true, the only possibility is for q to be true.


#               Unit Resolution Rule

#           P or Q
#            Not P
#   -------------------------
#             Q

# This, as it turns out, is quite a powerful inference rule in terms of what it can do, in part because we 
#can quickly start to generalize this rule. The q in our first sentence doesn't need to just be a single
#propositional symbol. It could be multiple, all chained together in a sinlge clause, as we'll call it.
# For example, if we had something like p or q1 or q2 or q3, so on and so forth, up until qn, so we've 
#had n different other variables, and we have not p, well then what happens when these two complement
#each other, is that these two clauses resolve, so to speak, to produce a new clause that is just
#q1 or q2 all the way up to qn.
# And in an or statement, the order of the arguments in the or statement doesn't actually matter.
# The idea here is that if we have p in one clause and not p in the other clause, then we know that
# one of these remaining things must be true.
# We have resolved them in order to produce a new clause.


#       P or Q1 or Q2 or ...or Qn
#                Not P
#   -----------------------------------
#         Q1 or Q2 or ...or Qn

# But it turns out we can generalize this idea even further, in fact, and display even more power that we can
#have with this resolution rule.
# Let's take another example. Let's say that we know the same piece of information that,
#Ron is in the Great Hall or Hermione is in the library. And the second piece of information that we know
#is that Ron is not in the Great Hall or Harry is sleeping. So it's not just a single piece of information.
# We have two different clauses. What do we know here?
# Well again, for any propositional symbol like Ron is in the Great Hall, there are only two possibilites.
# Either Ron is in the Great Hall, in which case, based on resolution, we know that Harry must be sleeping,
#or Ron is not in the Great Hall, in which case we know based on the same rule that Hermione is in the 
#library. Based on those two things in combination, we can say based on these two premises that we can 
#conclude that either Hermione is in the library or Harry is sleeping. 
# Again, because these two conflict with each other, we know that one of these must be true.
# 


#       (Ron is in the Great Hall) or (Hermione is in the library)
#          (Ron is not in the Great Hall) or (Harry is sleeping)
#   --------------------------------------------------------------------
#           (Hermione is in the library) or (Harry is sleeping)

# Stated more generally, we can name this resolution rule by saying that if we know p or q is true,
#and we also know that not p or r is true, we resolve these two clauses together to get a new clause
#q or r, that either q or r is true.

#           P or Q
#         Not P or R
#   -----------------------------
#           Q or R

# And again, much as in the last case, q and r don't need to just be single propositional symbols. It 
#could be multiple symbols.
# So if we had a rule that had p or q1 or q2 or q3, so on and so forth, up until qn, where n is just
#some number. And likewise, we had not p or r1 or r2, so on and so forth, up until rm, where m is 
#just some other number.
# We can resolve these two clauses together to get one of these must be true, q1 or q2 up until qn 
#or r1 or r2 up until rm. This is just a genenralization of the same rule we saw before.

#                P or Q1 or Q2 or ...or Qn
#                Not P or R1 or R2 or ...Rm
#   -------------------------------------------------------
#        Q1 or Q2 or ...or Qn or R1 or R2 or ...or Rm

# Each one of these things here are what we are going to call a clause.

# Clause -
#        - Disjunction of literals 
#        - Disjunction means it's a bunch of things that are connected with or.
#        - Conjunction on the other hand, are things that are connected with and.
#        - Literal is either a propositional symbol or the opposite of a propositional symbol.

# Example of Clause (e.g P or Q or R)

# Meanwhile, what this gives us an ability to do is it gives us the ability to turn logic, any logical sentence, 
#into something called conjunctive normal form. A conjunctive normal form sentence is a logical sentence
#that is a conjunction of clauses.

# Conjunctive normal form -
#                         - Logical sentence that is a conjunction of clauses.
#                         - Conjunction means things that are connected to one another using and.

# A conjunction of clauses means it is an and of individual clauses, each of which has or's in it.

# Conjunctive Normal Form Example (A or B or C) and (D or Not E) and (F or G)
# Everything in parentheses is one clause. All of the clauses are connected to each other using an and.
# And everything in the clause is separated using an or.
# This is a standard form that we can translate a logical sentence into that just makes it easy to 
#work with and easy to manipulate.
# It turns out that we can take any sentence logic and turn it into conjunctive normal form just by
#apllying some inference rules and transformations to it.
# Let's take a look at how we can do that.
# This is the process for taking a logical formula and converting it into conjunctive normal form,
#otherwise known as cnf.
# We need to take all of the symbols that are not part of conjunctive normal form. The bi-conditionals
#and the implications and so forth, and turn them into something that is more closely like
#conjunctive normal form.
# Step one will be to eliminate bi-conditionals, those if and only if double arrows. And we know how
#to eliminate bi-conditionals because we saw there was an inference rue to do just that.
# Any time we have an expression like alpha is and only if beta, we can turn that into alpha implies 
#beta and beta implies alpha based on that inference rule we saw before.
# Likewise, in addition to eliminating bi-conditionals, we can eliminate implications as well, the
#if then arrows. We can do that using the same inference rule we saw before too, taking alpha
#implies beta and turning that into not alpha or beta because that is logically equivalent to this first thing 
#here. Then we can move nots inwards because we don't want nots on the outside of our expressions.
# Conjunctive normal form requires that it's just clause and clause and clause and clause.
# Any nots need to be immediately next to propositional symbols. But we can move those nots around
#using De Morgan's Laws by taking something like not A and B turn it into not A or Not B, for example,
#Using De Morgan's Laws to manipulate that. After that, all we'll be left with are ands and ors.
# Those are easy to deal with using the distributive law to distribute the ors so that the ors end up
#on the inside of the expression, so to speak, and the ands end up on the outside.

# Conversion to CNF -
#                   - Eliminate bi-conditionals
#                       - turn (a <-> B) into (a -> B) and (B -> a)
#                   - Eliminate implications
#                       - turn (a -> B) into Not a or B
#                   - Move Not inwards using De Morgan's Laws
#                       - e.g turn Not(a and B) into Not a or Not B
#                   - Use distributive law to ditribute or wherever possible

# This is the general pattern for how we'll take a formula and convert it into conjunctive normal form.
# This is an example of how we can do this, and why we would want to.

#       (P or Q) implies R  -  implication
#       Not(P or Q) or R  -  Step one - Elimination of Implication
#       (Not P and Not Q) or R -  Step two - De Morgan's Laws
#       (Not P or R) and (Not Q or R) -  Step three - Distributive Law = Conjunctive Normal Form

# This process can be used by any formula to take a logical sentence and turn it into this conjunctive
#normal form, where we have clause and clause and clause and so on. 
# How is this useful to us?
# Because once they're in this form where we have these clauses, these clauses are the inputs to 
#the resolution inference rule that we saw a moment ago, that if we have two clauses where there's
#something that conflicts, or something that complementary between those two clauses, we can resolve them
#to get a new clause, to draw a new conclusion. We call this process inference by resolution, using
#the resolution rule to draw some sort of inference. And it's based on the same idea, that if we have
#p or q, this clause, and we have not p or r, that we can resolve these two clauses together, to get
#q or r as the resulting clause, a new piece of information that we didn't have before.

#  Inference by Resolution

#                P or Q 
#              Not P or R
#   -------------------------------------
#               (Q or R)

# Now, a couple of key points that are worth noting about this before we talk about the actual algorithm.
# One thing is that, let's imagine we have p or q or s, and we also have not p or q or s. The resolution
#rule says that because this p conflicts with this not p, we would resolve toput everything else together
#to get q or s or r or s. But it turns out that s would be redundant. So in resolution, when we do 
#this resolution process, we'll usually do a process known as factoring, where we take any duplicate 
#variables that show up and just eliminate them. So q or s or r or s just becomes q or r or s.


#           P or Q or S
#         Not P or R or S
#   ----------------------------
#          Q or R or S - This conclusion was factored to eliminate the additional S.

# One final question worth considering is what happens if we try to resolve p and not p together?
# If we know that p is true and we know that not p is true, well, resolution says we can merge clauses together
#and look at everything else. Well, in this case, there is nothing else, so we are left with what we might
#call the empty clause. We are left with nothing. And the empty clause is always false. The empty clause
#is equivalent to just being false. That's pretty reasonably becausee it's impossible for both p and 
#not p to both hold at the same time. Since only one can be true, if we try and resolve these two, it's
#a contradiction, and we'll end up getting an empty clause where the empty clause is equivalent to false.

#            P
#          Not P
#   -----------------------------------------------------------------------
#           () - This is an empty clause that is equivalent to false.

# This is the basis for our Inference by Resolution algorithm.
# Here's how we are going to perform inference by resolution at a very high level.
# We want to prove that our kknowledge entails some query alpha, that based on the knowledge
#we have, we can prove conclusively that alpha is going to be true. How are we going to do that,
#we are going to try to prove that if we know the knowledge and not alpha, that that would be a 
#contradiction. 
# This is a common technique in computer science more generally, this idea of proving something
#by contradiction. If we want to prove that something is true, we can do so by first assuming that
#it is false and showing that it would be contradictory, showing that it leads to some contradiction. 
# And if the thing we are trying to prove, if when we assume it's false, leads to a contradiction,
#then it must be true. That's the logical approach or the idea behind a proof by contradiction.
# And that's what we are going to do here.
# We want to prove that this query of alpha is true. So we are going to assume that it's not true.
# We are going to assume not alpha. And we are going to try and prove that it's a contradiction.
# If we do get a contradiction, well, then we know that knowledge entails the query of alpha.
# If we don't get a contradiction, there is no entailment. 
# This is the idea of a proof of contradiction, of assuming the opposite of what we're trying to prove.
# And if we can demonstrate that that's a contradiction, then what we're proving must be true. 


# Inference by Resolution -
#                         - To determine if KB entails query of a:
#                            - Check if (KB and Not a) is a contradiction?
#                               - If so, then KB entails query of a.
#                               - Otherwise, no entailment.


# But more formally, how do we actually do this? How do we check that knowledge base and not alpha
#is going to lead to a contradiction?
# Here is where resolution comes into play. 
# To determine if our knowledge base entails some query alpha, we're going to convert knowledge base 
#not alpha to conjunctive normal form, that form where we have a bunch of clauses that are all "anded"
#together.
# And when we have these individual clauses, now we can keep checking to see if we can use resolution
#to produce a new clause.
# We can take any pair of clauses and check, is there some literal that is the opposite of each other
#or complimentary to each other in both of them?
# For example, we have a p in one clause and a not p in another clause.
# If ever we have that situation where once we convert to conjunctive normal form and we have a bunch
#of clauses, we see teo clauses that we can resolve to produce a new clause, then we'll do so.
# This process occurs in a loop.
# We are going to keep checking to see if we can use resolution to produce a new clause and keep 
#using those new clauses to try to generate more new clauses after that.
# Now, it just so may happen that eventually we may produce the empty clause, and the empty clause
#we know to be false. And if we have a contradiction, that's exactly what we were trying to do in
# a fruit by contradiction.
# If we have a contradiction, then we know that our knowledge base must entail this query alpha. And 
#we know that alpha must be true.
# And it turns out, we can show that otherwise, if we don't produce the empty clause, then there is 
#no entailment.
# If we run into a situation where there are no more new clauses to add, we've done all the resolution
# we can do, and yet we still haven't produced the empty clause, then there is no entailment 
#in this case.
# This is the resolution algorithm.



# Inference by Resolution -
#                         - To determine if KB entails query a:
#                             - Convert (KB and Not a) to Conjunctive Normal Form.
#                             - keep checking to see if we can use resolution
#                               to produce new clauses.
#                                - If ever we produce the empty clause (equivalent to False),
#                                  we have a contradiction, and KB entails query a.
#                                -Otherwise, if we can't add new clauses, no entailment.

# This is an example of an Inference by Resolution

#   Does (A or B) and (Not B or C) and (Not C) entail query A?

# Step one, we will assume that A is False. (A or B) and (Not B or C) and (Not C) and (Not A) - This is
#now in conjunctive normal form, and we have 4 different clauses.

# Step two, now we can start applying the resolution rule to our clauses. (Not B and C) and (Not C)
# turn into (Not B). Then we repeat the process for (A or B) and (Not B) and (Not A). Next
#(A or B) and (Not B) turn into (A). Lastly, we can repeat the process one more time. (A) and (Not A)
#will leave us with an empty clause, which we know to be false. 
# We can safely conclude that our KB (A or B) and (Not B or C) and (Not C), does in fact enatil A.

# Here we will take a look at one final type of logic. A first-order logic. Which is a little more powerful
#than a propositional logic, and is going to make ut easier for us to express certain types of ideas. In 
#first order logic we are going to have two different types of symbols. We're going to have constant
#symbols that are going to represent objects like people or houses. And then predicate symbols, which we can 
#think of as relations or functions that take an input and evaluate them to true or false, for example,
#that tell us whether or not some property of some constant or some pair of constants or multiple constants
#actually holds.


#                First-Order Logic

#   Constant Symbol             Predicate Symbol
#---------------------        ---------------------
#   Minerva                         Person
#   Pomona                          House
#   Horace                          BelongsTo
#   Gilderoy
#   Gryffindor
#   Hufflepuff
#   Ravenclaw
#   Slytherin

# Let's take a look at some examples of what a sentence in first order logic might actually look like.

#               First-Order Logic

#   Person(Minerva) - This sentence in first order logic effectively means Minerva is a person, or the
#person property applies to the Minerva object. So if we want to say something like Minerva is a person
#here is how we express that idea using first order logic.

#   House(Gryffindor) - This is how we would express the idea that Gryffindor is a house.
# And all of the same logical connectives that we saw in propositional logic, those are going to work here
#too. And, Or, Implication, By-conditional, Not. In fact we can use not in our next sentence.

#   Not House(Minerva) - This sentence in first order logic means something like, Minerva is not a house.
#It is not true that the house property applies to Minerva.
# Meanwhile, in addition to some of these predicate symbols that just take a single argument, some of 
#our predicate symbols are going to express binary relations, relations between two of its arguments.

#   BelongsTo(Minerva, Gryffindor) - This is how we would use binary relation to express the idea that
#Minerva belongs to Gryffindor.

# And here is one of the key differences, between this and propositional logic. In propositional logic
#we needed one symbol for Minerva Gryffindor, and one symbol for Minerva Hufflepuff etc... In this case,
#we just need one symbol for each of our people, and one symbol for each of our houses. And Then we
#can express as a predicate something like, belongsto, and say, belongsto minerva gryffindor, to express
#the idea that minerva belongs to house gryffindor house.
# So already we can see that first order logic is quite expressive in being able to express these sorts
#of sentences using the existing constant symbols and predicates that already exist, while minimizing
#the number of new symbols that we need to create. We can just use eight symbols for people for houses,
#instead of 16 symbols forevery possible combination of each.
# But first order logic gives us a couple of additional features that we can express even more complex
#ideas. And these more additional features are generally known as quantifiers. And there are two main
#quantifiers in first order logic, First of which is universal quantification.

#   Universal Quantification - Lets us express an idea like something is going to be true for all values
#of a variablle. Like for all values f x, some statment is going to hold true.
# So what might a sentence in universal quantification look like?
# Well, we're going to use the upside down capital A to mean for all. So upside down capital A(x) means
#for all values of x, where x is any object, this is going to hold true.

# Upside down capial A x BelongsTo(x Gryffindor) implies Not BelongsTo(x Hufflepuff)
# Lets try to parse this out. Translated into english, this sentence is saying something like for all
#objects x, if x belongs to gryffindor, then x does not belong to hufflepuff.
# Or put more simply, anyone in gryffindor is not in hufflepuff.
# So this universal quantification lets us express an idea like something is going to hold true for 
#all values of a particular variable.

# Next, we will look at Existential Quantification. Whereas universal quantification said that something
#is going to be true for all values of a variable, existential quantification says that some expression
#is going to be true for some value of a variable, at least one value of the variable.
# Let's take a look at a sample sentence using existential quantification. One such sentence looks like this.
# There exist an x (backwards E stands for exist) such that house x and belongs to minerva x.

#   Existential Quantification - backwards E x House(x) BelongsTo(Minerva x)
# In other words,there exist some object x where x is a house and minerva belongs to x.
# Put more simply, we're just saying that minerva belongs to a house. 
# There's some object that is a house and minerva belongs to a house.

# And combining this universal and existential quantification, we can create far more sophisticated
#logical statemnets than we were able to just using propositional logic. 
# We could combine these to say something like this. For all x, person x implies there exists a y such
#that house y and belongs to xy.

# Upsidedown capital A x Person(x) -> (backwards E y House(y) and BelongsTo(x, y))

# Let's try and parse this out and understand what is being said.
# Here we are saying that for all values of x, if x is a person, then this is true. So in other words,
#we are saying for all people, and we call that person x, this statement is going to be true. 
# What statement is true for all people? Well there exists a y that is a house, so there exists some 
#house, and x belongs to y. 
# We are saying that for all people out there, there exists some house such that x, the person,
#belongs to y, the house. 
# Simply put, Every person belongs to a house, that for all x, if x is a person, then ther exists a 
#house that x belongs to.