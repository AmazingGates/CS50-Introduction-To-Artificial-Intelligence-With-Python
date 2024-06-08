# Here in our final section we will be going over and understanding Natural Language


# So far in this course we have been taking problems that we want to solve intellegently, and framing them in ways that
#the computer is going to undestand.

# We've been taking problems and framing them as search problems, or constraint satisfaction problems, or optimization
#problems, for example.

# In essence we have been trying to communicate about problems in ways that our computer is going to be able to
#understand.

# Today, the goal is going to be to get computers to understand the way you and I communicate naturally.

# Via our own natural languages, languages like english.

# But natural languages contain a lot of complexity that's going to make it challenging for computers to be able to
#understand, so we'll need to explore some new tools and some new techniques, to allow computers to make sense of
#natural language.

# So what is it exactly that we're trying to get computers to do?

# Well, they all fall under this general heading of natural language processing.

# Getting computers to work with natural language.

# And these task include task like...

#   Natural Language Processsing 

# Automatic Summarization
# Information Extraction
# Machine Translation
# Question Answering
# Text Classification
#...

# And there are several other types of task that all fall under this heading of natural language processing.

# But before we take a look at how the computer might try to solve these types of task, it might be useful for us
#to think about languages in general.

# What are the kinds of challenges that we might need to deal with as we start to think about language and getting a
#computer to be able to understand it.

# So one part of language that we'll need to consider, is the syntax of language.

# Syntax is all about the structure of language.

# Language is composed of individual words, and those words are composed together in some kind of structured whole.

# And if our computer is going to be able to understand language, it's going to need to understand something about
#that structure.

# So let's take a couple of examples.

# Here for instance, is a sentence below...

# "Just before nine o'clock Sherlock Holmes stepped briskly into the room"

# That sentence is made up of words, and those words together form a structured whole.

# This is syntactedly valid as a sentence.

# But we could take some of those same words, rearrange them, and come up with a sentence that is not syntactedly
#valid.

# Let's look at an example of this below.

# "Just before Sherlock Holmes nine o'clock stepped briskly the room"

# It's still composed of valid words, but they're not in any sort of logical form.

# This is not a syntactedly well formed sentence.

# Another interesting challenge, is that some sentences will have multiple possible valid structures.

# Here's an example of that below.

# "I saw the Man on the mountain with a telescope"

# Here, this is a valid sentence, but it actually has two different possible structures, that can be interpreted two
#different ways.

# Maybe I, the one with the telescope doing the seeing, or maybe the Man on the mountain is the one with the telescope.

# And so natural language is ambiguous.

# Sometimes the same sentence can be interpreted in multiple ways, and that's something that we'll need to think about
#as well.

# And this lends itself to another problem within language that we'll need to think about, which is semantics.

# Well syntax is all about the structure of language, semantics is all about the meaning of language.

# It's not enough for a computer just to know that a sentence is well structured if it doesn't know what that sentence
#means.

# And so semantics is going to concern itself with the meaning of words and the meaning of sentences.

# So if we go back to that same sentence as before...

# "Just before nine o'clock Sherlock Holmes stepped briskly into the room"

# We could come up with another sentence, for example...

# "A few minutes before nine, Sherlock Holmes walked quickly into the room"

# And those are two different sentences, with some of the words the same, and some of the words different, but the 
#two sentences have essentially the same meaning.

# And so ideally whatever model we build, will be able to understand that these two sentences, while different, 
#mean something very similar.

# Some syntactedly well formed sentences don't mean anything at all.

# For example...

# "Colorless green ideas sleep furiously"

# This is syntactedly structured sentence, but taken as a whole it doesn't mean anything.

# And so if our computers are going to be able to work with natural language and perform task in natural language
#processing, these are things we will need to take into account.

# We will need to think about syntax and we need to thibk about semantics.

# So how could we go about trying to teach a computer how to understand the structure of natural language?

# One approach we might take is by thinking about the rule of natural language.

# Our natural languages have rules.

# And only if we could formalize those rules, that we could give those rules to a computer, and a computer will be
#able to make sense of them and understand them.

# So let's try to do exactly that. 

# We're going to try to define a formal grammar.

#   Formal Grammar -
# - A system of rules for generating sentences in a language

# This is going to be a rule based approached to natural language processing.

# We are goning to give the computer some rules that we know about language, and have the computer use those rules
#to make sense of the structure of language.

# And there are a number of different types of formal grammar, each one of them has slightly different use cases.

# But today, we're going to focus specifically on kind of grammar, known as a context-free grammar.

# So how does a context-free grammar work?

# Well here is a sentence that we might want a computer to generate.

#   she saw the city

# And we're going to call each of these words a terminal symbol.

# A terminal symbol because once our computer has generated the word, there's nothing else for it to generate.

# Once it has generated the sentence, the computer is done.

# We're going to associate each of the terminal words with a non-terminal symbol, that generates it.

#    N   V   D   N
#   she saw the city

# So here we have N, which stands for Noun, we have V, which stands for Verb, We have D, Which stands for Determiner,
# a determiner is a word like The, A, or And in english, and we have N again, for another Noun.

# So each of these non-terminal symbols can generate the terminal symbol that we ultimately care about generating.

# But how do we know, or how does the computer know which non-terminal symbols are associated with which terminal
#symbols.

# To do that, we need to some kind of rule.

# Here are some what we call, re-writing rules, that have a non-terminal symbol on te left hand side of the arrow,
#and on the right side, is what that non-terminal symbol can be replaced with.

# N -> She   | City | Car | Harry
# D -> The   | A | An | ...
# V -> Saw   | Ate | Walked | ...
# P -> To    | On | Over | ...
# ADJ -> Blue  | Busy | Old } ...

# We can also have non-terminal symbols that are replaced by other non-terminal symbols.

# Here is an interesting rule.

# NP -> N | D N

# So what does this mean?

# NP, stands for a Noun Phrase.

# Sometimes when we have a noun phrase in a sentence, it's not just a single word, it could be multiple words.

# And here we're saying a noun phrase could be just a noun, or, it could be a determiner followed by a noun.

#       NP
#       |
#       N
#       |
#      She

# So we might have a noun phrase that is just a noun, like she, that's a noun phrase.

# Or we could have a noun phrase that is multiple words.

#       NP
#      /  \
#     D    N
#    /      \
#  The      City
#

# Something like, The City, also acts as a noun phrase, but in this case, it's composed of two words, a determiner (The)
#followed by a noun(City).

# We could do the same for Verb Phrases.

# VP -> V | V NP

# A Verb Phrase might be just a Verb, or it might be Verb followed by a NP.

#       VP
#       |
#       V
#       |
#     Walked

# So we could have a verb phrase that's just a single word, like the word walked, or we could have a verb phrase 
#that is an entire phrase.

#           VP
#          /  \
#         V    NP
#        /    /  \
#       |    D    N
#       |    |    |
#      Saw  The  City


# Something like Saw the city, as an entire verb phrase.

# A sentence meanwhile, we might then define as a Noun Phrase, followed by a Verb Phrase.

# S -> NP VP

# This would allow us to generate a sentence like, She saw the city, an entire sentence, made up of a Noun Phrase,
#which is just the word she, and then a Verb Phrase, which is saw the city, saw which is a Verb, and then, the city,
#which itself is also a Noun Phrase.

# And so if we could give these rules to a computer, explaining to it what non-terminal symbols could be replaced
#by what other symbols, then a computer could take a sentence and begin to understand the structure of that sentence.

# So let's take a look at an example of how we might do that.

# And to do that we are going to use a python package library called nlkt, or the Natural Language Tool Kit, which 
#we'll see a couple of times today, it contains a lot of helpful features and functions that we can use for trying 
#to deal with and process natural language.

# Here we'll take a look at how we can use nltk in order to parse a context-free grammar.

# We will start off by importing nltk


from msilib import add_data
import nltk

# Next, we will define a context-free grammar, saying that a Sentence is a Noun Phrase, followed by a Verb Phrase

# Then we will define what a Noun Phrase, then defining what a Verb Phrase is.

# And then giving some examples of what we can do with these non-terminal symbols, D, for determiner, N, for Noun,
#and V, for Verb.

grammar = nltk.CFG.fromstring("""
    S -> NP VP

    NP -> D N | N
    VP -> V | V NP
                              
    D -> "the" | "a" 
    N -> "she" | "city" | "car" 
    V -> "saw" | "walked"
""")

# Next, we're going to use nltk to parse that grammar.

parser = nltk.ChartParser(grammar)

# Then, we'll ask the user for some input, in the form of a sentence, and split it into words.

# And then, we'll fuse this context-free grammar parser, to try and parse that sentence and print out the resulting
#syntax tree.

sentence = input("Sentence: ").split()
try:
    for tree in parser.parse(sentence):
        tree.pretty_print()
except ValueError:
    print("No parse tree possible")

# Let's run our code and take a look at an example.

# This was pretty simple.

# Let's take a look at code with a little more complexity


# Here, our Sentence is still a Noun Phrase followed by a Verb Phrase, but we will add some other non-terminal 
#symbols too.

# We have AP, Adjective Phrase.

# And PP, for Prepersional Phrase.

# And we specified that we could have an Adjective Phrase before a Noun Phrase, or Prepersional Phrase after a Noun, 
#for example.

# So lots of additonal ways that we might try to structure a sentence, and interpret and parse one of those 
#resulting sentences.

# Now let's run this and see what it can handle.

grammar = nltk.CFG.fromstring("""
    S -> NP VP

    AP -> A | A AP
    NP -> N | D NP | AP NP | N PP | NP AP
    PP -> P NP
    VP -> V | V NP | V NP PP

    A -> "big" | "blue" | "small" | "dry" | "wide"
    D -> "the" | "a" | "an" 
    N -> "Mona-Lia" | "Brian" | "city" | "car" | "street" | "dog" | "binoculars" | "bed" 
    P -> "on" | "over" | "before" | "below" | "with" 
    V -> "saw" | "walked" | "kissed" | "drained" 
""")

parser = nltk.ChartParser(grammar)

sentence = input("Sentence: ").split()
try:
    for tree in parser.parse(sentence):
        tree.pretty_print()
except ValueError:
    print("No parse tree possible")


# To try a more complex sentence, we will run the sentence, "Mona-Lia saw the dog with the binoculars".

# Notice that we are returned two possible parse trees for this sentence.

# Notice that the first sentence is a little ambiguous in our own natural langauge.

# Who has the binoculars?

# Is it she who has the binoculars, or the dog who has the binoculars?

# And nltk is able to identify both possible structures as stated.

# In the second tree, "the dog with the binoculars" is an entire Noun phrase.

# This indicates that it's the dog that has the binoculars.

# In the first parse tree, "the dog", is just the Noun Phrase, and "with the binoculars" is a Prepositional Phrase
#modifying "saw".

# So "Mona-Lia saw the dog", and "Mona-Lia", was using the binoculars. in order to see "the dog" as well.

# This allows us to get a sense for the structure of natural languge, but it relies on us writing all of these rules,
#and it would take a lot of effort to write all of the rules for any possible sentence that someone might write or
#say in the english language. 

# Language is complicated and as a result there are going to be some very complex rules. 

# So what else might we try?

# We might, try to take a statistical lense, towards approaching this problem of natural language processing.

# If we were able to give the computer a lot of existing data of sentences written in the english language, what
#could we try to learn from that data.

# Well it might be difficult to try and interpret long pieces of text all at once, so instead what we might want to do
#is break up that longer text into smaller pieces of information instead.

# In particular, we might try to create n-gram.

#   n-gram -
# - A continuous sequence of n-items from a sample of text

# It might be n characters in a row, or n words in a row for example.

# So let's take a passage from Sherlock Homles, and let's look for all of the trigrams.

# A trigram is an n gram where n is equal to 3.

# So in this case we are looking for sequrnces of 3 words in a row.

# "How often have I said to you that when you have eliminated the impossible whatever remains, however improbable, must
#be the truth?"

# So the trigrams here would be phrases like, "How often have", that's 3 words in a row, "often have I", is another,
# "have I said" is also a trigram, "I said to", "said to you", "to you that", these are all trigrams, sequences of
#3 words that appear in sequence.

# And if we could give the computer a large corpus of text, and have it pull out all of the trigrams in this case,
#we could get a sense for what sequences of 3 words tend to appear next to each other in our own natural language,
#and as a result, get some sense for what the structure of the language actually is.

# So let's take a look at an example of that.

# How can we use nltk to try to get access to information about n grams.

# First we will bring over some more imports.

# We also need to import nltk but we have already done that so we don't have to again.

from collections import Counter

import math
import os
import sys



# This is a python program that is going to load a corpus of data, just some text files into our computers memory,

def main():
    "Calculate top term frequencies for corpus of documents"

    if len(sys.argv) !=3:
        sys.exit("Usage: python ngrams.py n corpus")
    print("Loadin Data...")

    n = int(sys.argv[1])
    corpus = load_data(sys.argv[2])

    # And then we're going to use nltk ngrams function which is going to go through the corpus of text, pulling out all
    #of the ngrams for a particular value of n.

    # Compute n-grams
    ngrams = Counter(nltk.ngrams(corpus, n))


    # And then, by using pythons counter class, we're going to figure out what are the most common ngrams inside of 
    #this entire corpus of text.

    # Print most common n-grams
    for ngram, freq in ngrams.most_common(10):
        print(f"{freq}: {ngram}")

def load_data(directory):
    directory = ("banknotes.csv") 


# We were able to do this with a process called tokenization

#   Tokenization -
# - The task of splitting a sequence of characters into pieces (tokens)

# In this case we're splitting a long sequence of text into individual words and then looking at sequences of those
#words to get a sense for the structure of natural language.

# So once we've done the tokenization, once we've built up our corpus of ngrams, what can we do with that information?

# Well, one thing that we might try, is we could build a markov chain.

# Whuch we might recall from when we talked about propability.

# Recall that a markov chain is some sequence of values, where we can predict one value, based on the values 
#that came before it.

# And as a result, if we know what all of the common ngrams in the english language, what words tend to be
#associated with what other words in sequence, we can use that to predict what word might come next in a sequence
#of words.

# And so we could build a markov chain for language in order to try and generate natural language that follows the
#same statistical pattern as some input data.

# So let's take a look at that, and build a markov chain for natural language.

# And as input, we're going to use the works of William Shakespeare.

# Well use this code to generate our patterns.

import markovify # Have to pip install
import sys


# We're going to read in the sample of text

# Read text from file 
if len(sys.argv) !=2:
    sys.exit("Usage: python generator.py sample.txt")
with open(sys.argv[1]) as f:
    text = f.read()
    

# Then we're going to train a markov model based on that text

# Train model
text_model = markovify.Text(text)


# And then we're going to have the markov chain generate some sentences. We're going to generate a sentence that 
#doesn't appear in the original text but follows the same statistical patterns.

# Generating it based on the ngrams trying to predict what word is likely to come next that we would expect based
#on those statistical patterns.

# Generate sentence
print()
for i in range(5):
    print(text_model.make_sentence())
    print()


# What we're going to get are 5 new sentences, where these sentences are not necessarily sentences from the original
#input text themself,  but just follow the same statistical pattern.

# It's predicting what word is likely to come next, based on the input data that we seen and the types of words that 
#tend to appear in sequence there too.

# Of course, so far there is no ganruntee that the sentences generated actually mean anything, or make any sense,
#they just happen to follow the statistical pattern that our computer is already aware of.

# We'll return to this issue of how to generate text in a more accurate and meaningful way a little bit later.

# So let's turn out attention to a slightly different problem, and that's the problem of text classification.

# Text classification is the problem when we have some text, and we want to put that text into some sort of category.

# We want to apply some sort of label to that text.

# And this kind of problem shows up in a wide variety of places.

# A common place might be our email inbox, for example.

# We get an email, and we want our computer to be able to identify whether the email belongs in our inbox, or whether
#it should be filtered out into spam.

# So we need to classify the text.

# Is it a good email, or is it spam.

# Another common usecase, is sentiment analysis.

# We might want to know whether the sentiment of some text is positive or negative.

# And so how might we do that?

# This comes up in situations like product review, where we might have a bunch of reviews for a product on some 
#website. "The Gates Family really liked the service they got on the Disney Cruise, it's all they talked about for 
#a month straight, we're looking forward to going next year.", "Bam Bam and Itachi, my Sons, hated the burgers from 
#Dairy Queen.", "My daughter Amelia loved her birthday tiara from Amazon.", "I fell in love with the 357 Magnum Smith 
#and Wesson that my Wife got me for christmas, Cabela's is the best!.", My Wife, Mona Lia, didn't like the waiteress
#we got at Texas Roadhouse, she said that her work shirt was way to tight, so we left. I think the waitress was doing a 
#fine jugs, I mean job."

# These are example sentences we might see on a product review website.

# We could pretty easily look at the reviews and decide, which ones are positive, and which ones are negative.

# But how do we know that, and how can we train a computer to be able to figure that out as well. 

# Well, we might have caught sight of particular key words, and those particular words tend to mean something 
#positive or negative.

# So we might have caught sight of words like "Loved", "Liked", and "Best", that tend to be associated with positive 
#messages, and words like "Hated", " So We Left" and "Didn't Like", tend to be associated with negative messages.

# So if only we could train a computer to be able to learn what words tend to be associated with positive vs negative
#messages, then maybe we could train a computer to do this kind of sentiment analysis as well.

# So we're going to try to do just that.

# We're going to use a model known as bag-of-words model

#   Bag-Of-Words -
# - Model that represents text as an unordered collection of words.

# For the purpose of this model, we're not going to worry about the sequence and the ordering of the words, which word
#came first, second or third we're just going to treat the text as a collection of words in no particular order.

# And we're losing information there, because the order of words is important, and we'll get back to that a little
#bit later.

# But for now, to simplify our model, it will help us tremendosly just to think about text as some unordered
#collection of words.

# And in particular, we're going to use the bag of words model to build something known as a naive bayes classifier.

# So what is a navie bayes classifier?

# It's a tool that is going to allow us to classify text based on bayes rule.


#   Baye's Rule


#
#            P(a|b) P(b)
# P(b|a) = ----------------
#                P(a)
#

# We might remember this from when we talked about probability.

# Baye's rule says that the probability of b given a, is equal to the probability of a given b, multiplied by the 
#probability of b, divided by the probability of a.

# So how are we going to use this rule to be able to analyze text?

# Well, what are we interested in?

# We're intereested in the probability that a message as a positive sentiment, and the probability that a message has
#a negative sentiment.

# Which for simplicity, we're just going to represent as P and N. p(P) means positive, and p(N) means negative.

# And so, if we had a review, something like, "The Gates Family really liked the service they got on the Disney Cruise, 
#it's all they talked about for a month straight, we're looking forward to going next year.", then what we're interested 
#in is not just the probability that a message has a positive sentiment, but the conditional probability that a message 
#has positive sentiment, given that this is the message.

# P(P|"The Gates Family really liked the service they got on the Disney Cruise, it's all they talked about for 
#a month straight, we're looking forward to going next year.")

# But how do we go about calculating this value?

# The probability that the message is positive given that the review is this sequence of words.

# Well here is where the bag of words model comes in.

# Rather than treat this review as a string of a sequence of words in order, we're just going to treat it as an
#unordered collection of words.

# We're going to try to calculate the probability that the review is positive, given that all of the words in our
#review, are in the rewview, in no particular order, just an unoredered collection of words.

# Basically saying that we don't really care what order the words are in, just as long as all of the words in the 
#review are there.

# And this, is a conditional probability, which we can then apply baye's rule to, to try and make sense of.

# So according to baye's rule, this conditional probability is equal to the probability that all of the words were in
#the review, given the review is positive, multiplied by the probability that the review is positive, divided by
#the probability that all of these words happen to be in the review.


# P(P|"The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year")

#                                     equal to

# P(P|"The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year"|P) P(P)
#---------------------------------------------------------------------------------------------------------------------
# # P("The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year")

# This is the value that we're going to try and calculate.

# One thing we might notice is that the deniminator, the proabability that all the words appear in the review, doesn't
#actually depend on whether or not we're looking at the positive sentiment or negative sentiment case.

# So we could actually get rid of the denominator because we don't need to calculate it.

# We could just say that the first probability (see line 657) is propoetional to the numerator (see line 663)

# And in the end, we're going to need to normalize the probability distribution to make sure that all of the values
#sum up to the value one.

# So now how do we calculate this value?

# This is the probability of all of these words, given positive, times the probability of positive. 

# And that by the definition of joint probability, is just one big joint probability, the probability that all of 
#these things are the case, that it's a positive review, and all the words are in the review.

# But still, it's not entirely obvious how we calculate that value.

# And here is where we need to make one more assumption.

# This is where the naive part of naive bayes comes in.

# We're going to make the assumption that all of the words are independent of each other. 

# And by that we mean that if the word "Gates" is in the review, that doesn't change the probability that the word 
#"family" is in the review, or the word "Disney" is in the review, for example.

# And in practice, this assumption might not be true, it's almost certainly the case that the probbility of words
#do depend on each other.

# But it's going to simplify our analysis and still give us reasonably good results, just to assume that the words 
#are independent of each other, and they only depend on whether it's positive or negative.

# We might for example expect the word "liked" to appear more often in a positive review than in a negative review.

# So what does that mean?

# Well if we make this assumption, then we can say that this value, 

# P(P|"The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year")

#the probability that we're interested in, is not directly proportional to, but it's naively proportional to this value,

# # P(P, "The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year").


# Meaning that it's not exactly this.

# P(P|"The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year"|P) P(P)
#
#                                         Proportional To
#
# P(P, "The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year")


# But more like this.

# P(P|"The", "Gates", "Family", "really", "liked", "the", "service", "they", "got", "on", "the", "Disney", "Cruise", 
#"it's", "all", "they", "talked", "about", "for", "a", "month", "straight", "we're", "looking", "forward", 
#"to", "going", "next", "year"|P) P(P)
#
#                              Naively Proportional To
#
# # P(P)P("The"|P), P("Gates"|P), P("Family"|P), P("really"|P), P("liked"|P), P("the"|P), P("service"|P), P("they"|P),
#P("got"|P), P("on"|P), P("the"|P), P("Disney"|P), P("Cruise"|P), P("it's"|P), P("all"|P), P("they"|P), P("talked"|P),
#P("about"|P), P("for"|P), P("a"|P), P("month"|P), P("straight"|P), P("we're"|P), P("looking"|P), P("forward"|P), 
#P("to"|P), P("going"|P), P("next"|P), P("year"|P)).

# The probability that the review is positive, Times the probability that "Gates" is in the review, given that it's
#positive, times the probability the that "family" is in the review, given that it's positive, and so on and so forth.

# And now this value which looks a little more complex is actually a value that we can calculate pretty easily.

# So how are we going to estimate the probability that the review is positive?

# Well if we have some training data, some example data, of example reviews, where each one has already been labeled
#as positive or negative, then we can estimate the probability that our review is positive just by counting the number 
#of positive samples, and dividing by the total number of samples that we have in our training data.

#         Number of positive samples
# P(P) = -----------------------------
#          Number of total samples

# And for the conditional probability, the probability of liked, given that's it's positive, well that's going to be 
#the number of samples with liked in it, divided by the total number of positive samples.

#                   Number of positive samples with liked
# P("liked"|P) = -------------------------------------------
#                        Number of positive samples

# So let's take a look at an actual example, to see how we could try and calculate these values.

# # P(P)P("The"|P), P("Gates"|P), P("Family"|P), P("really"|P), P("liked"|P), P("the"|P), P("service"|P), P("they"|P),
#P("got"|P), P("on"|P), P("the"|P), P("Disney"|P), P("Cruise"|P), P("it's"|P), P("all"|P), P("they"|P), P("talked"|P),
#P("about"|P), P("for"|P), P("a"|P), P("month"|P), P("straight"|P), P("we're"|P), P("looking"|P), P("forward"|P), 
#P("to"|P), P("going"|P), P("next"|P), P("year"|P)).

# Note: For now we will imagine that our review says, "My Wife loved it", to follow along with the instructor.

# Here we have some sample data, based on the review we're following along with.

# The way to interpret the sample data is that based on the sample data, 49% of the reviews are positive, and 51% are 
#negative.

# And in our second table, we have some conditional probabilities.

# We have, if the review is positive, then there is a 30% chance that the word "my" appears in it.

# And if the review is negative, there is a 20% chance that the word "my" is in it.

# And based on our training data, among the positive reviews, 1% of them contain the word "Wife".

# And among the negative reviews, 2% contain the word "Wife".

# so, using this data, let's try and calculate the value of the review sentence that we're following along with, the
#value that we're interested in.

# To do that, we'll need to multiple all of these values together.

# The probability of positive, and then all of the positive conditional probabilities.

# And when we do that, we get some value. (P 0.00014112)

# And then we can do the samething for the negative case.

# We will get some other value for that as well. (N 0.00006528)

# 

#   _________________           ________________________________________
#  |____P____|___N___|         |___________|______P_______|______N______|   
#  |___0.49__|__0.51_|         |_____My____|____0.30______|____0.20_____|
#                              |___Wife___|_____0.01_____|_____0.02____|
#                              |___Loved___|_____0.32_____|_____0.08____|
#                              |____it_____|______0.30____|_____0.40____|


# (P 0.00014112)

# (N 0.00006528)

# Now these values don't sum to 1, they're not a probability distribution yet, but we can normalize them and get some
#values.

# P 0.6837

# N 0.3163

# And that tells us that we're going to predict that "My Wife Loved it", we think that there is a 68% chance that that
#is a positive sentiment review.

# And a 32% chance that it's a negative review.

# So, what problems might we run into here.

# What could potentially go wrong when doing this kind of analysis in order to analyze what text is a positive or
#a negative sentiment.

# Well, a couple of problems might arise.

# One problem might be what if the word "Wife" never appears for any of the positive reviews.

# If that were the case, then when we try to calculate the value, the probability that we think the review is 
#positive, we're going to multiple all these values together, and were just going to get 0 for the positive case.

# That's because were going to multiple by that 0 value.

# And so we're going to say that we think there is no chance that the review is positive because it contains the
#word "Wife", and in our training data we've never seen the word "Wife" appear in a positive sentiment message 
#before.

# And that's probably not the right analysis.

# Because in cases of rare words, it might be the case that in no where in our training data did we ever see the word
#"Wife" appear in a message that has a positive sentiment.

# So what can we do to solve this problem?

# Well one thing we'll often do is some kind of additive smoothing, where we add some value (alpha) to each value in
#our distribution just to smooth out the data a little bit.

#   Additive smoothing -
# - Adding a value (alpha) to each value in our distribution to smooth the data

# And a common form of this is Laplace Smoothing

#   Laplace Smoothing -
# - Adding 1 to each value in our distribution: pretending we've seen each value one more time than we actually have.

# If we've never seen the word "Wife" for a positive review, we pretend we've seen it once.

# That if we've seen it once, we pretend we've seen it twice, just to avoid the possibility that we might multiple 
#by 0, and as a result get some results we don't want in our analysis.

# So let's see what this looks like in practice.

# Let's try to do some naive bayes classification in order to classify text as either positive or negative.

# First we'll get our imports 

import nltk
import os
import sys


#def main():

    # We're going to load some sample data into memory. 

    # Read data from files
#    if len(sys.argv) !=2:
#        sys.exit("Usage: python sentiment.py corpus")
#    positives, negatives = load_data(sys.argv[1]) 


    # Some exmples of positive and negative reviews.

    # Create a set of all words
#    words = set()
#    for document in positives:
#        words.update(document)
#    for document in negatives:  
#        words.update(document)  


    # Then we're going to train a Naive Bayes Classifier on all of this training data.

    # Training data that includes all of the words we see in positive reviews, and all of the words we see in negative 
    #reviews.

    # Extract features from text
#    training = []
#    training.extend(generate_features(positives, words, "Positive"))
#    training.extend(generate_features(negatives, words, "negative"))


    # And then we're going to try to classify some input.

    # We're goinng to do this based on a corpus of data

    # Classify a new sample
#    Classifier = nltk.NaiveBayesClassifier.train(training)
#    s = input("s: ")
#    result = (classify(Classifier, s, words)) 
#    for key in result.samples():
#        print(f"{key}: {results.prob(key):.4f}")

#def extract_words(document):
#    return set(
        
#    )


# And so, this definitely was a useful tool that we can use to try and make some predictions, but we had to make some
#assumptions in order to get there.

# So what if we wanted to build some more sophiticated models.

# Use some tools from machine learning to try and take better advantage of language data to be able to draw more
#accurate conclusions and solve new types of task and new types of problems.

# Well,we've seen a couple of times now, that when we want to take some data take some input, put it in a way that 
#the computer is going to make sense of, it can be helpful to take that data and turn it into numbers ultimately.

# And so what we might want to try and do is come up with some word representation.

# Some way to take a word and translate its meaning into numbers, because for example if we wanted to use a neural
#network, to be able to process language, give our language to a neural network and have it make some predictions
#and form some analysis there.

# A neural network takes as input and produces as output, a vector of values, a vector of numbers.

# And so what we might want to do, is take our data and some how take words and somehow convert them into numerical
#representation.

# So how might we do that?

# How might we take words and turn them into numbers?

# Let's take a look at an example.

# Here's a sentence here.

# "He wrote a book"

# He [1, 0, 0, 0]
# Wrote [0, 1, 0, 0]
# A [0, 0, 1, 0]
# Book [0, 0, 0, 1]

# Let's say we want to take each of those words and turn it into a vector of values.

# Here's one way we might do that.

# We'll say that "He" is going to be a vector that has a 1 in the first position, and the rest of the values are 0.

# "Wrote" has a 1 in the second position, and the rest of the values are 0.

# "A" has a 1 in the third position, and the rest of the values are 0.

# And "Book" has a 1 in the fourth position, and the rest of the values are 0.

# So each of these words now has a distinct vector representation.

# And this is what we often call a one-hot representation.

#   One-Hot Representation -
# - Representation of meaning, of a word, as a vector with a single 1, and with other values as 0.

# And so when doing this we now have a numeric representation for every word, and we can pass in those vector
#reprsentations into a neural network or other models that require numeric data as input.

# But this one hot representation actually has a couple of problems, and it's not ideal for a few reasons.

# One reason is, here we're just looking forwards, but if we imagine we have a vocabulary of thousands of words or
#more, these vectors are going to get quite long in order to have a distinct vector for every possible word in our
#vocabulary.

# And as a result of that, these longer vectors are gonna be more difficult to deal with, more difficult to train, and
#so on and so forth, and so that might be a problem.

# Another problem is a liitle bit more subtle.

# If we want to represent a word as a vector and in particular, the meaning of a word, as a vector, then ideally 
#it should be the case that words that have similar meanings, should also have similar vector representations, so that
#their close to each other inside a vector space.

# But that's not really going to be the case with these one hot representations, because if we take some similar
#word, let's say the word "Wrote", and the word "Author", which mean similar things, but they have entirely different
#vector representations.

# Likewise, book and novel. 

# Those two words mean somewhat similar things, but they have entirely different vector representations.

# This is because they each have a 1 in some different position.

# And so that's not ideal either.

# So what we might be interested in instead, is some of distributive representation.

#   Distributive Representation -
#- Representation of meaning of a word distributed across multiple values, instead of just being a One-Hot with a
#one in one position.

# Here is what a distributive representation of words might be.

# Each word is associated with some vector of values with the meaniing distributed across multiple values, ideally
#in such a way that similar words have a similar vector representation.

# "He wrote a book"

# He [-0.34, -0.08, 0.02, -0.18, 0.22, ...]
# Wrote [-0.27, 0.40, 0.00, -0.65, -0.15, ...]
# A [-0.12, -0.25, 0.29, -0.09, 0.40, ...]
# Book [-0.23, -0.16, -0.05, -0.57, 0.05, ...]

# But how are we going to come up with these values.

# Where do these values come from.

# How can we define the meaning of a word in this distributed sequence of numbers.

# To do that, we're going to draw inspiration from a quote from British linguist J.R Firth, who said "You shall know
#a word by the company it keeps".

# In other words we're going to define the meaning of a word based on the words that appear around it. The context
#words around it.

# Take for example, this context. For blank he ate.

# [For |    | he |  ate |]

# We might wonder what words could reasonably fill in that blank.

# Well it might be words like "breakfast", or "lunch", or "dinner", all of those reasonably fill in that blank.

# So what we're going to say is because the words breakfast lunch and dinner appear i a similar context, that 
#they must have a similar meaning.

# And that's something a computer can understand and try to learn.  

# A computer could look at a big corpus of text, look at what words tend to appear in similar context of each other, 
#and use that to identify which words have a similar meaning and should therefore appear closer to each other inside
#a vector space.

# And so, one commmon model for doing this is known as the word2vec model.

#   Word2Vec -
# - Model for generating word vectors

# It's a model for generating word vectors, a vector representation for every word, by lookng at data, and looking
#at the context in which a word appears.

#                   .book
#                               .memoir
#      .breakfast              
#                              .lunch
#     .dinner
#                                       .novel

# The idea is going to be this, if we start out with all of the words just in some random positional space, and try 
#it on some training data, what the word2vec model will do is start to learn what words appear in similar context
#and will move these vectors around in such a way that hopefully the words with similar meanings, "breakfast, lunch
#dinner", and "book, memoir, novel", will hopefully appear to being near to each other as vectors as well.

#                             .memoir
#                          .book
#  .breakfast
#                        .novel
#      .lunch
#  .dinner

# One other interesting side effect of word2vec is that its also able to capture something about the relationship
#between words as well. 

# Let's take a look at an example.

#           .King



# .Man

# Here for instance are two words, Man and King.

# And these are each represented by word2vec as vectors.

# So what might happen if we subtracted one from the other?

# Calculated the value King minus Man.


#             .King
#            /
#  King-Man /
#          /
#        .Man

# Well that would be the vector that will take us from Man to King, somehow represent this relationship between
#the vector representation for the word Man, and the vector representaion for the word King.

# And that's what the value, King minus Man represents.

# Now, what would happen if we took the vector representation of the word Woman, and added that same value, King 
#minus Man,to it.

# What would we get as the closest word to that, for example.

#             
#            /
#  King-Man /
#          /
#        .Woman

# When we calcuate this value, we get back the word "Queen".

# So that is the closest word to the vector representation of the word King, minus the representation of the word Man, 
#plus, the representation of the word Woman.

#             .Queen
#            /
#  King-Man /
#          /
#        .Woman

# We've somehow been able to capture the relationship between King and Man, and when we apply it to the word Woman,
#we get as a result, the word Queen.

# So word2vec has been able to capture not just the words, and how they're similar to eachother, but also something about the
#relationship words and how those words are connected to each other.

# So now that we have this vector representation of words, what can we now do with it.

# Now we can represent words with numbers, and so we might try to pass those words as inputs, to say, a neural network.

# Neural Networks we've seen are very powerful tools for identifying patterns and making predictions.

# Recall that a neural network we can think of as a bunch of units, but really what it's doing is taking some input,
#passing it into the network, and then producing some output.

#[  input --->  network --->  output]

# And by providing the neural network with training data, we're able to update those weights inside of the network
#so that the neural network can do a more accurate job of translating those inputs into those outputs.

# And now that we can represent words as numbers, that could be the input or the output, we could imagine passing a word in
#in as input to a neural network, and getting a word out.

# And so when might that be useful?

# One common use for neural networks is in machine translation.

# When we want to translate text from one language into another.

# Say translate english into french, for example, by passing english as an input into the neural network and getting
#french as an output.

# We might imagine for instance that we could take the english word for lamp, pass it into the neural network, and the
#the french word lampe as output.

#[  lamp --->  network --->  lampe]

# But in practice, when we're translating text from one language to another, we're usually not just interested in
#translating a single word from one language to another, but a sequence, say a sentence, or a paragraph of words.

# Here for example is another paragraph again taking from Sherlock Holmes, written in english.

# And what we might want to do is take that entire sentence, pass it into the neural network, and get as output, a 
#french translation of the same sentence.

# The only light in the room                                         La piece n etait eclairee
#came from the lamp upon the        --->    Neural Network --->      que par la lampe placee sur
#table at which I had been reading.                                  la table ou je lisais.

# But recall that a neural networks input and output needs to be of some fixed size.

# And a sentence is not of a fixed size, it varys.

# We might have shorter sentences, we might have longer sentences.

# So somehow we need to solve the problem of translating a sequence into another sequence, by means of a neural
#network.

# And that's going to be true, not only for machine translation, but also for other problems.

# Problems like question answer.

# If we want to pass as input a question, something like the example below, feed that as input into the neural network,
#we would hope that what we would get as output is a sentence like the example below.

# What is the                                   The Capital
#capial of      --->    Neural Network   --->    is Boston
#massachusetts

# Again, we've translated some sequence into some other sequence.

# And if we've ever had a conversation with AI chat bot, or have ever asked our phone a question, it needs to do 
#something like this, it needs to understand the sequence of words that us the human provided as input, and then 
#the computer needs to generate some sequence of words to output.

# So how can we do this?

# Well one tool that we can use is the recurrent neural network, which we took a look at last time, which is a way
#for us to provide a sequence of values to a neural network by running the neural network multiple times, and each
#time we run the neural network, what we're going to do is we're going to keep track of some hidden state, and 
#that hidden state is going to passed from one run of the neural network to the next run of the neural network,
#keeping track of all the relevent information.

# And so let's take a look at how we could apply that to something like this, and in particular, we're going to
#look at an architexture known as a encoder-decoder architexture, where we're going to encode the question into
#some kind of hidden state, and then use a decoder to decode that hidden state into the output that we're 
#interested in.

# So what's that going to look like?

# We'll start with the first word, the word what.

# That goes into our neural network, and it's going to produce some hidden state.

# This is some information about the word what, that our neural network is going to need to keeo track of.

# Then when the second word comes along, we're going to feed it into that same encoder neural network, but it's
#going to get as input that hidden state as well.

# So we pass in the second word, we also get the information about the hidden state, and that's going to continue
#for the other words in the input.

# This is going to produce a new hidden state.

# So when we get to third word, the, that goes into the encoder and also gets access to the hidden state, and
#then it produces a new hidden state that gets passed into the next run where we use the word capital, and the
#same thing is going to repeat for the other words that appear in the input.

# So, of is next, and then massachusetts, which produces one final piece of hidden state.

# Now, somehow we need to signal the fact that we're done, there's nothing left in the inputs.

# And we typically do this by passing some kind of special token, like an end token, into the neural network,
#and now the decoding process is going to start.

# We're going to generate the word The. 

# But in addition to generating the word The, this decoder network is also going to generate some kind of hidden
#state.

# And so what happens the next time?

# Well to generate the next word, it might be helpful to know what the first word was.

# So we might pass the first word The back into the decoder network, it's going to get as input the hidden state, 
#then it's going to generate the next word capital, and that's also going to generate some hidden state.
 
# And we'll repeat that passing capital into the network to generate the word is.

# And then one more time in order to get the fourth word Boston.

# And at that point we're done.

# But how do we know we're done.

# Ussually we'll do this one more time, pass boston into the decoder network and get an output, some end token
#to indicate that is the end of our input.



# [What       --->    neural network]
#                           |
#                     hidden state
#                           |
# [is         --->    neural network]
#                           |
#                     hidden state
#                           |
# [the        --->    neural network]
#                           |
#                     hidden state
#                           |
# [capital    --->    neural network]
#                           |
#                     hidden state
#                           |
# [of         --->     neural network]
#                           |
#                     hidden state
#                           |
# [Massachusetts ---> neural network]
#                           |
#                     hidden state
#                           |
# [<end>      --->    neural network]   --->    The
#                           |
#                     hidden state
#                           |
# [The        --->    neural network]   --->    capital
#                           |
#                     hidden state
#                           |
# [capital    --->    neural network]   --->    is
#                           |
#                     hidden state
#                           |
# [is         --->    neural network]   --->    Boston
#                           |
#                     hidden state
#                           |
# [Boston     --->    neural network]   --->    <end>


# And so this then is how we can use a recurrent neural network to take some input, encode it into some hidden state,
#and then use that hidden state to decode it into the output we're interested in.

# To visualize it in a slightly different way, we have some input sequence.

# This is just some sequence of words.

# That input sequence goes into the encoder, which in this case is a recurrent neural network, generating these
#hidden states along the way, until we generate some final hidden state, at which point we start to decoding
#process, again, using a recurrent neural network that's going to generate the output sequence as well.

# So we've got the encoder, which is encoding the information about the input sequence into the hidden state, and
#then the decoder, which takes that hidden state and uses it in order to generate the output sequence.

#
#                  Output Sequence   <end>
#                        ____ ____ 
#                       |    |    |    |
#                       |    |    |    |
#  []-|>[]-|>[]-|>[]-|>[]-|>[]-|>[]-|>[]
#   |    |    |    |    |              |
#   |____|____|____|    |              |
#    Input Sequence   <end>          <end>


# But there are some problems, and for many years this was the state of the art, the recurrrent neural network and variants
#on this approach, which were some of the best ways we knew in order to perform task in natural language processing.

# But there are some problems that we might want to try and deal with, and have been dealt with over the years
#to try and improve upon this kind of model.

# And one problem that we might notice happens in the encoder stage.

# We've taken this input sequence, the sequence of words, and encoded it all into the finalpiece of hidden state.

# And that final piece of hidden state needs to contain all of the information from the input sequence that we 
#need in order to generate the output sequence.

# And while that's possible, it becomes increasingly difficult as the sequence gets larger.

# For larger and larger input sequences it's going to become more difficult to store all of the information we need
#about the input inside that single hidden state piece of context.

# That's a lot of information to pack into just a single value.

# It might be useful for us, when generating output, to not just refer to that one value, but to all of the
#previous hidden values generated by the encoder.

# That might be useful, but how can we do that?

# We have a lot of different values we need to combine some how.

# So we can imagine adding them together, taking the average of them for example, but doing that would assume that
#all of these pieces of hidden state are equally important, but that's not the necessarily true either.

# Some of those pieces of hidden state are going to be more important than others depending on what word they most
#closely correspond to.

# The first hidden state most closely corresponds to the first word of the input sequence.

# The second hidden state most closely corresponds to the second word of the input sequence, for example, and some 
#of those are going to be more important than others.

# To make matters more complicated, depending on which word of the output sequence we're generating, different inut 
#words might be more or less important.

# And so what we really want is some way to decide for ourselves which of the input values are worth paying 
#attention to, at what point in time.

# And this is the key idea behind a mechanism known as attention.

# Attention is all about letting us decide which values are important to pay attention to when generating, in this 
#case, the next word in our sequence.

# so let's take a look at an example of that.

# Here's a sentence, "What is the capital of massachusettes", same sentence as before. 

# [What][is][the][capital][of][Massachusetts]

# And let's imagine that we were trying to answer that question by generating tokens of output.

# So what would that output look like.

# Well it's going to look something like, "The capital is", and let's say we're now trying to generate the last
#sentence.

# [The][capital][is][   ]

# What is that last word, how is the computer going to figure it out.

# Well what it's going to need to do is decide, which values it's going to pay attention to.

# And so the attention mechanism will allow us to calculate some attention scores for each word.

# Some value, corresponding to each word, determining how relevent is it for us to pay attention to that word
#right now.

#                                    |
#                    |               |
#                    |               |
#                    |               |
#    |               |               |
#    |               |      |        |
#    |    |          |      |        |
#    |    |   |      |      |        |
# [What][is][the][capital][of][Massachusetts]

# And in this case, when generating the fourth word of the output sequence, the most important words to pay
#attention to might be "capital" and "massachusttes", for example.

# Those words are going to be particularly relevent.

# And there a number of different mechanisms that have been used in order to calculate these attention scores.

# It could be something a simple as a dot product to see how similar two vectors are, or we can train an entire
#neural network to calculate these attention scores.

# But the key idea, is that during the training process for our neural network, we're going to learn how to
#calculate these attention scores.

# Our model is going to learn what is important to pay attention to in order to decide what the next word should
#be.

# So the result of all of this, calclating the attention scores, is that we can calculate some value for each 
#input word, determining how important is it for us to pay attention to that particular value.

#  0.04 0.02 0.01  0.28   0.03     0.54
# [What][is][the][capital][of][Massachusetts]

# And recall that each of the input words is also associated with one of the hidden state context vectors,
#capturing information about the sentence up til that point, but primarily focused on that word in particular.

#    |    |    |     |      |        |
#  0.04 0.02 0.01  0.28   0.03     0.54
# [What][is][the][capital][of][Massachusetts]

# And so what we can now do is if we have all of these vectors, and we have values representing how important is
#it for us to pay attention to those particular vectors, is we can take a weighted average.

# We can take all of these vectors, multiply them by their attention scores, then add them up to get some new
#vecctor value, which is going to represent context from the input, but specifically paying attention to the
#words that we think are most important.

# And once we've done that, that context vector can be fed into a decoder in order to say what word should be, in 
#this case, Boston.

#                      [The][capital][is][Boston]
#                                           |
#                                           |
#                                           |
#                                           |
#    | +  | +  |  +  |  +   |   +    |  =   |
#    x    x    x     x      x        x
#  0.04 0.02 0.01  0.28   0.03     0.54
# [What][is][the][capital][of][Massachusetts]

# So attention is this very powerful tool that allows any word, when we're trying to decode it, to decide which
#words from the input we should pay attention to, in order to determine what's important for generating the next word 
#of the output.

# And one of the first places this was really used was in the field of machine translation.

# And so when we combine the attention mechanism with a recurrent neural network, we can get very powerful and 
#useful results, where we're able to generate an output sequence by paying attention to the input sequence too.

# But there are other problems with this approach of using a recurrent neural network as well.

# In particular, notice that every run of the neural network depends on the output of the previous step.

# And that was important for getting a sense for the sequence of words and the ordering of those particular 
#words, but we can't run a unit of the neural network until after we've calculated the previous hidden state.

# And what that means is that it's very difficult to parallelize this process, that as the input sequence get
#longer and longer, we might want to use parallelism to try and speed up this process of training the neural 
#network and making sense of all of this language data, but it's difficult to do that, and it's slow to do that
#with a recurrent neural network because all of it needs to be performed in sequence.

# And that's become an increasing challenge as we started to get larger and larger language models.

# The more language data we have available to us to use to train our machine learning models, the more accurate 
#it can be, the better representation of language it can have, the better understanding it can have, and the better 
#results we can see.

# So we've seen this growth of large language models that are using larger and larger data sets, but as aresult they
#take longer and longer to train.

# And so this problem, that recurrent neural networks are not easy to paralelize has become an increasing problem.

# And as a result of that, that was one of the main motivations for a different architecture for thinking about
#how to deal with natural language, and that's known as the transformer architecture, and this has been a significant
#milestone in the world of natural language processing for really increasing how well we can perform these kinds
#of natural language processing tasks as well as how quickly we can train a machine learning model to be able to
#produce effective results.

# There are a number of different types of transformers in terms of how they work, but what we're going to take a 
#look at here is the basic architecture for how one might work with a transformer to get a sense for what's 
#involved and what we're doing.

# So let's start with the model that we were looking at before.

#
#                  Output Sequence   <end>
#                        ____ ____ 
#                       |    |    |    |
#                       |    |    |    |
#  []-|>[]-|>[]-|>[]-|>[]-|>[]-|>[]-|>[]
#   |    |    |    |    |              |
#   |____|____|____|    |              |
#    Input Sequence   <end>          <end>

# We're going to look specically at this encoder part of our encoder decoder architecture, where we used the recurrent
#neural network to take this input sequence and capture all of this information about the hidden state and the 
#information we need to know about that input sequence.

# Right now it all needs to happen in this linear progression, but what the transformer is going to allow us to do
#is, is process each of the words independently in a way that's easy to parallize rather than having each word
#wait for some other word.

# Each word is going to go through this same neural network and produce some kind of encoded representation of that
#particular input word.

# And all of this is going to happen in parallel.

# And it's happenning for all the words at once, but we'e really gonna just focus on what's happenning for one 
#word to make it clear, but know that what ever we're seeing happen for one word is happenning for all of 
#the other input words too.

# So what's going on here?

# Well we start with some input word, that input word goes into the neural network, and the output is hopefully
#some encoded representation of the input word, the information that we need to know about the input word that's
#going to relevent to us as we're generating the output.

# And because we're doing this each word independently, it's easy to parallize, we don't have to wait for the 
#previous word before we run this word through the neural network.

# But what did we lose in this process, by trying to parallelizethis whole thing?

# Well we've lost all notion of word ordering.

# The order of words is important. 

# The sentence, "Sherlock Holmes gave the book to Watson", has a different meaning then "Watson gave the book to
#Sherlock Holmes".

# And so we want to keep track of that information about word position.

# In the recurrent neural network that happens for us automatically.

# We could run each word one at a time through the neural network, get the hidden state, and pass it on to the next
#run of the neural network.

# But that's not the case here with the transformer where each word is being processed independent of all the other
#words.

# So what are we going to do to try to solve that problem.

# One thing we can do is add some kind of positional encoding to the input word.

# The positional encoding is some vector that represents the position of the word in the sentence.

# The first, the second word, the third word, so on and so forth, add that to the input word, and the result of that 
#is going to be a vector that captures multiple pieces of information.

# It captures the input word itself, as well as where in the sentence it appears.

# The result of that is that we can pass the output of that addition, the addition of the input word, and the 
#positional encoding, into the neural network, that way the neural network knows the word and where it aappears in 
#the sentence, and can use both of those pieces of information to determine how best to represent the meaning of that
#word in the encoded representation at the end of it.

# In addition to what we have here, in addition to the positional encoding and the feed forward neural netwrok,
#we're also going to add one addtional component, which is going to be a self attention step.

# This is going to be attention, where we're paying attention to the other input words, because the meaning or the 
#interpretation of an input word might vary depending on the other words in the input as well.

# So we're going to allow each word in the input to decide what other words in the input it should pay attention to
#in order to decide on its encoded representation.

# And that's going to allow us to get a better encoded representation for each word, because words are defined by their
#context, by the words around them and how they're used in that particular context.

# This kind of self attention is so valuable in fact, that often times a transformer will use multiple different 
#self attention layers at the same time, to allow for this model to be able to pay attention to multiple facets
#of the input at the same time.

# We call this multi headed attention, where each attention head can pay attention to something different and as a
#result this network can learn to pay attention to many different parts of the input for this input word, all at 
#the same time.

# And in the spirit of deep learning, these two steps, this multi headed self attention layer, and this neural network
#layer, that itself can be repeated multiple times too, in order to get a deeper representation, in order to learn
#deeper patterns within the input text and ultimately get a better representation of language, in order to get 
#useful encoded representations of all the input words.

# And so this is the process that a transformer might use in order to take an input word and get it as encoded
#representation.

# And the key idea is to really rely on the attention step in order to get information that's useful in order to 
#determine how to encode that word.

# And that process is going to repeat for all of the input words that are in the input sequence. 

# We're going to take all of the input words, encode them with some kind of positional encoding, feed them into the
#self attention and feed forward neural networks in order to ultimately get those encoded representations of the
#words.

# That's the result of the encoder. 

# We get a bunch of encoded representations that will be useful to us when it comes time to try and decode all
#of that information into the output sequence we're interested in.

# And again, this might take place in the context of machine translation, where the ouptut is going to be the
#same sentence in a different language, or it might be an answer to a question,in the case of an AI chatbot for
#example.

# And so now let's take a look how that decoder is going to work.

# Ultimately it's going to have a very similar structure.

# Anytime we're trying to generate the next output word we need to know what the previous output is, as well as 
#its positional encoding, where in the output sequence are we.

# And we're going to have these same steps.

# Self attention, because we migth want an output word to be able to pay attention to other words in that same 
#output, as well as a neural network, and that itself may repeat multiple times.

# But in this decoder, we're going to add one additional step.

# We're going to add an additional attention step, where instead of self attention, where the output word is going to
#pay attention to other output words, in this step, we're going to allow the output word to pay attention to the
#encoded representations.

# So recall that the encoder is taking all of the input words and transforming them into coded representations 
#of all of the input words.

# But it's going to be important for us to be able to decide which of those encoded representations we want to
#pay attention to when generating any particular token in the output sequence.

# And that's what this additional attention step is going to allow us to do.

# It's saying that every time we're generating a word of the output, we can pay attention to the other words of 
#the output because we might want to know, what are the words we generated previously, and we want to pay attention
#to some of them to decide what word is next in the sequence.

# But we also care about paying attention to the input words too.

# And we want the ability to decide which of the encoded representations of the input words are going to be
#relevent in order for us to generate the next step.

# And so these two pieces combine together.

# We have the encoder that takes all of the input words and produces the encoded representation.

# And we have the decoder that is able to take the previous output word, pay attention to that encoded input
#and then generate the next output word.

# And this is one of the possible architectures that we can use for a transformer. 