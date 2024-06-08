# And this type of network is what we call, A deep neural network

#   Deep Neural Network -
# - Neural Network with multiple hidden layers.

# And all deep learning is about, is it's using multiple layers to be able to predict and be able to model higher
#level features inside of the input, to be able to figure out what the output should be.

# This allows us to be able to model more and more sophisticated types of functions, that each of the hidden layers,
#if more than one, can calculate something a little bit different, and we can combine that information to figure
#out what the output should be.

# Of course as with any situation in machine learning, as we begin to make our models more and more complex, to model
#more and more complex functions, the risk we run is something like, overfitting.

# And we talked about overfitting last time, in the context of overfitting based on when we were training our models
#to be able to learn some sort of decision boundary.

# Where overfitting happens when we fit too closely to the training data, and as a result we don't generalize well
#to other situations as well.

# And one of the risk we run with a far more complex neural network that has many different nodes, is that we might
#overfit, based on the input data.

# We might grow over reliant on certain nodes to calculate things just purely based on the input data, that doesn't
#allow us to generalize very well to the output.

# And there are a number of strategies for dealing with overfitting, but one of the most popular in the context
#of neural networks, is a technique called dropout.


#   Dropout -
# - Temporarily removing units - selected at random - from a neural network to prevent over-reliance on certain 
#units.

# What generally happens in overfitting, is it would begin to over rely on certain units inside the neural network,
#to be able to tell us how to interpret the input data.

# What dropout will do is randomly remove some of these units in order to reduce the chance that we over rely on 
#certain units to make our neural network more robust.

# To be able to handle a situation even when we just dropout particular neurons entirely.

# So the way that might work, is if we have a network like this, and as we're training it, when we go about trying
#to update the weights the first time, we'll just randomly pick some percentage of the nodes to drop out of the
#network.

#       O   O
#   O           O
#       O   O
#   O           O   O
#       O   O
#   O           O
#       O   O
#

# It's as if those nodes aren't there at all. It's as if the weights associated with those nodes aren't there at all,
#and we'll train it this way.

#       O   O
#   O           O
#       O    
#   O           O   O
#           O
#   O            
#       O   O
#

# Then the next time that we update the weights, we'll pick a different set and just go ahead and just train that way.

# And then again randomly choose and train with other nodes that have been dropped out as well.

# And the goal of that is that after the training process, if we train by dropping out random nodes inside of this
#neural network, we hopefully end up with a network just a little bit more robust, that doesn't rely to heavily on
#any one particular node, but more generally learns how to approximate a function in general.

# So that then is a look at some of these techniques that we can use in order to implement a neural network.

# To get at the idead of taking this input, passing it through these various different layers in order to produce
#some sort of output.

# And what we'd like to do now is take those ideas and put them into code.

# And to do that there are a number of different machine learning libraries, neural netwrok libraries that we can 
#use, that will allow us to get access to some of the implementation of backpropagation, and all of these hidden
#layers.

# And one of the most popular, developed by google, is known as TensorFlow.

# A library that we can use for quickly creating neural networks and modeling them, and running them on some 
#sample data, to see what the output is going to be.

# We'll recall from last time that we had our banknotes file, that included information about counterfeit 
#banknotes as opposed to authentic banknotes.

# We had four different values for each banknote, and a categorization of whether that banknote is considered 
#to be authentic or counterfeit.

# And what we wanted to do was, based on that input information, figure out some function, that could calculate 
#based on the input information what category it belonged to.

# Here will add our imports and begin to write our code.

import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split

# What we are writing here is a neural network that will learn based on all of the inputs, whether or not we should
#categorize the banknote as authentic or counterfeit.

# The first step is the same as we saw before, we're really just reading the data in and getting it into an appropriate
#format.


# Read data in from file 
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row [4] == "0" else 0
        })


# Here we separated into a training and testing set.

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4 
)

# And now what we're doing down below is creating a neural network.

# We will use tf, which stands for tensorflow, tf.keras. keras is an API, a set of functions we can use to manipulate
#neural networks inside of tensorflow.

# Here we're saying go ahead and give us a model, that is a sequential model, a sequential neural network, meaning one 
#layer after another.

# Create a neural network
model = tf.keras.models.Sequential() 

# And now we're going to add to that model, what layers we want inside of our neural network

# So we're going to say model.add, go ahead and add a dense layer.

# And when we say dense layer, we just mean that each of the nodes in this layer will be connected to the nodes in
#the previous layer. A densely connected layer.

# This layer is going to have 8 units inside of it.

# So it's going to be a hidden layer inside our neural network with 8 different units, 8 artificial neurons, each 
#that might learn something different.

# 8 is just a number we choose as an example. Remember, the more units means the more complex functions we can learn,
#so maybe we can more accurately model the training data.

# But it comes at a cost. More units means more weights that we need to figure out how to update. So it might be 
#more exspensive to do that calculation.

# And we also run the risk of overfitting. If we have too many units and we learn to just on the training data,
#that's not good either.

# So there is a balance. There's often a testing process. Where we'll train on some data, and maybe validate how 
#well we're doing on a separate set data, often called a validation set.


# Next we specify what the input shape is. Meaning what does our input look like.

# Our input has four values, because the input shape is just 4. And then we specify what the activation for our
#function is. And the activation function is something we can choose, and there are a number we can choose from.

# Here we will use the ReLu.

# Add a hidden layer with 8 inputs, with ReLu activation
model.add(tf.keras.layers.Dense(8, input_shape = (4,), activation = "relu"))


# And then, we'll add an output layer. So we have our hidden layer, now we're adding one more layer, that will have
#just one unit, because all we want to do is predict something like counterfeit bill, or authentic bill, so we 
#just need a single unit.

# And the activation function we are going to use here, is the Sigmoid activation function. This just gave us a 
#probability, like what is the probability that this is a counterfeit bill as opposed to an authentic bill.

# So this is the structure of our neural network. A sequential neural network that has one hidden layer, with 8
#units inside of it, and then one output layer, that just has a single unit inside of it.

# And an output layer with one unit with sigmiod activation
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))


# Then we are going to compile our model.

# Train neural network
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)


# Now that we've defined our model, we call model.fit, and say go ahead and train the model.

# Train it on all the training data, plus all the training labels.

# So labels for each of those pieces of traiining data.

# And we're specifying that we want to train it for 20 #epochs, meaning go through each of these training points
#20 times. Go through the data 20 times and keep trying to update the weights.

# If we did it for more there is a chance that we'd end up with an even more accurate model.
model.fit(X_training, y_training, epochs = 33)


# And after we've fitted all of the data, we'll go ahead and just test it.

# Ultimately, this is going to tell us how well we did in this particular case.

# Evaluate how well the model performs
model.evaluate(X_testing, y_testing, verbose = 2)

# Note: At the end of the training #Epochs, we will aslo get back the accuracy of the testing data. This is the data
#we used to test our training data on.

# Remember, we are using the hold-out cross validation method. That is when we split our dataset up into training
#and testing sets. That is why we have testing data also.

# So just using these steps we were able to generate a neural network, that can detect counterfeit bills from
#authentic bills, based on the input data.

# This is the value of using a machine learning library like tensorflow.

# All we had to do was define the structure of the network, and define the data that we're going to pass into the 
#network, and then, tensorflow runs the backpropagation algorithm for learning what all of those weights should be,
#for figuring out how to train this neural network to be able to as accurately as possible figure out what the output
#values should be.


# So that then was a look at what it is that neural networks can do just using these sequences, of layer after 
#layer after layer. 

# And we can begin to imagine applying these to much more general problems.

# And one big problem in artificial intelligence, is the problem of computer vision.

#   Computer Vision -
# - Computational methods for analyzing and understanding digital images.

# We might have pictures that we want the computer to figure out how to deal with, how to process those images and
#figure out how to produce some sort of useful result.

# So comuter vision is all about taking an image and figuring out what sort of computation, what sort of calculation
#we do with that image.

# It's also relevent in the context of something like handwriting recognition. 

# Let's take the M-nist dataset for example, which is a big dataset of just handwritten digits, that we could use 
#to ideally try and figure out how to predict, given someones handwritting, given a photo of a digit they have
#drawn, can we predict whether it's a 0 or 1 or 2 etc.

# So this sort of handwritting recognition is yet another task that we might want to use computer vision task and 
#tools to be able to apply it towards.

# So how then can we use neural networks to be able to solve a problem like that.

# Well neural networks rely on some sort of input, where that input is just numerical data.

# We have a whole bunch of units, where each one of them just represents some sort of number, and so in the context
#of something like handwritting recognition, or in the context of just an image, we might image that an image is 
#really just a grid of pixels, grid of dots, where each dot has some sort of color, and in the context of something
#like handwritting recognition, we might image that if we just fill in each of the dots in a particular way, we
#can generate a character. A number, in this instance, based on which dot happens to be shaded in, and which dot is not.

# And we can represent each of the pixels values just using numbers.

# So for a particular pixel for example, 0 might represent entirely black, depending on how we're representing
#color.

# It's often common to represent color values on a 0 - 255 range, so we could represent the color using 8 bits for a
#particular value, like how much white is in the image.

# So 0 might represent all black, 255 might represent entirely white as a pixel, and somewhere in-between might
#represent some shade of gray, for example.

# But we might image not just having a single 0 - 255 range to determine how much white is in the image, but if we 
#had a color image, we might image 3 different 0 - 255 range values. R G B values.

# Where the R detremines how much red is in the image.

# We have one value to determine how much green is in the pixel.

# And one value for how much blue is in the pixel.

# And depending on how it is we set up these values of red, green, and blue, we can get a different color.

# And so any pixel can really be represented in this case, by 3 numerical values.

# A red value, a green value, and a blue value.

# And if we take a whole bunch of these pixels, assemble them together inside a grid of pixels, then we just have 
#a whole bunch of numerical values that we can use in order to form some sort of prediction task.

# And so what we might imagine doing, is using the same techniques we talked about before, just design a neyral network
#with a lot of inputs.

# That for each of the pixels, we might have 1 or 3 different inputs in the case of a color image, a different input
#that is just connected to a deep neural network for example.

# And this deep neural network might take all of the pixels inside of the image of like what digit the person drew,
#for example, and the output might be like 10 neurons that are classified as a 0, 1, 2, etc etc.

# Or just tells us in some way, what that digit happens to be.

# Now there are a couple of drawbacks to this approach.

# The first drawback to the approach is just the size of the potential input array.

# We'd have a whole bunch of inputs if we have a big image that has a lot of different channels, we're looking at a
#lot of inputs and therefore a lot of weights that we'd have to calculate.

# And our second problem is the fact that by flattening everything, just the structure of all the pixels, we would
#have lost access to a lot of the information about the structure of the image that's relevent.

# That really, when a person looks at an image, they're looking at particular features of an image.

# They're looking at curves and shapes, they're looking at what things can they identify in differnt regions 
#of the image, and maybe put those things together in order to get a better picture of what the overall image 
#is about.

# And by just turning it into pixel values for each of the pixels, sure we might be able to learn that structure,
#but it might be challenging in order to do so.

# It might be helpful to take advantage of the fact that we can use properties of the image itself, the fact that 
#it's structured in a particular way, to be able to improve the way that we learn based on that image too.

# So in order to figure out how we can train our neural networks, to better be able to deal with images, we'll
#introduce a couple of ideas, a couple of algorithms that we can apply, that will allow us to take the image, and
#extract some useful information out of that image.

# And the first idea we'll introduce, is the notion of image convolution.

#   Image Convolution -
# - Applying a filter that adds each pixel value of an image to its neighbors, weighted according to a kernel matrix.

# And the goal of image convolution then, is to extract some sort of interesting or useful features out of an image,
#to be able to take a pixel, and based on its neighboring pixels, maybe predict some sort of valuable information.

# Something like taking a pixel and looking at its neighboring pixels, we might be able to predict whether or not
#there's some sort of curve inside the image, or whether it's forming the outline of a particular line or a shape
#for example. 

# And that might be useful if we're trying to use all of these various different features to combine them to say
#something meaningful about an image as a whole.

# So how then does an image convolution work?

# Well we start with a kernel matrix.

# And the kernel matrix looks something like this.


#  ________________________________
# |         |           |          |
# |    0    |    -1     |    0     |
# |_________|___________|__________|         
# |         |           |          |
# |   -1    |     5     |   -1     |
# |_________|___________|__________|        
# |         |           |          |
# |    0    |    -1     |    0     |
# |_________|___________|__________|


# And the idea of this, is that given a pixel, the middle pixel, for example, we're going to multiply each of the
#neighboring pixels, by these values, in order to get some sort of result, by summing up all the numbers together.

# So if we take this kernel, which we can think of as a filter that we are going to apply to the image.

# And let's say that we take this image.

#  ____________________________________________
# |         |           |          |           |
# |   10    |    20     |    30    |    40     |
# |_________|___________|__________|___________|        
# |         |           |          |           |
# |   10    |    20     |    30    |    40     |
# |_________|___________|__________|___________|      
# |         |           |          |           |
# |   20    |    30     |    40    |    50     |
# |_________|___________|__________|___________|
# |         |           |          |           |
# |    20   |    30     |    40    |    50     |
# |_________|___________|__________|___________|

# This is a 4 by 4 image.

# We'll think of it as just a black and white image. Where each one is just a single pixel value.

# So somewhere between 0 and 255 for example.

# So we have a whole bunch of individual pixel values like this, and what we'd like to do, is apply our kernel
#to our image.

# And the way we'll do that is look at the size of our kernel, which is a 3 by 3.

# So we'll take it and apply it to the first 3 by 3 section of our image.

# And what we'll do is, we'll take each of the pixel values, multiple it by its corresponding value in the filter
#matrix, and add all of the results together.

# So for example we'll say, 10 times 0, plus 20 times -1, plus 30 times 0, so on and so forth, doing all of this 
#calculation.

# And in the end, if we take all of the values and multiply them by their corresponding values in the kernel,
#add the results together, for the that particular set of 9 pixels, we would get the value of 10.

#  _____________________
# |         |           |        
# |   10    |           |    
# |_________|___________|        
# |         |           |          
# |         |           |  
# |_________|___________|

# Note: Because we have a kernel size of 3 by 3, and an input size of 4 by 4, remembering that our stride is 1 by
#default unless otherwise specified, we would have a 2 by 2 output, as shown above.

# Next we will take our kernel, and slide it to the right, to look at the next 3 by 3 section, and repeat the 
#calculation process we performed on the first section.

# Once we are done with this set of calculations, we will have the number 20.

#  _____________________
# |         |           |        
# |   10    |    20     |    
# |_________|___________|        
# |         |           |          
# |         |           |  
# |_________|___________|

# Then we will shift our kernel to the bottom left and do the samething, which will produce a number of 40.

#  _____________________
# |         |           |        
# |   10    |    20     |    
# |_________|___________|        
# |         |           |          
# |   40    |           |  
# |_________|___________|

# Lastly we will slide the kernel to its final location to the right and perform the final calculations.

#  _____________________
# |         |           |        
# |   10    |    20     |    
# |_________|___________|        
# |         |           |          
# |   40    |    50     |  
# |_________|___________|

# And what we have now is what we'll call a feature map.

# We've taken our kernel and applied it to the various different regions, and what we get is some 
#representation of a filter version of that image.

# Note: That when our middle value of the input image is very different from the neighboring values, then we are 
#more likely to end up with a value higher than zero when doing calculations. But if the middle value is the same
#as all of its neighbors, then we have a higher a chance of calculationg a zero or possible a zero.

# And it turns out that this sort of filter can be used for something like detecting edges in an image.

# If we wanted to detect the boundaries between various different objects inside of an image for example, we might
#use a filter like this, which is able to tell whether the value of a pixel is different from the values of its
#neighboring pixels, or greater than, the pixels that happen to surround it.

# And so we can use this in terms of image filtering.

# When see an example of this in the code we write to formulate this.

# These are the imports we need.

# This time we add pythons image library, PIL as well.

import math
import sys
from PIL import Image, ImageFilter 


# Ensure correct usage
if len(sys.argv) !=2:
    sys.exit("Usage: Python filter.py filename")


# This is how we specify our image specifications

# Open image
image = Image.open(sys.argv[1]).convert("RGB")


# Here we will apply a kernel to our image. It's going to be a 3 by 3 kernel size, with the values specified in
#kernel =... 

# Filter image according to edge dection kernel
filtered = image.filter(ImageFilter.Kernel(
    size = (3, 3),
    kernel = [4, -3, 2, -1, 1, 1, -2, 3, -4],
    scale = 1
))


# And finally, we'll go ahead and show our image.

# Show resulting image
filtered.show()

# Notice that what we get is a filtered version of our original image.

# Just by taking our image, and applying that filter to each 3 by 3 grid, we've extracted all of the boundaries,
#all of the edges inside of the image that separate one part of the image from another.

# And we might image if a machine learning algorithm is trying to learn like what an image is of, a filter like this 
#could be pretty useful.

# So this type of idea of image convolution can allow us to apply filters to images that allow us to extract useful
#results out of our images.

# So that was the idea of image convolution. Applying some sort of filter to an image, to be able to extract some 
#useful features out of that image.

# But all the while, these images are still pretty big. There are a lot of pixels involved in the image. And 
#realistically speaking, if we have a really big image, that poses a couple of problems. 

# One, that means a lot of input going into the neural network. 

# Two, it also means that we really have to care about what's inside each particular pixel. 

# Where as realistically, when we're looking at an image, we don't care if something is one particular pixel,
#vs the pixel to the immediate right of it. They're pretty close together.

# We really just care about whether there is a particular feature in some region of the image, and maybe we don't care
#about exactly which pixel it happens to be in.

# And there's a technique we can use known as pooling.

#   Pooling -
# - Reducing the size of an input by sampling from regions in the input.

# What this means is that we're going to take a big image, and turn it into a smaller image, by using pooling.

# In particular, one of the most popular types of pooling is called max-pooling

#   Max-pooling -
# - Pooling by choosing the maximum value in each region

# So that then is pooling. Taking the size of the image, reducing it a little bit by just sampling from particular
#regions from inside of the image.

# And now we can put all of these ideas together. Pooling, Image Convolution, and Neural Networks, all together,
#into another type of neural network, called a concolution neural network.

#   Convolutional Neural Network -
# - Neural Networks that use convolution, ususally for analyzing images.

# And the way that a C N N works, is we start with some sort of input image, some grid of pixels.

# But rather than immediately putting that into the neural network layers that we seen before, we'll start by applying
#a convolution step.

# Where the convolution step involves applying some number of image filters to our original image, in order to get
#what we call a feature map.

# The result of applying some filter to an image.

# And we could do this once, but in general, we'll do this multiple times.

# Getting a whole bunch of feature maps, each of which might extract some different relevent feature out of the image.

# Some different important characteristic of the image that we might care about using in order to calculate what the 
#result should be.

# And in the same way that when we train neural networks, we can train neural networks to learn the weights between 
#particular units inside of the neural networks, we can also train neural networks to learn what those filters should 
#be, what the values of the filters should be, in order to get the most useful, most relevent information out of the
#original image just by figuring out what setting of those filters values, the values inside of that kernel, results 
#in minimizing the loss function, minimizing how poorly our hypothesis actually performs in figuring out the
#classififcation of a particular image, for example.

# So we first apply the convolution step to get a whole bunch of various different feature maps.

# But these feature maps are quite large. There are a lot of pixel values that happen to be here.

# And so a logical next step to take, is a pooling step.

# Where we reduce the size of the images by using max-pooling, for example, extracting the maximum value from any
#particular region.

# There are other pooling methods that exist as well depending on the situation.

# Something like average pooling, for instance, where instead of taking the maximum value of a region, we take the
#average value from a region, which has its uses as well.

# In effect, what pooling will do, it will take the feature maps, and reduce their dimensions so that we end up 
#with smaller grids and fewer pixels.

# And this then is going to be easier for us to deal with.

# It means fewer inputs that we have to worry about. It's also going to mean that we are more resilient, more
#robust against potential movement from particular values,just by one pixel, when ultimately we really don't care
#about those pixel differences that might arise in the original image.

# Now after we've done this pooling step, now we have a whole bunch of values that we can then flatten out, and put
#into more traditional neural networks.

# So we'll go ahead and flatten it, and end up with a traditional neural network that has one input from each of
#the values from each of the resulting feature maps, after we do the convolution, and after we do the pooling step.

# This is the general structure of a convolutional network. 

# We begin with the image, apply convolution, apply pooling, flatten the results, and then put that into a more 
#traditional neural network that might itself have hidden layers.

# We could have deep convolutional networks that have hidden layers in between the flattened layer, and the eventual
#output, to be able to calculate the various different features of those values.

# This can help us to be able to use convolution and pooling, to use our knowledge about the structure of an image
#to be able to get that better result, to be able to train neural networks faster, in order to better capture 
#particular parts of the image.

# And there's no reson necessarily why we have to use the steps once.

# In fact, in practice we'll often use convolution and pooling multiple times, in multiple different steps.

# What we might imagine doing is starting with an image, first applying convolution to get a whole bunch of maps,
#then applying pooling, then, applying convolution again.

# Because the maps are still pretty big, we can apply convolution to try and extract relevent features out of
#the first result, then take those results, apply pooling in order to reduce their dimensions, and then take that
#and feed it into a neural network that maybe has fewer inputs.

# So with that example, we would have two different convolution and pooling steps.

# We do convolution and pooling once, and then we do convolution and pooling a second time.

# Eaxh time extracting useful features from the layer before it, each time using pooling to reduce the dimensions
#of what we're ultimately looking at.

# And the goal now of this sort of model is that in each of these steps, we can begin to learn different types 
#of features of the original input.

# That maybe in the first step, we learn very low level features, just learn and look for features like edges and 
#curves and shapes, because based on pixels in the neighboring values, we can figure out, what are the edges, what
#are the curves, what are various different shapes that may be present there.

# But then, once we have a mapping that represents where the edges, and curves and shapes happen to be, we can
#image applying the same sort of process again, to begin to look for higher level features, like objects, or maybe
#people with eyes in facial recognition for example.

# Or maybe look for more complex shapes, like curves on a particular number that we are trying to recognize, a digit
#in a handwritten recognition sort of scenario.

# And then after all of that, now that we have these results, that represent these higher level features, we can
#pass them into a neural network, which is really just a deep neural network, where we might imagine making a 
#binary classification, or classifying into multiple categories, or performing various different task with this sort
#of model.

# So Convolutional neural networks can be quite powerful and quite popular when it comes to trying to analyze images.

# And so we might image applying this to a situation like hand writing recognition.

# We'll go ahead and see an example of that now.

# We'll write some code to reflect everything we've gone over.

# First we'll get our imports

import sys
import tensorflow as tf 

# One of the most popular datasets in machine learning is the mnist dataset, which is a dataset of a whole bunch of
#examples of peoples handwritten digits. And what we can do is immediately access that dataset which is built into
#the library. This dataset is found inside of the tensorflow library.

# Use MNIST handwritting dataset
mnist = tf.keras.datasets.mnist


# Now there's a bit of reshaping we need to do. Just turning the data into a format that we can put into our C N N.

# Doing things like taking all the values and dividing them by 255. Remember that color values tend to range from 0
#to 255. So we can divide them into 255, just to put them into a 0 or 1 range, to make it a little eaiser to
#train on.

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train /255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)


# But here's the interesting and important part

# Here's where we create Our C N N.

# We're saying go ahead and use a Sequential model.

# And before, where we had to use model.add, saying add a layer, add a layer, add a layer etc etc..., another way
#we can define it is by passing as input to our sequential neural network, a list of all the layers we want.

# Create a Convolutional Neural Network
model = tf.keras.models.Sequential([

    # And so here, the very first layer in our model is a convolutional one, where we're first going to apply
    #convolution to our image.

    # We're going to use 32 different filters. So our model is going to learn 32 different filters on the input image.

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3,3), activation ="relu", input_shape = (28, 28, 1)
    ),

    # Our max_pooling filter means that we are going to look at 2 by 2 regions inside of the image, and just extract
    #the maximum value.

    # Again, this helpful because it will reduce the size of our input

    # Max-Pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

    # Next we will flatten all of the units into a single layer, that we can then pass into the rest of the neural 
    #network.

    # Flatten Units
    tf.keras.layers.Flatten(),

    # And here is the rest of the neural network.

    # Here we're saying let's add a hidden layer to our neural network with 128 units.

    # And just to prevent overfitting, we'll add a dropout, that says when we're training, randomly dropout half of
    #the nodes from that layer, just to make sure we don't become over reliant on any particular node.

    # This will help us generalize and stop ourselves from overfiting.

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.5),

    # And then finally, we add an output layer.

    # The output is going to have 10 units, one for each category that we would like to classify a digit into, so 
    #0 to 9, different categories.

    # And the activation we're going to use here is called the softmax activation function. 

    # In short, what the softmax activation function is going to do is, take the the output, and turn it into a
    #probability distribution.

    # So ultimately it's going to tell us what did we estimate the probability is that this is a 2, as opposed to
    #a 3, or a 4, and we'll turn it into that probability distribution formula.

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(10, activation = "softmax")
])


# Train Neural Network
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = "accuracy"
)


# Next we will fit our model
model.fit(x_train, y_train, epochs = 10)


# Evaluate Neural Network Performance
model.evaluate(x_test, y_test, verbose = 2)


# Here we've added to our python program, if we provided a command line argument with the name of a file, we're
#going to go ahead and save the model to a file. 

# This can be quite useful too. Once we've done the training, which could take some time, we'd like to remember
#all the information we've learned so that we can use it later.

# And so tensorflow allows us to save a model to a file, such that later if we want to use the model that we've 
#learned, use the information we've learned to make some new prediction, we can just use the model that already exists.

# So what we're doing here, is after we're done doing all the calculation, we'll save the model to a file, such that
#we can use it a little bit later.

# Save model to file
if len(sys.argv) == 3:
   filename = sys.argv[2]
   model.save(filename)
   print(f"Model saved to {filename}.")


# Output

#Epoch 1/10
#1875/#1875 [==============================] - 294s 154ms/step - loss: 0.2289 - accuracy: 0.9311
#Epoch 2/10
#1875/#1875 [==============================] - 182s 97ms/step - loss: 0.0949 - accuracy: 0.9714
#Epoch 3/10
#1875/#1875 [==============================] - 195s 104ms/step - loss: 0.0751 - accuracy: 0.9772
#Epoch 4/10
#1875/#1875 [==============================] - 210s 112ms/step - loss: 0.0585 - accuracy: 0.9820
#Epoch 5/10
#1875/#1875 [==============================] - 164s 87ms/step - loss: 0.0504 - accuracy: 0.9837
#Epoch 6/10
#1875/#1875 [==============================] - 152s 81ms/step - loss: 0.0438 - accuracy: 0.9856
#Epoch 7/10
#1875/#1875 [==============================] - 163s 87ms/step - loss: 0.0367 - accuracy: 0.9883
#Epoch 8/10
#1875/#1875 [==============================] - 150s 80ms/step - loss: 0.0333 - accuracy: 0.9894
#Epoch 9/10
#1875/#1875 [==============================] - 160s 86ms/step - loss: 0.0301 - accuracy: 0.9901
#Epoch 10/10
#1875/#1875 [==============================] - 152s 81ms/step - loss: 0.0262 - accuracy: 0.9916
#313/313 - 7s - loss: 0.0368 - accuracy: 0.9904 - 7s/epoch - 22ms/step


# So that was a look at how we can use convolutional neural networks to begin to solve problems, with regards to
#computer vision. 

# To be able to take an image and begin to analyze it.

# So this is the type of analysis we might imagine that's happenning in self driving cars that are able to determine
#what filters to apply to an image to understand what it is that a user is looking at.

# Or the same type of idea that can be applied to facial recognition in social media to be able to determine how
#recognize faces in an image as well.

# We can imagine a neural network that instead of classifying into one of 10 different digits, can instead classify
#something like is this person A, or is it person B, trying to tell those people apart just based on convolution.

# And so now what we'll take a look at is yet another type of neural network that is popular with certain types
#of task.

# But to do so, we'll try to generalize and think about our neural network a little more abstarctly.

# Below we have an example of a deep neural network.

# Note: Image that all of the nodes are connected by weights.


#      O    O
# O              O
#      O    O
# O              O      O
#      O    O
# O              O 
#      O    O


# We have our initail input layer.

# A whole bunch of hidden layers that are performing certain types of calculations.

# And then an output layer that generates some sort of output that we care about calculating. 

# But we could imagine representing this a little more simply, using a more abstract representation of our neural
#network.

#  __________          ____________          __________
# |          |        |            |        |          |
# |  Input   |------> |  Network   |------> |  Output  |
# |__________|        |____________|        |__________|


# We have some input, which might be a vector of a whole bunch of different values. 

# That gets passed into a network that performs some sort calculation or computation.

# And that network produces some sort of output.

# That output might be a single value, it might be a whole bunch of different values, but this is the general
#structure of the neural network that we seen.

# There is some sort of input that gets fed into the network, and using that input the network calculates what the
#output should be.

# And this sort of model for a neural network, is what we might call a feed-forward neural network.


#   Feed-Forward Neural Network -
# - Neural network that has connections in only one direction.

# It moves from one layer, to the next layer, to the layer after that.

# The inputs pass through various different hidden layers, and then ultimately produce some sort of output.

# So feed-forward neural networks were very helpful for solving these types of classification problems that we saw
#before.

# We have a whole bunch of input, we want to learn what setting of weights will allow us to calculate the output
#effectively.

# But there are some limitations of feed-forward neural networks that we'll see in a minute.

# For instance, the input needs to be a fixed shape, or a fixed number of neurons in the input layer.

# And there's a fixed shape for the output.

# Like a fixed number of neurons in the output layer.

# And that has some limitations of its own.

# A possible solution for this, is instead of just a feed-forward neural network, where there are only connections 
#in one direction, from left to right effectively across the network, we can also imagine a recurrent neural network,
#where a recurrent neural network generates output that gets fed back into itself as input for future runs of that
#network.

# So as with a traditional neural network, we have inputs that get fed into the network, that get fed into the 
#output, and the only thing that determines the output is based on the original input and based on the calculation 
#we do inside of the network itself.

# This goes in contrast of a recurrent neural network, where in a recurrent neural network, we can imagine output
#from the network feeding back into itself, into the network again as input for the next time we do the calculations
#inside of the network.

# What this allows, is it allows the network to maintain some sort of state. To store some sort of information that
#can be used on future runs of the network.

# Previously the network just defined some weights, and we passed inputs through the network and generated an outputs.

# But the network wasn't saving any information based on those inputs to able to remember for future iterations, or
#for future runs.

# What a recuurent neural network will let us do is let the network store information that gets passed back in as input
#to the network again the next time we try and perform some action.

# And this is particularly helpful when dealing with sequencies of data.

# And so the strategy here is to use a recurrent neural network.

# A neural network that can feed its own output back into itself for input for the next time.

# And this allows us to do what we call a one to many relationship for inputs and outputs.

# But in vanilla, or traditional neural networks, these are what we might consider to be one to one neural networks.

# We pass in one set of values as inputs, we get back one vector of values as the output.

# But in this case, we wanna pass in one value as input, an image, for example, and we want to get a sequence, many 
#values of outputs, where each value is like a word in a sentence, for example, that gets produced by this particular
#algorithm.

# And so the way we might do this is, is we might imagine starting by providing input, the image, into out neural
#network, and the neural network is going to generate output, but the output is not going to be the whole sequence
#of words, because we can't represent the whole sequence of words using just the fixxed set of neurons.

# Instead the output is just going to be the first word, for example.

# We're going to train the network to output what the first word of the caption for our image should be.

# But now that the network generates outputs that can be fed back into itself, we can imagine the output of a network
#being fed back into the same network.

# Its the same network that's just getting different input.

# That networks output gets fed back itself, and it in turn generates a new output, and this other output is going to
#be something like the second word in the caption of our image, for example.

# That will be fed back into the network and generate yet another word based on the new calculations of the network.

# And so recurrent neural networks allow us to represent this sort of one to many structure.

# It provides one image as input and the neural network can pass data into the next run of the network, and then
#again and again, such that we can run the network multiple times, each time generating a different output, still
#based on that original input.

# And this is where recurrent neural networks become particularly useful when dealing with sequencies of inputs or
#outputs.

# Our output is a sequence of words, and since we can't very easily represent outputting an entire sequence of words,
#we'll instead output that sequance one word at a time, by allowing our network to pass information about what still
#needs to be said about the photo, into the next stage of running network.

# We can run the network multiple times, the same network with the same weights, just getting different inputs each
#time. 

# First getting input from the image, and then getting input from the network itself as additional information 
#about what additionally needs to be given in a particular caption, for example.

# We Can represent this with the example below.

#    _______          _________          ________
#   |       |        |         |        |        |
#   | Input |------> | Network |------> | Output | 
#   |_______|        |_________|        |________|
#                     ____|____          ____|___
#                    |         |        |        |
#                    | Network |        | Output |
#                    |_________|        |________|   
#                     ____|____          ____|___
#                    |         |        |        |
#                    | Network |        | Output |
#                    |_________|        |________|
#                     ____|____          ____|___
#                    |         |        |        |
#                    | Network |        | Output |
#                    |_________|        |________|
#
# So this then is a one to many relationship inside of a recurring neural network.

# But it turns out that there are other models we can use, other ways we can try and use this recurrent neural network
#to be able to represent data that might be stored in other forms as well.

# We saw how we can use neural networks in order to analyze images, in the context of convolutional neural networks
#that take an image, figure out various different properties of the image, in order to draw some sort of conclusion
#based on that.

# But we might imagine that something like youtube.

# They need to be able to do a lot of learning based on video.

# They need to look through videos to detect things like copyright violations, or they need to be able to look through
#videos to maybe identify what particular items are inside of the video, for example.

# And video, we might imagine is much more difficult to put as input into a neural network, because where as with
#the image we can just treat each pixel as a different value, videos are sequencies, they're sequencies of images, 
#and each sequence might be a different length, so it might be challenging to represent that entire video as a 
#single vector of values we can pass in to a neural network.

# And so here too, recurrent neural networks can be a valueable solution for trying to solve this type of problem.

# That is instead of just passing in a single input into our neural network, we can pass in the input one frame
#at a time, we might imagine.

# First, taking the first frame of the video, passing it into the network, and then maybe not having the network output
#anything at all.

# Let it take in another input, and this time, pass it into the network, but the network gets information from the
#last time we provided input to the network.

# Then we pass in a third input, then a forth input, where each time what the network gets, it gets the most
#recent input, like each frame of the video.

# But it also gets information the network processed from all of the previous iterations.

# So in frame number 4, we end getting the input for frame number 4, plus information the network has calculated
#from the first 3 inputs.

# And using all of that data combined, this recurrent neural network can begin to learn how to extract patterns 
#from a sequence of data as well.

# So we might imagine if we want to classify a video into a number of differnt genres, like an educational video,
#or a music video, or different types of videos, that's a classification task where we want to take as input
#each of the frames of the video, and we want to output something like what it is, or what category it happens to
#belong.

# And we can imagine doing this sort of thing, this sort of many to one learning, anytime our input is a sequence.

# And so input as a sequence in the context of a video, in could be in the context of something like someone that 
#typed a message and we want to be able to categorize that message, like if we're trying to take a movie review and 
#classify it as a positive or negative, for example, that input is a sequence of words and the output is a 
#classification, positive or negative.

# There too, a recurrent neural network might be helpful for analyzing sequences of words that are quite popular
#when it comes to dealing with langauge.

# It could even be used for spoken language as well.

# Spoken language is an audio wave form that could be segmented into distinct chunks, and each of those can be passed 
#in as an input into a recurring neural network to be able to classify someones voice, for example, in  order to do
#voice recognition, saying, is this one person, or another.

# Here are also cases where we might want this many to one architecture for a recurrent neural network.

# Here is an example of what that might look like displayed as a chart.

#    _________          __________
#   |         |        |          |
#   |  Input  |------> |  Network | 
#   |_________|        |__________|
#    ____|____          ____|_____ 
#   |         |        |          |
#   |  Input  |------> |  Network |
#   |_________|        |__________|   
#    ____|____          ____|_____
#   |         |        |          |
#   |  Input  |------> |  Network |
#   |_________|        |__________|
#    ____|____          ____|_____          ________
#   |         |        |          |        |        |
#   |  Input  |------> |  Network |------> | Output |
#   |_________|        |__________|        |________|
#

# And then there's one final problem to take a look at in terms of what we can do with these sorts of networks.

# Imagine something like google translate and what it's doing.

# So what google translate is doing, is it's taking some text written in one language, and converting it into text
#written in some other language, for example.

# Where now this input is a sequence of data, a sequence of words, and the output is a sequence of words as well,
#it's also a sequence.

# So here we want effectively something like a many to many relationship, our input is a sequence, and our output is a 
#sequence as well.

# And it's not quite going to work to just say, take each word in the input and translate it into a word in the 
#output, because ultimately different langauges put their words in different orders, and maybe one language uses
#two words for something where as another language only uses one.

# So we really want some way to take this information, this input, encode it somehow, and use that encoding to 
#generate what the output ultimately should be.

# And this has been one of the big advancements in automated translation technology, the ability to use neural
#networks to do this, instead of older more traditional methods.

# This has improved accuracy dramatically.

# And the way we might imagine doing this is again, using the recurrent neural network, with multiple inputs,
#and mutiple outputs.

# We start by passing in all the inputs.

# Input goes into the network, another input, like another word, goes into the input, and we do this multiple
#times, like once for each word in the input that we're trying to translate.

# And only after all of that is done does the network now start to generate output, like the first word of the
#translated sentence, then the next word of the translated sentence, then the next word, so on and so forth, where
#each time the network passes information to itself by allowing for this model of giving some sort of state from one 
#run of the network, to the next run, assembling information about all the inputs, and then passes that information
#about which part of the output in order to generate next.

# See an example of this below

#    _________          __________
#   |         |        |          |
#   |  Input  |------> |  Network | 
#   |_________|        |__________|
#    ____|____          ____|_____ 
#   |         |        |          |
#   |  Input  |------> |  Network |
#   |_________|        |__________|   
#    ____|____          ____|_____
#   |         |        |          |
#   |  Input  |------> |  Network |
#   |_________|        |__________|
#    ____|____          ____|_____          ________
#   |         |        |          |        |        |
#   |  Input  |------> |  Network |------> | Output |
#   |_________|        |__________|        |________|
#                       ____|_____          ________
#                      |          |        |        |
#                      |  Network |------> | Output |
#                      |__________|        |________|
#                       ____|_____          ________
#                      |          |        |        |
#                      |  Network |------> | Output |
#                      |__________|        |________|

# And there are a number of different types of these sort of recurrent neural networks.

# One of the most popular is known as the long-short-term memory neural network, in other words known as lstm.

# But in general, these types of networks can be very very powerful whenever we're dealing with sequences.

# Whether they're sequences of images, or specially sequences of words, when it comes with dealing with natural
#language.
