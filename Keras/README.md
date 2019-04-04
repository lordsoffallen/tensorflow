# Building Machine Learning Models Using Keras 2.0

It containes multiple projects using ResNet50, CIFAR10, VGG16, pretrained models, data preprocessing, importing and exporting trained models.
Use the code below to upload the model to google cloud.


# Using CNN in Layers

If we only train the neural network with pictures of numbers that are perfectly centered, the neural network will get confused if it 
sees anything else. For example, if we pass in an image where the object is not centered, the neural network won't be able to make a 
good prediction. But the object could appear anywhere in the image. It could just as easily appear at the bottom.  We need to improve 
our neural network so that it can recognize objects in any position. This is called **translation invariance**.  

This is the idea that a machine learning model should recognize an object no matter where it is moved within the image. Moving the object around the image doesn't change the fact that it's still just an object. The solution is to add a new type of layer to our neural network called the convolutional layer. Unlike a normal dense layer, where every node is connected to every other node, this layer breaks apart the image in a special way so that it can recognize the same object in different positions. The first step is to break the image into small, overlapping tiles.  

We do this by passing a small window over the image. Each time it lands somewhere, we grab a new image tile. We repeat this until we've covered the entire image. Next, we'll pass each image tile through the same neural network layer. Each tile will be processed the same way and we'll save a value each time. In other words, we're turning the image into an array, where each entry in the array represents whether or not the neural network thinks a certain pattern appears at that part of the image. Next, we'll repeat this exact process again. But this time, we'll use a different set of weights on the nodes in our neural network layer.
This will create another feature map that tells us whether or not a certain pattern appears in the image. But because we're using different weights, they'll be looking for a different pattern than the first time. We can repeat this process several times until we have several layers in our new array. This turns our original array into a 3D array. Each element in the array represents where a certain pattern occurs. But because we are checking each tile of the original image, it doesn't matter where in the image a pattern occurs. We can find it anywhere. This 3D array is what we'll feed into the next layer of the neural network. It will use this information to decide which patterns are most important in determining the final output.  

Adding a convolutional layer makes it possible for our neural network to be able to find the pattern, no matter where it appears in an image. In this example, we have only one convolutional layer. But normally, we'll have several convolutional layers that repeat this process multiple times. The rough idea is that we keep squishing down the image with each convolutional layer while still capturing the most important information from it. By the time we reach the output layer, the neural network will have been able to identify whether or not the object appeared. Convolutional neural networks are a secret weapon for image detection, they make it possible to efficiently detect objects.
It's the standard approach for building image recognition systems
