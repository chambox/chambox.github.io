# Neural networks in keras, a first look

Artifical intelligence (AI)  has featured on the news several times in the past few years. AI , machine  learning and deep learning feature in countless articles often outside technological oriented publications. You will hear often of self driving cars, chatbots, robots replacing humans in factories and many other stories, with AI as the vilan or the hero. At the heart of this technological revolution is neural networks (NN) which we briefly introduce in this article. The focus here is to introduce  NNs  with python using the keras API. 

The problem we will solve here is to classify grayscale images of handwritten digits (28x28 pixels)  into their 10 categories which ranges from 0-9. We a will use a classic dataset in the machine learning community, namely MNIST. MNIST has 60k training images and another 10k images for testing. Let's import the MNIST dataset. 


```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
```

`train_images` and `train_labels` represent the data that the model will learn from. We will then use `test_images` and `test_labels` to test the model. The data  object is a numpy array with each image of dimension 28x28 mapped to one and only one label in the range [0,9].  The training data looks like this:


```python
train_images.shape
```




    (60000, 28, 28)




```python
train_labels
```




    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)



The test data:


```python
test_images.shape
```




    (10000, 28, 28)




```python
test_labels
```




    array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)



Let's now create an architecture for the network. 

This will be the workflow:

1. Create an architecture for the network, and create a network object
2. Compile the network object
3. Prepare the training and test datasets
4.  Fit the network object to data


```python

```
