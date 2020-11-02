# Pneumonia Detection: A Computer Vision Study

## Our Data

Our dataset for this problem was provided on [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

> There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). <br>
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.<br>
For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

This data is a classic example of an **Image Classification** problem. In this notebook we will demonstrate the ability for an **Artificial Neural Network** to *learn* how to identify pneumonia in patient X-rays. The business applications of these sorts of algorithms are numerous and represent possible millions of dollars in savings for our health care system.

# Neural Networks - PyToch and TensorFlow

Data Science offer several very competent and useful frameworks for building, maintaining, and deploying Neural Networks. Of these many frameworks two have set themselves apart from the rest of the chaff by being flexible and easily put into a production environment. These are **PyTorch** and **Tensorflow**. **PyTorch** being developed and maintained by Facebook and **Tensorflow** being developed and maintained by Google. These are the two most popular and widely used frameworks. 

### What is a Neural Network? 

* **Artificial Neural Networks** at the most base level explanation is an attempt to model how the human brain functions on a level that concerns learning in order to make predictions or draw inferences.


* A deeper explanation describes that an **ANN** is comprised of **artificial neurons** that connect to other **artificial neurons** through **synaptic connections**. These **neurons** have an **edge** that is **weighted** through various transformations. Being passed through a **weighted edge** can decrease the strength of the signal between **layers** of **neurons**. Each **layer** can have a function that acts as a sort of gatekeeper which only allows a **neuron** to fire if it's signal reaches a certain threshold, this is known as an **activation function**. This is the analogous action that mimics learning. The final **layer** of **neurons** represents the **output** of the network. Predictions of the **ANN** are passed through a mathematical function that calculates the cost of the prediction, this cost function is known as the **loss function**. This **loss function** operates with **back-propogation** in order to pass information back through the **model** in order to calculate **gradients** and have the model learn.

### PyTorch and TensorFlow

The first model below is a **PyTorch** model. 

**PyTorch**, as previously stated, is a neural network framework developed and maintained by Facebook. The current trend shows that **PyTorch** and **Tensorflow** share similar interest within websearches, with **PyTorch** pulling slightly ahead. As seen in this image provided by Google Trends: 

<p align="center">
  <img src="images/pyttf.png" />
  <img src="images/pyttf_chart.png" />
</p>

This increase in interest can be attributed to several things, including adoption of **PyTorch** by Silicon Valley front-runners like Tesla. 

It's also because of the difference in architecture of the frameworks from not just a data science standpoint, but also a software engineering standpoint. **Tensorflow** is a statically graphed architecture, and **PyTorch** is a dynamic architecture. An excellent explanation was posted on [StackOverflow](https://stackoverflow.com/questions/46154189/what-is-the-difference-of-static-computational-graphs-in-tensorflow-and-dynamic):

> Both frameworks operate on tensors and view any model as a directed acyclic graph (DAG), but they differ drastically on how you can define them.<br><br>
TensorFlow follows ‘data as code and code is data’ idiom. In TensorFlow you define graph statically before a model can run. All communication with outer world is performed via tf.Session object and tf.Placeholder which are tensors that will be substituted by external data at runtime. <br><br>
In PyTorch things are way more imperative and dynamic: you can define, change and execute nodes as you go, no special session interfaces or placeholders. Overall, the framework is more tightly integrated with Python language and feels more native most of the times. When you write in TensorFlow sometimes you feel that your model is behind a brick wall with several tiny holes to communicate over. Anyways, this still sounds like a matter of taste more or less. <br><br>
However, those approaches differ not only in a software engineering perspective: there are several dynamic neural network architectures that can benefit from the dynamic approach. Recall RNNs: with static graphs, the input sequence length will stay constant. This means that if you develop a sentiment analysis model for English sentences you must fix the sentence length to some maximum value and pad all smaller sequences with zeros. Not too convenient, huh. And you will get more problems in the domain of recursive RNNs and tree-RNNs. Currently Tensorflow has limited support for dynamic inputs via Tensorflow Fold. PyTorch has it by-default.

<p align="center">
    <img src="Images/graphmap.gif" />
</p>
This `gif` is an example of how any neural network model can be displayed as a **directed acyclic graph**.

### Model Details

We have chose to code our model in a way that takes advantage of the **Object-Oriented** nature of the framework. There-by making our model easily extensible, increasing our debugging capabilities, helping with tunability of our model's parameters, and finally making the model more deployable from a production standpoint.

#### Convolutional Neural Network

We have chosen to make our model a slightly more advanced form of **ANN**, the **Convolutional Neural Network**. We chose this because **CNN**s are generally better at **image classification** tasks. **CNN**s are named after layers that are within their architecture. This flavor of **ANN** make use of **Convolutional Layers**, which in the simplest terms apply **convolutional filters** to an image as they scan over it. These **convolutional filters** can scan an image for generalized features, and as the information is passed forward in the sequential model it can pick out more and more complex features within your image. This also allows for **feature reduction** within the model, which is something that a **Fully Connected** (or **Dense**) **layer** cannot deal with innately. 

According to [Machine Learning Mastery](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/), a **convolution** is:

> In the context of a convolutional neural network, a convolution is a linear operation that involves the multiplication of a set of weights with the input, much like a traditional neural network. Given that the technique was designed for two-dimensional input, the multiplication is performed between an array of input data and a two-dimensional array of weights, called a filter or a kernel. <br><br>
This systematic application of the same filter across an image is a powerful idea. If the filter is designed to detect a specific type of feature in the input, then the application of that filter systematically across the entire input image allows the filter an opportunity to discover that feature anywhere in the image. This capability is commonly referred to as translation invariance, e.g. the general interest in whether the feature is present rather than where it was present.

#### Activation Function

An **activation function** is a simple concept that relates back to the idea of the neural network being an analogous system to the brain. The function is a mathematical equation that represents a threshold at which the neuron will fire and thus send its signal on to the next layer. This can be numerous things, such as a linear function that makes the output proportional to the input, or a sigmoid function that returns a value between 0 and 1. These are just two of the numerous functions that can represent activations of the neurons.

#### Loss Function

**Neural Networks** are typically trained by using **stochastic gradient descent** and this requires a method to measure the loss of the network. A neural network can be cast as learning based on optimization, where you are optimizing the weights of neurons. It optimizes these weights via **gradient descent** and associated algorithms, these algorithms seek to reduce the **error** of the next step by finding optimal weights for the neurons. This **error term** can also be referred to as a **loss**. This **loss** is calculated by a **loss function**, which is a distillation of the model itself to a single value. 

> The cost function reduces all the various good and bad aspects of a possibly complex system down to a single number, a scalar value, which allows candidate solutions to be ranked and compared.<br><br>
— Page 155, Neural Smithing: Supervised Learning in Feedforward Artificial Neural Networks, 1999.

#### Optimizer Function

**Loss functions** have a function that are closely akin to them, which with they work hand in hand to make an **ANN** work. This function is known as the **Optimizer function**. As we discussed before learning through an **ANN** is a function of optimization of loss. The **Optimizer** takes the information learned through the **loss function** and uses it to apply optimizations to the **ANN**, so that it may learn. It takes the guidance of the **loss function** and is able to tell the network where to change weights and by how much, so that learning may start taking place within the **ANN**.

#### Training and Testing

After we have initialized our **network** so that we have all of the required **layers**, **activation**, **loss**, and **optimizer** functions, we must finally feed **inputs** into our **ANN** so that it can utilize all of the aforementioned mathematical transformations to learn. The **ANN** passes the **inputs** through and produces **predictions** that are then measured by the **loss function**, so that it may **optimize** the weights. **Training**, as discussed previously, is ultimately a function of optimization.

**Training** generally takes place on a shuffled sub-set of the dataset so that the **ANN** may learn to generalize from the data. Datasets are generally subdivided into a **training set** and a **test set**. The **test set** is never seen by the **ANN**, so that it can be used to evaluate how well the **ANN** learned to generalize from the provided training data.