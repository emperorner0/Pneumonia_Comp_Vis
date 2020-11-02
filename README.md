Data Science offer several very competent and useful frameworks for building, maintaining, and deploying Neural Networks. Of these many frameworks two have set themselves apart from the rest of the chaff by being flexible and easily put into a production environment. These are **PyTorch** and **Tensorflow**. **PyTorch** being developed and maintained by Facebook and **Tensorflow** being developed and maintained by Google. These are the two most popular and widely used frameworks. 

What is a Neural Network though? 

* **Artificial Neural Networks** at the most base level explanation is an attempt to model how the human brain functions on a level that concerns learning in order to make predictions or draw inferences.


* A deeper explanation describes that an **ANN** is comprised of **artificial neurons** that connect to other **artificial neurons** through **synaptic connections**. These **neurons** have an **edge** that is **weighted** through various transformations. Being passed through a **weighted edge** can decrease the strength of the signal between **layers** of **neurons**. Each **layer** can have a function that acts as a sort of gatekeeper which only allows a **neuron** to fire if it's signal reaches a certain threshold, this is known as an **activation function**. This is the analogous action that mimics learning. The final **layer** of **neurons** represents the **output** of the network. Predictions of the **ANN** are passed through a mathematical function that calculates the cost of the prediction, this cost function is known as the **loss function**. This **loss function** operates with **back-propogation** in order to pass information back through the **model** in order to calculate **gradients** and have the model learn.

The first model below is a **PyTorch** model. 

**PyTorch**, as previously stated, is a neural network framework developed and maintained by Facebook. The current trend shows that **PyTorch** and **Tensorflow** share similar interest within websearches, with **PyTorch** pulling slightly ahead. As seen in this image provided by Google Trends: 

![Google Trends](images/pyttf.png)
![Google Trends](images/pyttf_chart.png)

This increase in interest can be attributed to several things, including adoption of **PyTorch** by Silicon Valley front-runners like Tesla. 

It's also because of the difference in architecture of the frameworks from not just a data science standpoint, but also a software engineering standpoint. **Tensorflow** is a statically graphed architecture, and **PyTorch** is a dynamic architecture. An excellent explanation was posted on [StackOverflow](https://stackoverflow.com/questions/46154189/what-is-the-difference-of-static-computational-graphs-in-tensorflow-and-dynamic):

> Both frameworks operate on tensors and view any model as a directed acyclic graph (DAG), but they differ drastically on how you can define them.<br><br>
TensorFlow follows ‘data as code and code is data’ idiom. In TensorFlow you define graph statically before a model can run. All communication with outer world is performed via tf.Session object and tf.Placeholder which are tensors that will be substituted by external data at runtime. <br><br>
In PyTorch things are way more imperative and dynamic: you can define, change and execute nodes as you go, no special session interfaces or placeholders. Overall, the framework is more tightly integrated with Python language and feels more native most of the times. When you write in TensorFlow sometimes you feel that your model is behind a brick wall with several tiny holes to communicate over. Anyways, this still sounds like a matter of taste more or less. <br><br>
However, those approaches differ not only in a software engineering perspective: there are several dynamic neural network architectures that can benefit from the dynamic approach. Recall RNNs: with static graphs, the input sequence length will stay constant. This means that if you develop a sentiment analysis model for English sentences you must fix the sentence length to some maximum value and pad all smaller sequences with zeros. Not too convenient, huh. And you will get more problems in the domain of recursive RNNs and tree-RNNs. Currently Tensorflow has limited support for dynamic inputs via Tensorflow Fold. PyTorch has it by-default.

![Directed Graph](images/graphmap.gif)
This `gif` is an example of how any neural network model can be displayed as a direct acyclic graph.

We have chose to code our model in a way that takes advantage of the **Object-Oriented** nature of the framework. There-by making our model easily extensible, increasing our debugging capabilities, helping with tunability of our model's parameters, and finally making the model more deployable from a production standpoint.

We have chosen to make our model a slightly more advanced form of **ANN**, the **Convolutional Neural Network**. We chose this because **CNN**s are generally better at **image classification** tasks. **CNN**s are named after layers that are within their architecture. This flavor of **ANN** make use of **Convolutional Layers**, which in the simplest terms apply **convolutional filters** to an image as they scan over it. These **convolutional filters** can scan an image for generalized features, and as the information is passed forward in the sequential model it can pick out more and more complex features within your image. This also allows for **feature reduction** within the model, which is something that a **Fully Connected**(or **Dense**) **layer** cannot deal with innately. 