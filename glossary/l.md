---
title: L
permalink: /l/

#=== THEMES
# = minima
layout: page
# = minimal mistake
# layout: archive-taxonomy
# layout: archive
# layout: categories
# layout: category
# layout: collection
# layout: compress
# layout: default
# layout: home
# layout: posts
# layout: search
## layout: single
# layout: splash
# layout: tag
# layout: tags

---

* toc
{:toc}

{% include links/all.md %}


# Label

 Name of a prediction in a supervised models. Correspond to a target attribute in unsupervised learning. Example of label: the agent-skill needed to result the customer's call.

 See also [L], [Data Point], [Labeling Function], [Supervised Learning], [Target Attribute]


# Labeling Function

 Use one or more function to label a sample. If you have more than one labeling function, use the majority rule, i.e. label the sample with the sample that has the maximum probability. :warning: A label is an enum, not a probability. Used by snorkel.

 See also [L], [Label], [Snorkel Program]


# Labeling Service

 Mechanical turk, crowd flower, instaML LOOP. Do you have the proper label? Have several people label the same image/entry and used the Dawid-Skene or majority vote algorithm!

 See also [L], [Dawid-Skene Algorithm], [Majority Vote Algorithm], [Unlabelled Data Algorithm]


# Lambda Model

 Beware, cannot use GPU for inference. ??? <== ????


# Language Model

 See also [L], [Language Modeling], [Large Language Model]


# Language Modeling

 Language modeling is the task of assigning a probability to a sequence of words in a text in a specific language. Simple language models can look at a word and predict the next word (or words) most likely to follow it, based on statistical analysis of existing text sequences. To create a language model that successfully predicts word sequences, you need to train it on large sets of data. Language models are a key component in natural language processing applications. You can think of them as statistical prediction machines, where you give text as input and get a prediction as the output. You’re probably familiar with this from the auto-complete feature on your smartphone. For instance, if you type “good,” auto-complete might suggest “morning” or “luck.”

 See also [L], [Language Model], [Large Language Model], [Natural Language Processing]


# Language Parsing

 ~ figuring out which group of words go together (as “phrases”) and which words are the subject or object of a verb. The NLP parser separates a series of text into smaller pieces based on the grammar rules. If a sentence that cannot be parsed may have grammatical errors.

 See also [L], [NLP Benchmark]


# Large Language Model

 See also [Language Model], [Language Modeling]


# Latent Diffusion Model

# LDM

 ![]( {{site.assets}}/l/latent_diffusion_model.png ){: width="100%"}

 The overall model will look like this:
  * you will have your initial image here X, and encode it into an information-dense space called the latent space, Z. This is very similar to a GAN where you will use an encoder model to take the image and extract the most relevant information about it in a sub-space, which you can see as a downsampling task. Reducing its size while keeping as much information as possible.
  * You are now in the latent space with your condensed input. You then do the same thing with your conditioning inputs, either text, images, or anything else,
  * and merge them with your current image representation. WE condition LDMs either via concatenation or by a more general cross-attention mechanism. This attention mechanism will learn the best way to combine the input and conditioning inputs in this latent space. Adding attention, a transformer feature, to diffusion models. These merged inputs are now your initial noise for the diffusion process. Then, you have the same diffusion model I covered in my Imagen video but still in this sub-space.
  * Finally, you reconstruct the image using a decoder which you can see as the reverse step of your initial encoder. Taking this modified and de-noised input in the latent space to construct a final high-resolution image, basically upsampling your result.
 And voilà! This is how you can use diffusion models for a wide variety of tasks like super-resolution, inpainting, and even text-to-image with the recent stable diffusion open-sourced model through the conditioning process while being much more efficient and allowing you to run them on your GPUs instead of requiring hundreds of them. 

 {% pdf "{{site.assets}}/l/latent_diffusion_model_paper.pdf" %}

 {% youtube "https://www.youtube.com/watch?v=RGBNdD3Wn-g" %}

 More at :
  * [https://pub.towardsai.net/latent-diffusion-models-the-architecture-behind-stable-diffusion-434ba7d91108](https://pub.towardsai.net/latent-diffusion-models-the-architecture-behind-stable-diffusion-434ba7d91108)
  * [https://www.louisbouchard.ai/latent-diffusion-models/](https://www.louisbouchard.ai/latent-diffusion-models/)
  * [code - https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

 See also [L], [Conditioning], [Cross-Attention], [Diffusion Model], [Diffusion Process], [Image Decoder], [Image Encoder], [Latent Space], [Pixel Space], [U-Net Architecture]


# Latent Dirichlet Allocation

# LDA

 Used as a topic modeling technique that is it can classify text in a document to a particular topic. It uses Dirichlet distribution to find topics for each document model and words for each topic model. Johann Peter Gustav Lejeune Dirichlet was a German mathematician in the 1800s who contributed widely to the field of modern mathematics. There is a probability distribution named after him ‘Dirichlet Distribution’ which is the basis of Latent Dirichlet Allocation (--LDA--).


# Latent Perturbation

 Used yo find out what the latent variable are/mean in a latent variable model. :warning: The model learns by itself that those are important variables based on the provided training sample. :warning: The loss function defines what is learned and HOW it learns it! Latent perturbation is useful to see how entangled or disentangled latent variables are.

 See also [L], [Disentangled Variational Autoencoder], [Latent Variable], [Latent Variable Model]


# Latent Space

 `~ A compressed/downsampled space that contains as much information as possible space`. Formally, a latent space is defined as an abstract multi-dimensional space that encodes a meaningful internal representation of externally observed events. Samples that are similar in the external world are positioned close to each other in the latent space. To better understand the concept, let’s think about how humans perceive the world. We are able to understand a broad range of topics by encoding each observed event in a compressed representation in our brain. For example, we don’t keep in mind every detail of the appearance of a dog to be able to recognize a dog in the street. As we can see in the illustration below, we keep an internal representation of the general appearance of a dog:

 ![]( {{site.assets}}/l/latent_space_in_mind.png ){: width="100%"}

 In a similar way, the latent space tries to provide a compressed understanding of the world to a computer through a spatial representation.
 Deep learning has revolutionized many aspects of our life with applications ranging from self-driving cars to predicting serious diseases. Its main goal is to transform the raw data (such as the pixel values of an image) into a suitable internal representation or feature vector from which the learning subsystem, often a classifier, could detect or classify patterns in the input. So, we realize that deep learning and latent space are strongly related concepts since the internal representations of the former constitute the latter. As we can see below, a deep learning model takes as input raw data and outputs discriminative features that lie in a low-dimensional space referred to as latent space. These features are then used to solve various tasks like classification, regression, or reconstruction:

 ![]( {{site.assets}}/l/latent_space.png ){: width="100%"}

 To better understand the importance of latent space in deep learning, we should think of the following question: Why do we have to encode the raw data in a low-dimensional l atent space before classification, regression, or reconstruction?
 The answer is data compression. Specifically, in cases where our input data are high-dimensional, it is impossible to learn important information directly from the raw data. 

 More at:
  * [https://ai.stackexchange.com/questions/11285/what-is-the-difference-between-latent-and-embedding-spaces](https://ai.stackexchange.com/questions/11285/what-is-the-difference-between-latent-and-embedding-spaces)

 See also [L], [Convolutional Neural Network], [Encoder]][Latent Variable], [Latent Variable Model], [Latent Vector], [Pixel Space], [Representation Space], [Semantic Space], [Word Embedding Space]


# Latent Space Compression

 {% youtube "https://www.youtube.com/watch?v=NqmMnjJ6GEg" %}

 See also [L], [Encoder], [Latent Space]


# Latent Variable

 Myth of the cave = where observation are only a projection of other objects. Latent variables are not directly observable, but are the true explanatory factors (that are casting the shadows that we see !)

 ![]( {{site.assets}}/l/latent_variable.png ){: width="100%"}

 See also [L], [Latent Space], [Latent Variable Model]


# Latent Variable Model

 See also [L], [Autoencoder], [Generative Adversarial Network], [Latent Space], [Latent Variable], [Variational Autoencoder]


# Latent Vector

 The input of a GAN acts as a latent vector since it encodes the output image \mathbf{G(z)} in a low-dimensional vector \mathbf{z}. To verify this, we can see how interpolation works in the latent space since we can handle specific attributes of the image by linearly modifying the latent vector. In the image below, we can see how we can handle the pose of a face by changing the latent vector of the GAN that generates it: 

 ![]( {{site.assets}}/l/latent_vector.png ){: width="100%"}

 See also [L], [Latent Space]


# Layer

 See also [L], [Hidden Layer], [Input Layer], [Output Layer]


# LeakyReLU Activation Function

 See also [L], [Activation Function], [Exploding Gradient Problem], [ReLU Activation Function], [Vanishing Gradient Problem]


# Learning Method

 All of those are or should be machine learning algorithm type! Here is a non-exhaustive list:
  * experience : learn from the past/data
  * [unsupervised learning] : try, fail, learn from failures ? Takes a long time / many iteration!
  * [imitation learning] : clone behavior of experts <== good to get started, but do you understand?
  * [supervised learning] : with a teacher
  * [reinforcement learning] : reward based
  * [task-based learning] : focus on goal, use all of your skills to complete it and develop new ones (be motivated to find new skills)
  * feedback-based learning : get feedback from the crowd (experts and non-experts), select the feedback you want -- always try your best --> develop a persona
  * [transfer learning] : priors + I learned that concept before, no need to relearn
  * [weak-supervised learning] : augment the data (i.e. create data!) which has been labeled (supervised)
  * [semi-supervised learning] : label existing data based on data that has been labeled

 See also [L], [Feedback-Based Learning], [Machine Learning Type], [Self-Supervised Learning]


# Learning Process

 * Changing weights in an ANN using backpropagation

 See also [L], [Backpropagation]


# Learning Rate

 The learning rate `controls how rapidly the weights and biases of each network are updated during training`. A higher learning rate might allow the network to explore a wider set of model weights, but might pass over more optimal weights. Iterative learning: (1) observe difference between predicted answer, and correct answer. (2) Adjust the model a 'small amount' (at each pass /epoch) to make the prediction closer to the correct answer. Size of update at each iteration. Relative weight of new iteration vs old iterations?

 ```
new_value = expected_value + alpha * ( observed_error )
          = expected_value + alpha * ( observed_value - expected_value)
          = (1 - alpha) * expected_value + alpha * observed_value

with alpha = learning_rate
 ```

 Beware:
  * the learning rate, alpha, is between 0 and 1
  * if alpha = 1  ==> immediately forget the past!
  * if alpha = 0  ==> oblivious to observation = no change!
  * A starting value can be between 0.01 and 0.1 which implies that updates with be between 1% and 10% of the observed error.

 See also [L], [Gradient Descent], [Hyperparameter], [Loss Function], [Prior], [Transfer Learning]


# Learning Strategy

 See also [L], [Learning Method], [Learning Rate], [Learning Velocity]


# Learning Velocity

 How fast you learn to execute a task.

 See also [L], [Learning Rate], [Sample Strategy], [Sample Efficiency]


# LIDAR 

 See also [L], [Autonomous Vehicle]


# LightGBM

 An ensemble method.
 LightGBM, short for light gradient-boosting machine, is a free and open-source distributed gradient-boosting framework for machine learning, originally developed by Microsoft. It is based on [Decision tree algorithms][Decision Tree] and used for ranking, [classification] and other machine learning tasks. The development focus is on performance and scalability.

 More at:
  * [https://lightgbm.readthedocs.io/en/latest/index.html](https://lightgbm.readthedocs.io/en/latest/index.html)
  * [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
  * [https://en.wikipedia.org/wiki/LightGBM](https://en.wikipedia.org/wiki/LightGBM)

 See also [L], [Ensemble Method]


# Linear Autoencoder

 Let’s first suppose that both our encoder and decoder architectures have only one layer without non-linearity (linear autoencoder). Such encoder and decoder are then simple linear transformations that can be expressed as matrices. In such situation, we can see a clear link with PCA in the sense that, just like PCA does, we are looking for the best linear subspace (hidden state?) to project data on with as few information loss as possible when doing so. Encoding and decoding matrices obtained with PCA define naturally one of the solutions we would be satisfied to reach by gradient descent, but we should outline that this is not the only one.

 See also [L], [Autoencoder], [Principal Component Analysis]


# Linear Discriminant Analysis

# LDA

 Linear Discriminant Analysis(or LDA for short) was proposed by Ronald Fisher which is a Supervised Learning algorithm. It means that you must use both features and labels of data to reduce dimension while PCA only uses features. Another key point : the purpose of LDA is to find a new space in which reduced-dimension dataset is good for classification task. To meet this goal, LDA uses 2 metrics: Within-class variance and Between-class variance. The core idea is quite straightforward: finding vectors w which maximize the distance between mean vectors of 2 classes and minimize the variance within each class. A little bit explanation: within-class variance stands for scatter. The smaller this quantity, the lower data points scatter and vice versa. We want to classify classes, of course we have to maximize the distance between each class, that's why maximizing distance between mean vectors. However, we also need to take into account the scatter of data.The greater the within-class variance, the more data points of 2 classes overlap and it culminates in bad result for classification. Now you know why we need to minimize the scatter.

 More at:
  * [https://iq.opengenus.org/pca-vs-lda/](https://iq.opengenus.org/pca-vs-lda/)

 See also [L], [Dimensionality Reduction], [Principal Component Analysis]


# Linear Regression

 Find an equation. Best fit. Ex: https://www.desmos.com/calculator/fmhotfn3qm.

 Not how long it will take for my car to stop given my speed (linear regression), but whether I am going to hit the tree or not (logistic regression).

 ![]( {{site.assets}}/l/linear_regression.png ){: width="100%"}

 More at :
  * introduction - [https://towardsdatascience.com/linear-regression-the-actually-complete-introduction-67152323fcf2](https://towardsdatascience.com/linear-regression-the-actually-complete-introduction-67152323fcf2)
  * code - [https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)

 See also [L], [Classification], [Multiple Linear Regression], [Non-Linear Regression], [Prediction Error], [Regression]


# LLaMA Model

 Using the scaling method described in [Chinchilla][Chinchilla Model]
 65 Billion parameters.

 {% youtube "https://www.youtube.com/watch?v=E5OnoYF2oAk" %}

 {% pdf "{{site.assets}}/l/llama_model_paper.pdf" %}

 More at:
  * [https://ai.facebook.com/blog/large-language-model-llama-meta-ai/](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
  * paper [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)

 See also [L], ...


# Log Loss Function

 See [Binary Cross-Entropy Loss Function]


# Log Transformation

 A [Feature Distribution Transformation]

 {% youtube "https://www.youtube.com/watch?v=LCDiQxB5S84" %}

 See also [L], ...

# Logical Reasoning

 If-then-else rules used in [expert systems][Expert System]

 ```
# Knowledge base
All men are mortal
# Input
Aristotle is a men
# Inference
==>
# New fact
Aristotle is mortal!

# If Aristotle is man AND all men are mortal, then Aritotle is mortal!
 ```

 Ex: personal assistant with memory and can infer from dialog new things (i.e graph network?) !

 See also [L], [Reasoning]


# Logistic Regression

 Not how long it will take for my car to stop given my speed (linear regression), but whether I am going to hit the tree or not (logistic regression). used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned a probability between 0 and 1 and the sum adding to one.

 ![]( {{site.assets}}/l/logistic_regression_data.png ){: width="100%"}

 ![]( {{site.assets}}/l/logistic_regression_fitting.png ){: width="100%"}

 Beware:
  * To turn a probability into a classification, we need to use a threshold (P>0.5 or P<0.5)!
   * What about using a different P threshold? ==> multiple confusion matrix ==> ROC Curve

 See also [L], [ML Algorithm Evaluation], [Regression], [ROC Curve]


# Long Short Term Memory Cell

# LSTM Cell

 * overview
  * input signal = previous state + new info
  * blue activation function = sigmoid activation function = switch (keep or forget, impact or no-impact)
  * red activation function = tanh --> add, no effect, or substract
 * cell state = highway that transfers information down to the sequence chain = memory of the network
 * gates
  * forget gate = decide which information should be thrown (=0) out or kept (=1) away (information = previous state + new input info) (sigmoid = 1 --> keep or = 0 forget!)
  * input gate = update the cell state with (transformed) input signal
  * output gate used to compute the hidden state = tanh(cell state) gated by input signal

 ![]( {{site.assets}}/l/long_short_term_memory_cell.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=8HyCNIVRbSU" %}

 {% youtube "https://www.youtube.com/watch?v=S27pHKBEp30" %}

 See also [L], [Hidden State], [LSTM Network]


# Long Short Term Memory Network

# LSTM Network

 `A multi-layer Recurrent Neural Network, aka RNN, where a neuron is feeding its output to self, remembers its previous output. Good for sequences`. Used in speech recognition, Text to speech, handwriting recognition. Started becoming widespread in 2007. They are a type of Recurrent Neural Network that can efficiently learn via gradient descent. Using a gating mechanism, LSTMs are able to recognise and encode (short and very) long-term patterns (basic RNN can only remember a given length, i.e have short term memory because of vanishing gradient problem). LSTMs are extremely useful to solve problems where the network has to remember information for a long period of time as is the case in music and text generation.
 
 LSTMs also have the RNN chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

 ![]( {{site.assets}}/l/long_short_term_memory_unrolled.png ){: width="100%"}

 with

 ![]( {{site.assets}}/l/long_short_term_memory_repeating_module.png ){: width="100%"}

 In its chain, a LSTM can optionally use a Gated Recurrent Unit (GRU) cell, which is simpler than the one represented above.

 {% youtube "https://www.youtube.com/watch?v=WCUNPb-5EYI" %}

 {% pdf "{{site.assets}}/l/long_short_term_memory_paper.pdf" %}

 Beware:
  * `Are now deprecated by attention-based models, such as transformers? Yes`

 More at
  * LSTM with keras - [https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
  * [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

 See also [L], [Attention-Based Model], [Gated Recurrent Unit Cell], [Gradient Descent], [Recurrent Neural Network], [Transformer Model], [Vanishing Gradient Problem]


# Loss Function

 Loss function is used for parameter estimation.

 `The loss function must encode what you want your model to do!` The loss function will take two items as input: the output value of our model and the ground truth expected value. The output of the loss function is called the loss which is a measure of how well our model did at predicting the outcome. A high value for the loss means our model performed very poorly. A low value for the loss means our model performed very well. In most learning networks, error is calculated as the difference between the actual output y and the predicted output ŷ. The function that is used to compute this error is known as Loss Function also known as Cost function. The loss function allows us to find the best line. The model is iterated to minimize the loss function using the gradient descent algorithm. Selection of the proper loss function is critical for training an accurate model. Certain loss functions will have certain properties and help your model learn in a specific way. Some may put more weight on outliers, others on the majority.

 The most common loss functions are:
  * [Mean Squared Error (MSE)][Mean Square Error Loss Function]: Used in a linear regression, the best line is the one that minimize the root-mean square of the error.
  * [Mean Absolute Error (MAE)][Mean Absolute Error Loss Function]
  * [Hinge Loss Function]
  * [Huber Loss Function]
  * [0-1 Loss Function] : 0=correct 1=not-correct classification
  * [binary cross-entropy loss function] (aka Log loss function) : Used with logistic regression because the logistic regression function (sigmoid or ?) is not linear and loss function needs to have a single minimum
  * [cross-entropy loss function]
  * custom function !
 Choose your loss function based on
  * the original estimator function (?) e.g. lineear or sigmoid
  * must have a global minimum and not local ones

 More at :
  * choosing a loss function - [https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

 See also [L], [Activation Function], [Backpropagation], [Discriminator], [Gradient Descent], [Linear Regression], [Optimizer], [Prediction Error], [Representation Space], [Residual]


# Loss Graph

 ![]( {{site.assets}}/l/loss_graph.png ){: width="100%"}

 See also [L], [Discriminator  Loss], [Generator Loss], [Loss Function]
