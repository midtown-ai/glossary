---
title: P
permalink: /p/

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


# Pathways Language Model

# PaLM Model

 {% pdf "{{site.assets}}/p/palm_model_paper.pdf" %}

 More at :
  * [https://medium.com/@tech_optimist/palm-on-my-forehead-not-another-large-language-model-6dddd641211b](https://medium.com/@tech_optimist/palm-on-my-forehead-not-another-large-language-model-6dddd641211b)

 See also [P], [Chain Of Thought Prompting], [Pathways Model Architecture]


# Parameter

 A model parameter is something that the ML can learn from the data. For example, the weight of an input in a perceptron. Indeed the model has to have parameters to make predictions. This "parameters" are not set by humans. Hyperparameters cannot be learned from the data and are set by humans. Ex: number of layers in the neural network. 
  * GPT-3 possesses 175 billion weights connecting the equivalent of 8.3 million neurons arranged 384 layers deep.

 See also [P], [Hyperparameter]


# Passive Learning

 The main hypothesis in active learning is that if a learning algorithm can choose the data it wants to learn from, it can perform better than traditional methods with substantially less data for training. But what are these traditional methods exactly? These are tasks which involve gathering a large amount of data randomly sampled from the underlying distribution and using this large dataset to train a model that can perform some sort of prediction. You will call this typical method passive learning. One of the more time-consuming tasks in passive learning is collecting labelled data. In many settings, there can be limiting factors that hamper gathering large amounts of labelled data.

 See also [P], [Active Learning], [Random Sampling]


# Pathways Model Architecture

 An architecture to support (1) transfer learning, (2) multimodal learning, (3) Sparse activation, i.e NOT dense networks/models

  {% youtube "https://www.youtube.com/watch?v=Nf-d9CcEZ2w" %}

  More at 
   * [https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

  See also [P], [Google Company], [Multimodal Learning], [PaLM Model], [Sparse Activation]


# Perceiver IO Model

 Product arbitrary size outputs - reconstructing the input

 ![]( {{site.assets}}/p/perceiver_io_model.png ){: width="100%"}


 See also [P], [Attention-Based Model], [Perceiver Model]


# Perceiver Model

  * convert input to simple 2D byte array
  * Encode information about the input array using a smaller number of latent feature vectors using transformer-style attention
  * final aggregation down to a category label
  . :warning: Used for classification

 ![]( {{site.assets}}/p/perceiver_model.png ){: width="100%"}

 See also [P], [Attention-Based model], [Transformer Model]


# Perceptron

 A neural network consisting of only 1 layer and 1 neuron.

 See also [P], [Artificial Neuron]


# Perplexity

 Used with language model. The smaller its value, the better.

 {% youtube "https://www.youtube.com/watch?v=NURcDHhYe98" %}

 See also [P], [Cross-Entropy], [Entropy], [Generative Model], 


# Personal Assistant

 See also [P], [Logic Model]


# Piano Roll

 As a 2D matrix, also known as a piano roll, with time on the horizontal and pitch on the vertical axis.

 See also [P], [U-Net Architecture]


# Pipe Mode

 The training set is streamed all the way to the inference point.

 See also [P], [File Mode]


# Pixel RNN

 Autoregressive models such as PixelRNN instead train a network that models the conditional distribution of every individual pixel given previous pixels (to the left and to the top). This is similar to plugging the pixels of the image into a char-rnn, but the RNNs run both horizontally and vertically over the image instead of just a 1D sequence of characters. PixelRNNs have a very simple and stable training process (softmax loss) and currently give the best log likelihoods (that is, plausibility of the generated data). However, they are relatively inefficient during sampling and don???t easily provide simple low-dimensional codes for images.

 More at:
  * paper - [https://arxiv.org/abs/1601.06759](https://arxiv.org/abs/1601.06759)

 See also [P], [RNN]


# Pixel Space

 In the pixel space, operations are done based on the values/parameters of pixels.

 See also [P], [Latent Space], [Space], 


# Pluribus Model

 Pluribus is a computer poker player using artificial intelligence built by Facebook's AI Lab and Carnegie Mellon University. Pluribus plays the poker variation no-limit Texas hold 'em and is "the first bot to beat humans in a complex multiplayer competition".

 More at:
  * [https://en.wikipedia.org/wiki/Pluribus_(poker_bot)](https://en.wikipedia.org/wiki/Pluribus_(poker_bot))

 See also [P], [Facebook Company]


# Point Estimator

 This definition of a point estimator is very general and allows the designer of an estimator great flexibility. While almost any function thus qualifies as an estimator, a good estimator is a function whose output is close to the true underlying ?? that generated the training data. Point estimation can also refer to estimation of relationship between input and target variables referred to as function estimation.

 See also [P], [Estimator], [Function Estimation]


# Point-E Model

 While recent work on text-conditional 3D object generation has shown promising results, the state-of-the-art methods typically require multiple GPU-hours to produce a single sample. This is in stark contrast to state-of-the-art generative image models, which produce samples in a number of seconds or minutes. In this paper, we explore an alternative method for 3D object generation which produces 3D models in only 1-2 minutes on a single GPU. Our method first generates a single synthetic view using a text-to-image diffusion model, and then produces a 3D point cloud using a second diffusion model which conditions on the generated image. While our method still falls short of the state-of-the-art in terms of sample quality, it is one to two orders of magnitude faster to sample from, offering a practical trade-off for some use cases. We release our pre-trained point cloud diffusion models, as well as evaluation code and models, at this https URL.

 More at:
  * paper - [https://arxiv.org/abs/2212.08751](https://arxiv.org/abs/2212.08751)
  * code - [https://github.com/openai/point-e](https://github.com/openai/point-e)
  * blog - [https://the-decoder.com/point-e-openai-shows-dall-e-for-3d-models/](https://the-decoder.com/point-e-openai-shows-dall-e-for-3d-models/)
  * article - [https://techcrunch.com/2022/12/20/openai-releases-point-e-an-ai-that-generates-3d-models/](https://techcrunch.com/2022/12/20/openai-releases-point-e-an-ai-that-generates-3d-models/)

 See also [P], [CLIP Model], [DALL-E Model]


# Policy Gradient Solution

 {% youtube "https://www.youtube.com/watch?v=YOW8m2YGtRg" %}

 {% youtube "https://www.youtube.com/watch?v=tqrcjHuNdmQ" %}

 More at : 
  * [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)
  * [https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
  * colab - [https://colab.research.google.com/github/DJCordhose/ai/blob/master/notebooks/rl/pg-from-scratch.ipynb](https://colab.research.google.com/github/DJCordhose/ai/blob/master/notebooks/rl/pg-from-scratch.ipynb)

 See also [P], [Proximal Policy Optimization], [Reinforcement Learning]


# Policy Neural Network

 See also [P], ...


# Polynomial Regression

 A form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. Examples:
   * cubic regression
   * quadratic regression

 ![]( {{site.assets}}/p/polynomial_regression.png ){: width="100%"}

 More at:
  * polynomial regression - [https://towardsdatascience.com/polynomial-regression-the-only-introduction-youll-need-49a6fb2b86de](https://towardsdatascience.com/polynomial-regression-the-only-introduction-youll-need-49a6fb2b86de)
  * code - [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures)

 See also [P], [Cubic Regression], [Quadratic Regression], [Regression]


# Positional Encoding

 When a model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the words/tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed. Example of positional encoding formula:

 ![]( {{site.assets}}/p/positional_encoding_formula.png ){: width="100%"}

 See also [P], [Multi-Head Attention]


# Positive Attribute

 A cat has legs, fur, 2 eyes, etc. Those are positive attributes. 

 See also [P], [Attribute], [Negative Attribute]


# Pre-Trained Model

 ~ a base model that support transfer learning. A pre-trained model, keeping with Gladwell???s 10,000 hours theory, is the first skill you develop that can help you acquire another one faster. For example, mastering the skill of solving math problems can help you more quickly acquire the skill of solving engineering problems. A pre-trained model is trained (by you or someone else) for a more general task and is then available to be fine-tuned for different tasks. Instead of building a model from scratch to solve your problem, you use the model trained on a more general problem as a starting point and give it more specific training in the area of your choice using a specially curated dataset. A pre-trained model may not be 100% accurate, but it saves you from reinventing the wheel, thus saving time and improving performance.

 Useful if 
  * retraining the model is expensive or time consuming
  * model architecture allow transfer learning (through fine-tuning). 
  . :warning: as far as I can tell, this is only possible today with transformer models (?)
 See also [P], [Supervised Fine-Tuning], [Transfer Learning], [Transformer Model], [Upstream Task]


# Precision

 Precision is the fraction of the tumors that were predicted to be malignant (of one call) that are actually malignant (of that class).
 
 ```
               TP               correctly identified       
Precision = --------- =    ----------------------------
             TP + FP           ground truth for class
 ```

 See also [P], [Confusion Matrix]


# Prediction

 The desired outcome of Machine Learning. Those predictions are based on hidden patterns in the training data. Low bias and low variance: every is close to the bulls eye. High bias and low variance: everything is clustered but with an offset from the bullseye (eg systematically to high). Low bias and high variance: appears to be centered on target but far from the bullseye. High variance and high bias: all over the place and not on target! Beware that the goal is not to minize the prediction error and not necessarily the bias and the variance.

 See also [P], [Bias], [Machine Learning], [Prediction Error], [Variance]


# Prediction Error

 To minimize the prediction error, you need a balancing act between the bias and the variance of the data.
 
 ![]( {{site.assets}}/p/prediction_error.png ){: width="100%"}
 
 ```
Prediction Error = actual_value - predicted_value
 ```

   * Relative Approximation Error
   * Root Mean Square
   * Mean Absolute Error
   * Mean Absolute Percentage Error

 ![]( {{site.assets}}/p/prediction_error_measurement.png ){: width="100%"}

 See also [P], [Bias], [Gradient Descent], [Loss Function], [Mean Absolute Error], [Mean Absolute Percentage Error], [Prediction], [Relative Approximation Error], [Root Mean Square Error], [Variance]


# Predictive Maintenance

 Predict when a part will fail. See [Regression], [Supervised Learning]


# Predictor Variable

 Input X / independent variable to estimate the dependent variable in regression.

 See also [P], [Regression], [Response Variable]


# Principal Component Analysis

# PCA

 The most popular dimensionality reduction method is Principal Component Analysis (PCA), which reduces the dimension of the feature space by finding new vectors that maximize the linear variation of the data. Why use? many situations that require low dimensional data including:
  * data visualisation
  * data storage
  * computation

 PCA can reduce the dimension of the data dramatically and without losing too much information when the linear correlations of the data are strong (PCA is based onlinear algebra!). :warning: And in fact you can also measure the actual extent of the information loss and adjust accordingly.) Principal Component Analysis (PCA) is a very popular technique used by data scientists primarily for dimensionality reduction in numerous applications ranging from stock market prediction to medical image classification. Other uses of PCA include de-noising and feature extraction. PCA is also used as an exploratory data analysis tool. To better understand PCA let???s consider an example dataset composed of the properties of trucks. These properties describe each truck by its color, size, compactness, number of seats, number of doors, size of trunk, and so on. Many of these features measured will be redundant and therefore, we should remove these redundancies and describe each truck with fewer properties. This is precisely what PCA aims to do. PCA is a technique used to compress d features into `p<<d` features, while preserving as much of the information as possible (~ compression !). A classic demonstration for PCA is given with images. A black and white image can be represented as an n by d matrix of integers, determining the grayscale of each pixel. PCA provides a low-rank (low-dimension) representation of that matrix that can be stored with (n+d) p numbers, instead of nd, such that the compressed image looks almost the same as the original. In the context of machine learning (ML), PCA is a dimension reduction technique. When the number of features is large, ML algorithms either are at risk of overfitting, or require too much time to train. To that end, `PCA can reduce the input dimension. The way PCA reduces the dimension is based on correlations. Two features are correlated if, given the value of one, you can make an educated guess about the value of the other`. PCA, with a target dimension of p, finds p features such that a linear function of these will do the best job of predicting the original d features. This kind of information-preserving objective makes the output of PCA useful to downstream tasks.

 See also [P], [Dimensionality Reduction], [Feature Extraction], [Linear Autoencoder], [Linear Discriminant Analysis], [ML Algorithm], [Overfitting], [Synthesized Variable]


# Prior

 `~ Prior knowledge of how the world works. Also the reason why human learn so much faster than computers!`. Fundamental concepts that are transfered from one task to the other that humans do not have to learn and therefore reduce the training time significantly. ~ stereotypes, shortcuts. Examples of priors are
  * concepts of objects
  * look similar = act similar
  * object semantics = a key 
  * object properties = a monster can kill me, a key can open a door --> go to key before door
  * affordance : In HCI, to ???afford??? means to ???invite??? or to ???suggest.??? Ex: when you see a button, you assume you can push it to trigger an action. If the button is next to an elevator, you assume it is used to call the elevator. If you recognize a ladder, you assume you can climb it. etc
 . :warning: Note that many priors are innate (look at faces) while others are learned (object permanence = when an object disappear from my view, it still exists) 
 
 {% youtube "https://www.youtube.com/watch?v=Ol0-c9OE3VQ" %}
 
 Beware priors can also slow down or even prevent learning
  * False Affordances. This type of affordance doesn???t have an apparent function when it is observed. That causes the individual to perceive possibilities of action that do not exist in reality. A placebo is an example of a false affordance.
  * Hidden Affordances. These affordances offer the potential for actions to be taken, but are not necessarily perceived by individuals within their environment. One might look at a book and think, ???I could use that for reading.??? It could also be used to smash a spider that looks like trouble.
  * Perceptible Affordances. These clues offer information to individuals that allows them to understand what actions can be taken and encourages them to take that action.A

 More at:
  * [https://rach0012.github.io/humanRL_website/](https://rach0012.github.io/humanRL_website/)

 See also [P], [Learning Rate], [Transfer Learning]


# Prompt Engineering

 See also [P], [ChatGPT Model], [DALL-E Model]


# Prompt Injection

 More at:
  * example of PI - [https://twitter.com/goodside/status/1598253337400717313](https://twitter.com/goodside/status/1598253337400717313)

 See also [P], [ChatGPT Model], [GPT Model]


# Proximal Policy Optimization

# PPO

 We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

 {% pdf "{{site.assets}}/p/proximal_policy_optimization_paper.pdf" %}

 More at:
  * official page - [https://openai.com/blog/openai-baselines-ppo/](https://openai.com/blog/openai-baselines-ppo/)
  * code - [https://github.com/openai/baselines](https://github.com/openai/baselines)
  * paper - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
  * [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)

 See also [P], [Policy Gradient Solution], [Reinforcement Learning]


# PyBullet

 {% youtube "https://www.youtube.com/watch?v=hmV4v_EnB0E" %}

 More at :
  * www - [https://pybullet.org/wordpress/](https://pybullet.org/wordpress/)
  * video paper - [https://xbpeng.github.io/projects/ASE/index.html](https://xbpeng.github.io/projects/ASE/index.html)

 See also [P], [Isaac Gym], [OpenAI Gym], [RobotSchool]


# PyTorch ML Framework

 See also [P], [Deep Learning Framework], [Machine Learning Framework]
