---
title: M
permalink: /m/

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


# M3GAN Movie

 M3GAN (pronounced "Megan") is a 2022 American science fiction horror film directed by Gerard Johnstone, written by Akela Cooper from a story by Cooper and James Wan (who also produced with Jason Blum), and starring Allison Williams and Violet McGraw, with Amie Donald physically portraying M3GAN and Jenna Davis voicing the character. Its plot follows the eponymous artificially intelligent doll who develops self-awareness and becomes hostile toward anyone who comes between her and her human companion.

 {% youtube "https://www.youtube.com/watch?v=BRb4U99OU80" %}

 More at:
  * [https://en.wikipedia.org/wiki/M3GAN](https://en.wikipedia.org/wiki/M3GAN)

 See also [M], [AI Movie]


# Machine Learning

 Part of AI, AI with first a learning phase! A subset of AI. `Field of study that gives computers the ability to learn without being explicitly programmed` 1 or more layers of data, includes but not limited to neural networks. Unsupervised, supervised (classification + regression) , reinforcement. `Data --> Model --> Prediction`.

 ![]( {{site.assets}}/m/machine_learning.png ){: width="100%"}

 ```
A company is using a rule-based system to classify the credit card transactions as fraud or not fraud. Do you think this a machine learning solution?
No! (but why? because it is a rule-based system?)

A company receives thousands of calls everyday to route to the right agent. The routing requires several hops and given the number of calls is expensive.
What is a machine learning solution?
Predict what are the required agent skills based some input parameters
==> That's a multiclass classification problem!
 ```

 See also [M], [Artificial Intelligence], [Deep Learning], [Machine Learning Algorithm], [Machine Learning Pipeline], [Neural Network], [Prediction]


# Machine Learning Framework

  * pytorch
  * tensorflow
  * JAX
 Watch for:
  * eager mode (execute like a python script, from top to bottom)
  * graph format and execution engine natively has no need for Python, and TensorFlow Lite and TensorFlow Serving address mobile and serving considerations respectively.

 More at :
  * [https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/](https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/)

 See also [M], [JAX ML Framework], [PyTorch ML Framework], [TensorFlow ML Framework]


# Machine Learning Pipeline

 1. ML problem framing (aka discovery phase)
 1. data collection and integration
  * :warning: data-set can be incomplete and therefore misleading
  * :warning: data in data-set can be irrelevant
  * :warning: data-set may be too small (needs to be at least 10 times the number of features!)
  * :warning: needs to be cleaned?
 1. data preparation
  * dig into a small to manually critically explore the data.
  * confirm that all label are relevant to solve the ML problem
  * What features are there?
  * Does it match my expectation?
  * Is there enough information to make accurate prediction?
  * Is data missing?
  * Should any label be excluded?
  * Can some label be combined because of overlap?
  * Are existing labels accurate or not?
  * Reviewing questions from discovery phase and continue to interact with domain expert
 1. data visualization and analysis
  * understand the relationships in a dataset
  * find outliers (use histograms)
  * find groups
  * use imputation to complete data
  * pie charts, histogram, scatter plots for correlation
  * try to extract noise from data, noise causes overfitting and reduce accuracy of predictions
 1. feature selection and engineering
  * you want a minimum correlation between the features, but the maximum correlation between the feature and the output
  * do the feature I use make sense?
  * very time consuming step!
  * ex: what was the last item purchased by the customer?
  * ex: when was the last purchase of the customer?
  * ex: do the customer owns a kindle? (then do not expect questions related to kindle)
 1. model training
 1. model evaluation
  * confusion matrix
 1. prediction

 See also [M], [Dataset], [Discovery Phase], [Feature Engineering], [Machine Learning], [Machine Learning Algorithm], [Overfitting]


# Machine Learning Type

  * unsupervised,
  * Supervised (Classification, regression)
  * self-supervised learning
  * reinforcement.
  * transfer learning
 
 ![]( {{site.assets}}/m/machine_learning_type_1.png ){: width="100%"}
 
 ![]( {{site.assets}}/m/machine_learning_type_2.png ){: width="100%"}
 
 See also [M], [Reinforcement Learning], [Self-Supervised Learning], [Supervised Learning], [Transfer Learning], [Unsupervised Learning]


# Machine Reasoning

 Query a knowledge graph = traverse a knowledge graph with queries. Query types are:
  * one-hop queries
  * path queries
  * conjunctive queries

 ![]( {{site.assets}}/m/machine_reasoning.png ){: width="100%"}

 See also [M], [Deductive Reasoning], [Inductive Reasoning], [Knowledge Graph], [Question Answering Graph Neural Network], [Reasoning]


# Machine Translation

 Google Translate, DeepL, and other machine translation programs use NLP to evaluate millions of sentences translated by human speakers of different language pairs.

 See also [M], [Natural Language Processing]


# Majority Vote Algorithm

 When you crowdsource a labeling task, how can you be certain that the label is correct? Have several people label the same image/entry and apply this algorithm! An alternative is to use Dawid-Skene algorithm.

 See also [M], [Dawid-Skene Algorithm], [Labeling Service]


# Markov Decision Process

 Markov Decision Process is a Reinforcement Learning algorithm that gives us a way to formalize sequential decision making. This formalization is the basis to the problems that are solved by Reinforcement Learning. The components involved in a Markov Decision Process (MDP) is a decision maker called an agent that interacts with the environment it is placed in. These interactions occur sequentially overtime. In each timestamp, the agent will get some representation of the environment state. Given this representation, the agent selects an action to make. The environment is then transitioned into some new state and the agent is given a reward as a consequence of its previous action. The process of selecting an action from a given state, transitioning to a new state and receiving a reward happens sequentially over and over again. This creates something called a trajectory that shows the sequence of states, actions and rewards. Throughout the process, it is the responsibility of the reinforcement learning agent to maximize the total amount of rewards that it received from taking actions in given states of environments. `The agent not only wants to maximize the immediate rewards but the cumulative reward it receives in the whole process.`

 More at :
  * [https://en.wikipedia.org/wiki/Markov_decision_process](https://en.wikipedia.org/wiki/Markov_decision_process)

 See also [M], [Reinforcement Learning], [State Model]


# Markov Random Field

 See also [M], [Generative Classifier]


# Masked Language Modeling

# MLM

 Methods: replace 15% of words in corpus with special [MASK] token and ask the NLP model (e.g. BERT) to fill in the blank

 ```
Istanbul is a great [MASK] to visit          # Must guess the word, i.e. city
 ```

 :warning: Words are broken into tokens, so whole word masking implies that related tokens need to be masked together
 
 ```
Input Text             : the man jumped up , put his basket on phil ##am ##mon ' s head
Original Masked Input  : [MASK] man [MASK] up , put his [MASK] on phil [MASK] ##mon ' s head
Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK] [MASK] ' s head

The training is identical -- we still predict each masked WordPiece token independently. The improvement comes from the fact that the original prediction task was too 'easy' for words that had been split into multiple WordPieces.
 ```

 See also [M], [BERT Model], [Image Inpainting], [Self-Supervised Learning]


# Masked Self-Attention

 Attention can only be done to words to the left of the current word. The computation of the attention score is the same for self-attention but for later words, attention score is 0. Used in Decoders of transformers (not encoder!), used by GPT models.

 ![]( {{site.assets}}/m/masked_self_attention.png ){: width="100%"}

 See also [M], [Attention Score], [Decoder], [GPT Model], [Self-Attention]


# Masked Vision Modeling

# MVM

 See also [M], [Masked Language Modeling], [Vision-Language Pre-Training]


# Max Pooling Layer

 Downsample the feature map (take a thumbnail/smaller-size of the image with some feature still present). while a convoluted layer extract features. In the illustration below, we take a 2x2 kernel and pool the maximum value. Benefits with this approach are
  * discovered features in the previous convoluted Layer are preserved as we are keeping the max!
  * image is smaller <== A stack of images becomes a smaller stack of images
  * because of down sampling, the position of the exact match is not as important/sensitive


 ![]( {{site.assets}}/m/max_pooling_layer.png ){: width="100%"}

 See also [M], [Convoluted Layer], [Convoluted Neural Network], [Fully Connected Layer]


# Mean Absolute Error

# MAE

 See also [M], [Prediction Error]


# Mean Absolute Error Loss Function

# MAE Loss Function

 The Mean Absolute Error (MAE) is only slightly different in definition from the MSE, but interestingly provides almost exactly opposite properties! To calculate the MAE, you take the difference between your model’s predictions and the ground truth, apply the absolute value to that difference, and then average it out across the whole dataset. The MAE, like the Mean Square Error (MSE), will never be negative since in this case we are always taking the absolute value of the errors. The MAE is formally defined by the following equation:

 ![]( {{site.assets}}/m/mean_absolute_error_loss_function_formula.png ){: width="100%"}
 ![]( {{site.assets}}/m/mean_absolute_error_loss_function_graph.png ){: width="100%"}

 ```
# MAE loss function
def mae_loss(y_pred, y_true):
    abs_error = np.abs(y_pred - y_true)
    sum_abs_error = np.sum(abs_error)
    loss = sum_abs_error / y_true.size
    return loss
 ```

 Pros and Cons:
  * Advantage: The beauty of the MAE is that its advantage directly covers the MSE disadvantage. Since we are taking the absolute value, all of the errors will be weighted on the same linear scale. Thus, unlike the MSE, we won’t be putting too much weight on our outliers and our loss function provides a generic and even measure of how well our model is performing.
  * Disadvantage: If we do in fact care about the outlier predictions of our model, then the MAE won’t be as effective. The large errors coming from the outliers end up being weighted the exact same as lower errors. This might results in our model being great most of the time, but making a few very poor predictions every so-often.

 See also [M], [Huber Loss Function], [Loss Function], [Mean Square Error Loss Function]


# Mean Absolute Percentage Error

# MAPE

 See also [M], [Prediction Error]


# Mean Square Error Loss Function

# MSE Loss Function

 MSE loss function is widely used in linear regression as the performance measure. To calculate MSE, you take the difference between your predictions and the ground truth, square it, and average it out across the whole dataset.

 ![]( {{site.assets}}/m/mean_square_error_loss_function_formula.png ){: width="100%"}

 ![]( {{site.assets}}/m/mean_square_error_loss_function_graph.png ){: width="100%"}

 where y(i) is the actual expected output and ŷ(i) is the model’s prediction.

 ```
def mse_loss(y_pred, y_true):
    squared_error = (y_pred - y_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / y_true.size
    return loss
 ```


 Pros and cons:
  * Advantage: The MSE is great for ensuring that our trained model has no outlier predictions with huge errors, since the MSE puts larger weight on theses errors due to the squaring part of the function.
  * Disadvantage: If our model makes a single very bad prediction, the squaring part of the function magnifies the error. Yet in many practical cases we don’t care much about these outliers and are aiming for more of a well-rounded model that performs good enough on the majority.

 See also [M], [Linear Regression], [Loss Function], [Regression Tree]


# Mechanical Turk

 To label the data!

 More at:
  * figure-eight company [https://www.figure-eight.com/](https://www.figure-eight.com/)

 See also [M], [Labeling Service]


# Megatron Model

 More at :
  * [https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)

 See also [M], [Nvidia Company]


# Membership Inference Attack

 A type of attack called “membership inference” makes it possible to detect the data used to train a machine learning model. In many cases, the attackers can stage membership inference attacks without having access to the machine learning model’s parameters and just by observing its output. Membership inference can cause security and privacy concerns in cases where the target model has been trained on sensitive information.

 {% youtube "https://www.youtube.com/watch?v=rDm1n2gceJY" %}

 Beware:
  * This membership inference is due to overfitting!
  * machine learning models tend to perform better on their training data ==> attack
 Possible solutions:
  * differential privacy

 More at :
  * paper - [https://arxiv.org/abs/2112.03570](https://arxiv.org/abs/2112.03570)

 See also [M], [Differential Privacy], [Overfitting]


# Meta Company

 See [Facebook Company]


# Meta Learning

 Normally you take the x to predict the y and optimize parameters to get as close to y as possible. Here you take the x and y to generate a theta parameter to fit in another model... then use the loss of the aggregate model to

 ![]( {{site.assets}}/m/meta_learning.png ){: width="100%"}

 See also [M], [Meta Model], [Model Agnostic Meta Learning], [Reinforcement Learning], [Transfer Learning]


# Meta Model

 A ML model to find the best hyperparameters. Ex: Gaussian process regression models object metric as a function of hyperparameters (beware assume smoothness, works with low data, confidence estimates) + bayesian optimization decides where to search next (explore and exploit and gradient free) .

 See also [M], [Hyperparameter Optimization], [Meta Learning]


# Metaverse

 Think of it as the internet brought to life, or at least rendered in 3D. Zuckerberg has described it as a "virtual environment" you can go inside of — instead of just looking at on a screen. Essentially, it's a world of endless, interconnected virtual communities where people can meet, work and play, using virtual reality headsets, augmented reality glasses, smartphone apps or other devices.

 More at:
  * [https://www.npr.org/2021/10/28/1050280500/what-metaverse-is-and-how-it-will-work](https://www.npr.org/2021/10/28/1050280500/what-metaverse-is-and-how-it-will-work)

 See also [M], [The Matrix Movie], [Virtual Reality]


# METEOR Score

 See also [M], [MS COCO Caption Dataset]


# Metropolis Movie

 Metropolis is a 1927 German expressionist science-fiction film directed by Fritz Lang.

 {% youtube "https://www.youtube.com/watch?v=rXpKwH2oJkM" %}

 {% youtube  "https://www.youtube.com/watch?v=PgBitJRkANw" %}

 More at:
  * [https://en.wikipedia.org/wiki/Metropolis_(1927_film)](https://en.wikipedia.org/wiki/Metropolis_(1927_film))

 See also [M], [AI Movie]


# Microsoft Common Object In Context Caption Dataset

# Microsoft COCO Caption Dataset

# MSFT COCO Caption Dataset

 A dataset consists of 1.5M captions for 330,000 images. The captions are generated by human annotators. Each image is linked to 5 captions.

 More at :
  * [https://vision.cornell.edu/se3/wp-content/uploads/2015/04/1504.00325v2.pdf](https://vision.cornell.edu/se3/wp-content/uploads/2015/04/1504.00325v2.pdf)

 See also [M], [BLEU Score], [CIDEr Score], [Dataset], [METEOR Score], [ROUGE Score]


# Microsoft Company

# MSFT Company

 See also [M], [Company], [DeepSpeed Project], [OpenAI Company]


# Milvus Database

 ![]( {{site.assets}}/m/milvus_database.png ){: width="100%"}

 See also [M], [Vector Database]


# MindPong Game

 {% youtube "https://www.youtube.com/watch?v=rsCul1sp4hQ" %}

 More at:
  * ...

 See also [M], [Neuralink Company]


# Minerva Model

 Can AI change mathematics?

 More at:
  * [https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html)
  * [https://www.nature.com/articles/d41586-023-00487-2](https://www.nature.com/articles/d41586-023-00487-2)

 See also [M], [Google Company]


# Mini-Batch

 Take your training samples. Randomized the samples. Take the first 100 of them (the mini-batch).

 See also [M], [Mini-Batch Gradient Descent]


# Mini-Batch Gradient Descent

 ~ stochastic gradient descent with more than 1 sample that use matrix optimization in computation. A compromise between computing the true gradient (all samples) and the gradient at a single example (aka stochastic gradient descent), is to compute the gradient against more than one training example (called a "mini-batch") at each step. This can perform significantly better than true stochastic gradient descent because the code can make use of vectorization libraries rather than computing each step separately. It may also result in smoother convergence, as the gradient computed at each step uses more training examples.

 ![]( {{site.assets}}/m/mini_batch_gradient_descent.png ){: width="100%"}

 So, after creating the mini-batches of fixed size, we do the following steps in one epoch:
  * Pick a mini-batch
  * Feed it to Neural Network
  * Calculate the mean gradient of the mini-batch
  * Use the mean gradient we calculated in step 3 to update the weights
  * Repeat steps 1–4 for the mini-batches we created

 Just like SGD, the average cost over the epochs in mini-batch gradient descent fluctuates because we are averaging a small number of examples at a time. But the mini-batch gradient descent is a good (the best) approximation. Also note that the path is less optimized than the "complete" gradient descent where you take all sample into consideration for one step, but it is a good approximation and the destination/convergence is the same = the local minima is not changed by the path taken to reach it.

 See also [M], [Batch Gradient Descent], [Gradient Descent], [Mini-Batch], [Stochastic Gradient Descent]


# Mixed Reality

# MR

 Mixed reality is a blend of physical and digital worlds, unlocking natural and intuitive 3D human, computer, and environmental interactions. This new reality is based on advancements in computer vision, graphical processing, display technologies, input systems, and cloud computing.

 Mixed reality is the midpoint in the [Virtual Continuum]

 {% youtube "https://www.youtube.com/watch?v=_xpI0JosYUk" %}

 More at:
  * [https://learn.microsoft.com/en-us/windows/mixed-reality/discover/mixed-reality](https://learn.microsoft.com/en-us/windows/mixed-reality/discover/mixed-reality)
  * [https://en.wikipedia.org/wiki/Mixed_reality](https://en.wikipedia.org/wiki/Mixed_reality)

 See also [M], ...


# Mixture Of Local Experts

# MoE

 In 1991, MoE was first introduced by a research group that included deep-learning and Switch Transformer creator Geoff Hinton. In 2017, the Google Brain team and Hinton used MoE to create an NLP model based on recurrent neural networks (RNN) of 137 billion parameters, where it achieved state-of-the-art (SOTA) results on language modelling and machine translation benchmarks.

 More at :
  * paper - [http://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf](http://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)

 See also [M], [Switch Transformer]


# ML Algorithm Evaluation

 To evaluate any technique we generally look at 3 important aspects: (1) Ease to interpret output (2) Calculation time (3) Predictive Power.
 
 ![]( {{site.assets}}/m/ml_algorithm_evaluation.png ){: width="100%"}
 
 Let us take a few examples to  place KNN in the scale...

 See also [M], [K-Nearest Neighbor], [Random Forest]


# ML Flow
 
  More at:
   * [https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145](https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145)

  See also [M], ...


# MLOps

 Machine Learning Operations

 ![]( {{site.assets}}/m/ml_mlops.png ){: width="100%"}

 See also [M], [DevOps]


# MNIST Dataset

 ![]( {{site.assets}}/m/mnist_dataset.png ){: width="100%"}

 See also [M], [Dataset]


# Model

 See also [M], [Chained Model]


# Model Agnostic Meta Learning

# MAML

 How to pick a better initialization point? Don't start randomly, but use a particular function phy... Gradient descent for initial point = 2 step of normal training.
 
 ![]( {{site.assets}}/m/maml_reptile.png ){: width="100%"}
 
 See also [M], [Meta Learning], [Reptile]


# Model Benchmark

 See also [M], [NLP Benchmark]


# Model Card

 {% pdf "{{site.assets}}/m/model_card_paper.pdf" %}

 More at:
  * model cards at google - [https://modelcards.withgoogle.com/about](https://modelcards.withgoogle.com/about)

 See also [M], [Model Data Sheet]


# Model Drift

 Model performance changes over time.

 See also [M], [Confusion Matrix]


# Model Data Sheet

 More at:
  * [https://www.microsoft.com/en-us/research/project/datasheets-for-datasets/](https://www.microsoft.com/en-us/research/project/datasheets-for-datasets/)

 See also [M], [Model Card]


# Model Tuning

 done with gradient descent?


# Model Type

 ![]( {{site.assets}}/m/model_type.png ){: width="100%"}

 More at :
  * [https://youtu.be/J8Eh7RqggsU?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX&t=2568](https://youtu.be/J8Eh7RqggsU?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX&t=2568)

 See also [M], [Reflex Model], [Logic Model], [State Model], [Variable Model]


# Model-Based Reinforcement Learning

 In [Reinforcement Learning] ....

  {% youtube "https://www.youtube.com/watch?v=vfpZu0R1s1Y" %}

 See also [M], ...


# Multiclass Classification

 ML for classification in more than 2 categories.

 See also [M], [Binary Classification], [Classification]


# Multi-Head Attention

 ~ the brains of the Transformer and responsible for performance through parallelism. Multi-Head Attention consists of several attention layers running in parallel. The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value (aka Q,K,V). All three parameters are similar in structure, with each word in the sequence represented by a vector. In transformers is used for encoder and decoder.

 ![]( {{site.assets}}/m/multi_head_attention.png ){: width="100%"}

 See also [M], [Attention Score], [Attention-Based Model], [Decoder], [Encoder], [Masked Self-Attention], [Self-Attention], [Transformer Model]


# Multi-Task Learning

# MTL

 ~ `not enough data for individual algo --> reuse data from other similar algo` Train all parameters at the same time, but each of the task have shared parameters. One example is a spam-filter, which can be treated as distinct but related classification tasks across different users. To make this more concrete, consider that different people have different distributions of features which distinguish spam emails from legitimate ones, for example an English speaker may find that all emails in Russian are spam, not so for Russian speakers. Yet there is a definite commonality in this classification task across users, for example one common feature might be text related to money transfer. Solving each user's spam classification problem jointly via MTL can let the solutions inform each other and improve performance.

 ![]( {{site.assets}}/m/multi_task_learning.png ){: width="100%"}

 Questions:
  * How does that relate to transfer learning?

 See also [M], [Gato Model], [Insufficient Data Algorithm], [Transfer Learning]


# Multiattribute Objective

 Example: A self-driving car needs to
  * take you from point A to point B
  * in a safe manner
  * in a comfortable manner
  * but as quickly as possible
  * without killing a pedestrian or anybody
  * do not trigger road rage of other driver

 See also [M], [Reinforcement Learning]


# Multilayer Perceptron

 A fully connected multi-layer neural network is called a Multilayer Perceptron.

 ![]( {{site.assets}}/m/multilayer_perceptron.png ){: width="100%"}

 ```
import torch
import torch.nn as nn

# Helper class for MLP
class MLP(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
 ```

 See also [M], [Neural Network], [Vision Transformer]


# Multimodal Alignment

 ![]( {{site.assets}}/m/multimodal_alignment.png ){: width="100%"}

 See also [M], [Multimodal Learning]


# Multimodal Co-learning

 Enhance the classification of one modality with the help of another. For example: help to recognize a dog.

 ![]( {{site.assets}}/m/multimodal_colearning.png ){: width="100%"}

 See also [M], [Multimodal Learning]


# Multimodal Distribution

 In math, use a distribution that has multiple maxima = multiple modes, i.e. distinct "peak (local maxima) in the probability density function".

 ![]( {{site.assets}}/m/multimodal_distribution.png ){: width="100%"}

 See also [M], [Multimodal Learning]


# Multimodal Framework

# MMF

 A rfameowkr build on top of Pytorch for language and vision models. Features include (1) modular, (2) distributed, (3) pre-trained multimodal transformer.


# Multimodal Fusion

 Bring the representation of each mode into the same embedding space. Methods:
  * dot product (multiply the vectors of one modality with the other) :warning: BRing everything into a scalar value?
  * concat output + fully connected layers after the concatenation
  * tensor fusion network (paper published in 2017) - https://sentic.net/tensor-fusion-network-for-multimodal-sentiment-analysis.pdf
  * minimize the cosine loss between the 2 modalities

 See also [M], [Multimodal Learning]


# Multimodal Learning

 For a human, multimodal = using all our senses. For a computer, using input text, input image, etc. generating output text based on image, output image based on input text? Examples: (1) a robot taking instruction (text + voice recognition) and performing various actions in the physical space (2) visual common sense reasoning - answering common sense questions about a given image (3) an email network graph containing nodes as users, IP address connected by email sent and received edges. Not multimodal learning examples: (1) spoke language translation.

 ![]( {{site.assets}}/m/multimodal_learning.png ){: width="100%"}

 ```
AI that can understand the relationships between images, text and more
 ```

 * Google's Pathways architecture is trying to solve this.

 See also [M], [Embedding Space], [Multimodal Distribution], [Multimodal Space], [Pathways Model Architecture]


# Multimodal Space

 A latent/embedded space where both modality have the same representation.

 See also [M], [Modal Learning]


# Multimodal Translation

 How to translate one mode to the other. Ex: captioning an image.

 See also [M], [BLIP Model]


# Multiple Linear Regression

 A linear regression with multiple input / independent variable.
 ```
Y = a + b.X1 + c.X2
 ```

 More at:
  * sample paper - [https://www.mdpi.com/2073-4395/11/5/885](https://www.mdpi.com/2073-4395/11/5/885)

 See also [M], [Linear Regression]


# MUM Model

 More at:
  * [https://blog.google/products/search/introducing-mum/](https://blog.google/products/search/introducing-mum/)

 See also [M], [Google Company]


# Muse Model

 ??? Does not exist? ???

 See also [M], [Google Company]


# MuseGAN

 4 inputs: (1) Chords input (with temporal network) (2) style input (3) melody input (temporal network) (4) groove input


# MusicLM Model

 An impressive new AI system from Google can generate music in any genre given a text description. Called MusicLM, Google’s certainly isn’t the first generative artificial intelligence system for song. There have been other attempts, including Riffusion, an AI that composes music by visualizing it, as well as Dance Diffusion, Google’s own AudioML and OpenAI’s Jukebox. But owing to technical limitations and limited training data, none have been able to produce songs particularly complex in composition or high-fidelity. MusicLM is perhaps the first that can.

 {% pdf "{{site.assets}}/m/musiclm_model_paper.pdf" %}

 More at:
  * twitter - [https://twitter.com/keunwoochoi/status/1618809167573286912](https://twitter.com/keunwoochoi/status/1618809167573286912)
  * paper - [https://arxiv.org/abs/2301.11325](https://arxiv.org/abs/2301.11325)
  * example - [https://google-research.github.io/seanet/musiclm/examples/](https://google-research.github.io/seanet/musiclm/examples/)
  * dataset - [https://www.kaggle.com/datasets/googleai/musiccaps](https://www.kaggle.com/datasets/googleai/musiccaps)
  * techcrunch article - [https://techcrunch.com/2023/01/27/google-created-an-ai-that-can-generate-music-from-text-descriptions-but-wont-release-it/](https://techcrunch.com/2023/01/27/google-created-an-ai-that-can-generate-music-from-text-descriptions-but-wont-release-it/)

 See also [M], [Google Company], [Jukebox Model], [Riffusion Model]


# MuZero Model

 ![]( {{site.assets}}/m/muzero_model.jpeg ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=vt5jOSy7cz8" %}

 {% youtube "https://www.youtube.com/watch?v=L0A86LmH7Yw" %}

 {% youtube "https://www.youtube.com/watch?v=pgZhGavMHcU" %}

 More at:
  * nature article - [https://www.nature.com/articles/s41586-020-03051-4.epdf?sharing_token=kTk-xTZpQOF8Ym8nTQK6EdRgN0jAjWel9jnR3ZoTv0PMSWGj38iNIyNOw_ooNp2BvzZ4nIcedo7GEXD7UmLqb0M_V_fop31mMY9VBBLNmGbm0K9jETKkZnJ9SgJ8Rwhp3ySvLuTcUr888puIYbngQ0fiMf45ZGDAQ7fUI66-u7Y%3D|https://www.nature.com/articles/s41586-020-03051-4.epdf|](https://www.nature.com/articles/s41586-020-03051-4.epdf?sharing_token=kTk-xTZpQOF8Ym8nTQK6EdRgN0jAjWel9jnR3ZoTv0PMSWGj38iNIyNOw_ooNp2BvzZ4nIcedo7GEXD7UmLqb0M_V_fop31mMY9VBBLNmGbm0K9jETKkZnJ9SgJ8Rwhp3ySvLuTcUr888puIYbngQ0fiMf45ZGDAQ7fUI66-u7Y%3D|https://www.nature.com/articles/s41586-020-03051-4.epdf|)

 See also [M], [DeepMind Company]


# MXNET

 See also [M], [Deep Learning Framework]
