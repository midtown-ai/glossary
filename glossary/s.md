---
title: S
permalink: /s/

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


# SageMaker

 See also [S}, ...


# SageMaker Ground Truth

 Used for labelling the data by machines, internal employees, mechanical turk, or 3rd party partner.


# SageMaker Neo

 ~ compiler of ML models before they are distributed to the endpoint. Compatible with TensorFlow, XGBoost, MxNET, PyTorch, ... Allow the model to run without any frameworr, this reduce the memory footprint on the device (at the edge) by 100x, while improving the performance by x2.


# Sample

 See also [S], [Distribution]


# Sample Efficiency

 See also [S], [Learning Velocity]


# Sampling Error

 Sampling errors occur when a sample of data is used to estimate a population parameter, such as the mean or proportion. These errors occur because the sample may not be a perfect representation of the population, and there will be some degree of difference between the sample statistic and the true population parameter. There are two types of sampling errors:
  * Random sampling error: Random sampling error occurs due to chance. It occurs because each sample of data will have some degree of variation from the population. The larger the sample, the smaller the random sampling error is likely to be.
  * Systematic sampling error: Systematic sampling error occurs due to bias in the sampling process. Bias can be introduced if the sample is not selected at random, or if the sample is not representative of the population.
 Both of these types of sampling errors can be reduced by using a larger sample size or by using a more representative sample. However, it is important to note that it is never possible to completely eliminate sampling error.

 More at:
  * ...

 See also [S], [Resampling Method], [Sampling]


# Sampling Method

  * Bayesian Optimizatin Sampling Method

 See also [S], [Bayesian Optimization Sampling Method]


# Scaler

 Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance). For instance many elements used in the objective function of a learning algorithm (such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models) assume that all features are centered around 0 and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected. THe standard scaler does the following:
  * Standardize features by removing the mean and scaling to unit variance.
  * The standard score of a sample x is calculated as:
 
 ```
z = (x - u) / s

u == mean of discrete random variable
u = sum(x) / number_of_samples

v == variance of discrete random variable in training sample
v = 1 / number_of_sample * ( sum_of_each_sample ( sample_value - mean) ^ 2) )

s == standard deviation of discrete random variable in random sample
s = sqrt(v)
 ```

 Example in code
 
 ```
>>> from sklearn.preprocessing import StandardScaler
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler()
>>> print(scaler.mean_)                       # (0 + 0 + 1 + 1 ) / 4 = 0.5
[0.5 0.5]
                                              # the variance is the average of each point from the mean.
>>> print(scaler.var_)                        # [ (0-0.5)^2 + (0-0.5)^2 + (1-0.5)^2 + (1-0.5)^2 ] / 4 =  (0.5)^2 = 0.25
[0.25, 0.25]
>>> print(scaler.transform(data))
[[-1. -1.]                                    # 0 --> (0 - 0.5) / sqrt(0.25) = -1
 [-1. -1.]
 [ 1.  1.]                                    # 1 --> (1 - 0.5) / sqrt(0.25) = 1
 [ 1.  1.]
>>> print(scaler.transform([[2, 2]]))
[[3. 3.]]                                     # 2 --> (2 - 0.5) / sqrt(0.25) = 3
 ```

 See also [S], [Estimator]


# Search Problem

 See also [S], [State Model]


# Scene Graph

 Do entity extract + add relationship. Can be built based from text or images.

 See also [S], [Entity Extraction], [Graph Neural Network], [Knowledge Graph], [Relation Extraction]


# Scikit Learn

 Since its release in 2007, scikit-learn has become one of the most popular machine learning libraries. scikit-learn provides algorithms for machine learning tasks including classification, regression, dimensionality reduction, and clustering. It also provides modules for pre-processing data, extracting features, optimizing hyperparameters, and evaluating models. scikit-learn is built on the popular Python libraries NumPy and SciPy. NumPy extends Python to support efficient operations on large arrays and multi-dimensional matrices. SciPy provides modules for scientific computing. The visualization library matplotlib is often used in conjunction with scikit-learn.

 See also [S], [Dataset]


# Self-Attention

 Use itself as input to
   * find to what pronouns are referring to, what noun is being modified by adjective, subject of verbs, etc
   * (implicit) grammar rules  --- when you learn to speak, you do not have to learn grammar rules first to speak!

 More at:
  * [https://towardsdatascience.com/self-attention-5b95ea164f61](https://towardsdatascience.com/self-attention-5b95ea164f61)

 See also [S], [Attention], [Attention-Based Model], [Masked Self-Attention], [Multi-Head Attention]


# Self-Organizing Map

 See also [S], [Unsupervised Deep Learning Model], [Unsupervised Learning]


# Self-Supervised Learning

 `~ automatic generation of labels from the unlabelled input data.` A type of learning that does not need labels to be applied by humans. The training dataset is generated from the data. Examples:
   * Next word prediction : "she planned to visit the"
   * Masked Language Modeling : "Alice chased the ____ rabbit and fell down the ____ into a new world."
   * Replaced word detection : "The cat pounced on the carrot"
   * Paraphrased sentences : "He needs to throw a lot of things -- he has a lot of stuff to get rid off."
 ![]( {{site.assets}}/s/self_supervised_learning.png ){: width="100%}

 See also [S], [Data2Vec Model], [Masked Language Modeling], [Next Word Prediction], [Replaced Word Detection], [Semi-Surpervised Learning], [Supervised Learning], [Unsupervised Learning], [Upstream Task]


# Semantic Search

 Semantic search leverages deep neural networks to intelligently search through data. You interact with it every time you search on Google. Semantic search is helpful when you want to search for something based on the context rather than specific keywords.

 See also [S], [Natural Language Processing]


# Semantic Segmentation

 semantic segmentation algorithm provides a fine-grained, pixel-level approach to developing computer vision applications. It tags every pixel in an image with a class label from a predefined set of classes (ex: a person). Tagging is fundamental for understanding scenes, which is critical to an increasing number of computer vision applications, such as self-driving vehicles, medical imaging diagnostics, and robot sensing.
 
 For comparison, the Amazon SageMaker Image Classification Algorithm is a supervised learning algorithm that analyzes only whole images, classifying them into one of multiple output categories. The Object Detection Algorithm is a supervised learning algorithm that detects and classifies all instances of an object in an image. It indicates the location and scale of each object in the image with a rectangular bounding box.
 
 Because the semantic segmentation algorithm classifies every pixel in an image, it also provides information about the shapes of the objects contained in the image. The segmentation output is represented as an RGB or grayscale image, called a segmentation mask. A segmentation mask is an RGB (or grayscale) image with the same shape as the input image.

 ![]( {{site.assets}}/s/semantic_and_instance_segmentation.png ){: width="100%}

 More at:
  * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)

 See also [S], [Convolutional Neural Network], [Instance Segmentation], [U-Net Architecture]


# Semantic Embedding

 ![]( {{site.assets}}/s/semantic_embedding.png ){: width="100%}

 See also [S], [Semantic Space], [Zero-Shot Learning]


# Semantic Space

 semantic space, where the knowledge from seen classes can be transferred to unseen classes.

 See also [S], [Embedding Space], [Latent Space], [Semantic Embedding], [Zero-Shot Learning]


# Semantic Understanding

 machine translation, information extraction, text summarization, question answering.

 See also [S], [NLP Benchmark]


# Semi-Supervised Learning

 ~ `dataset not fully labeled --> use similarity to label other data` Supervised learning and unsupervised learning can be thought of as occupying opposite ends of a spectrum. Some types of problem, called semi-supervised learning problems, make use of both supervised and unsupervised data; these problems are located on the spectrum between supervised and unsupervised learning.  Several algorithm can be used like neighbor infection or nearest labeled neighbor. Example A: In a series of picture, you recognize people, let's say 4 of them. If you tag one image, then the tags can be pushed to all the other images. Example B: Classification problem but only on a small subset you have is CORRECTLY labeled. Can you train your model with the rest of the unlabelled data? 

 See also [S], [K-Nearest Neighbor], [Self-Supervised Learning], [Supervised Learning], [Unlabelled Data Algorithm], [Unsupervised Learning], [Weak-Supervised Learning]


# Sensitivity

 ~ true positive rate. = probability of a positive test given the patient has the disease.  Sensitivity refers to the probability of a positive test, conditioned on truly being positive. Examples:
  * how many sick people were CORRECTLY identified as having the condition.

 See also [S], [Confusion Matrix], [Specificity]


# Sentiment Analysis

 Marketers collect social media posts about specific brands, conversation subjects, and keywords, then use NLP to analyze how users feel about each topic, individually and collectively. This helps the brands with customer research, image evaluation, and social dynamics detection.

 See also [S], [Natural Language Processing], [NLP Benchmark]


# Sequence To Sequence Model

# Seq2Seq Model

 = A supervised learning algorithm where the input is a sequence of tokens (for example, text, audio) and the output generated is another sequence of tokens. Example applications include: machine translation (input a sentence from one language and predict what that sentence would be in another language), text summarization (input a longer string of words and predict a shorter string of words that is a summary), speech-to-text (audio clips converted into output sentences in tokens). Recently, problems in this domain have been successfully modeled with deep neural networks that show a significant performance boost over previous methodologies. Amazon SageMaker seq2seq uses Recurrent Neural Networks (RNNs) and Convolutional Neural Network (CNN) models with attention as encoder-decoder architectures. Seq2Seq models are particularly good at translation, where a sequence of words from one language is transformed into a sequence of different words in another language. `Google Translate started using a Seq2Seq-based model in production in late 2016`

 ![]( {{site.assets}}/s/sequence_to_sequence_model.png ){: width="100%}

 Seq2Seq models consist of two parts: an encoder and a decoder. Imagine the encoder and decoder as human translators who can each speak only two languages, with each having a different mother tongue. For our example, we’ll say the encoder is a native French speaker and the decoder is a native English speaker. The two have a second language in common: let’s say it’s Korean. To translate French into English, the encoder converts the French sentence into Korean (known as context) and passes on the context to the decoder. Since the decoder understands Korean, he or she can now translate from Korean into English. Working together, they can translate the French language to English.

 See also [S], [Convolutional Neural Network], [Seq2Seq Transformer], [Natural Language Processing], [Recurrent Neural Network], [Supervised Learning], [Text-To-Speech Model]


# Sequence To Sequence Transformer

# Seq2Seq Transformer

 When you use the encoder and decoder of a transformer, you can do a sequence to sequence transformer!
 ![]( {{site.assets}}/s/sequence_to_sequence_transformer.png ){: width="100%}

 See also [S], [Decoder], [Encoder], [Encoder-Decoder Model]


# Serialized Flat File

 The model in compact form, delivered to endpoint.


# Shapley Value

 In [Game Theory], ...

 More at:
  * [https://en.wikipedia.org/wiki/Shapley_value](https://en.wikipedia.org/wiki/Shapley_value)

 See also [S], ...


# SHAP Value

 SHAP is another novel approach to [explainability][Explainable AI] developed by Scott Lundberg at [Microsoft][Microsoft Company] and eventually opened sourced.

 SHAP has a strong mathematical foundation based on Shapley values in game theory where each player in the cooperation is rewarded based on how important they are to cooperation.
 
 More at:
  * [https://towardsdatascience.com/what-is-explainable-ai-xai-afc56938d513](https://towardsdatascience.com/what-is-explainable-ai-xai-afc56938d513)

 See also [S], ...


# Short Circuit Movie

 Short Circuit is a 1986 American science fiction comedy film directed by John Badham and written by S. S. Wilson and Brent Maddock. The film's plot centers upon an experimental military robot that is struck by lightning and gains a human-like intelligence, prompting it to escape its facility to learn more about the world.

 {% youtube "https://www.youtube.com/watch?v=savkuEQKz8Q" %}

 More at:
  * https://en.wikipedia.org/wiki/Short_Circuit_(1986_film)

 See also [S], [AI Movie]


# Siamese Network

 A convolutional encoder that looks at similarities. Ex: Compare 2 images of a lion with nothing else.
 
 ![]( {{site.assets}}/s/siamese_network.png ){: width="100%}

 See also [S], [One Short Learning], [Similarity Function]


# Sigmoid Activation Function

 Pros:
  * output is between 0 and 1
  * sigmoid can be used as a switch (in LSTM and GRU cells)
  * solves exploding gradient problem

 Cons:
  * vanishing gradient problem

 See also [S], [Activation Function], [Exploding Gradient Problem], [Vanishing Gradient Problem]


# Sigmoid Function

 Used as an activation function and in logistic regression.

 ![]( {{site.assets}}/s/sigmoid_function.png ){: width="100%}

 See also [S], [Logistic Regression], [Sigmoid Activation Function]


# Similarity Function

 See also [S], [Encoder], [One-Shot Learning], [Siamese Network]


# Siri Assistant

 See also [S], [Apple Company]


# Skip Connection

 A way to alleviate the vanishing gradient problem by having activation skip hidden layers?

 See also [S], [Residual Block], [Residual Network Model]


# Slicing Function

 See also [S], [Snorkel Program]


# Snorkel Program

 ~ `Unlabelled data --> a weak supervision labeling function` Snorkel is a system for programmatically building and managing training datasets without manual labeling. In Snorkel, users can develop large training datasets in hours or days rather than hand-labeling them over weeks or months. Snorkel currently exposes three key programmatic operations: (1) Labeling data, e.g., using heuristic rules or distant supervision techniques (2) Transforming data, e.g., rotating or stretching images to perform data augmentation (3) Slicing data into different critical subsets for monitoring or targeted improvement.
 
 ![]( {{site.assets}}/s/snorkel_program.png ){: width="100%}

 More at:
  * [https://www.snorkel.org/blog/](https://www.snorkel.org/blog/)
  * new snorkel - [https://www.snorkel.org/blog/hello-world-v-0-9](https://www.snorkel.org/blog/hello-world-v-0-9)

 See also [S], [Data Augmentation], [Labeling Function], [Slicing Function], [Transform Function], [Unlabelled Data Algorithm]


# Softmax Function

 The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or negative, the softmax turns it into a small probability, and if an input is large, then it turns it into a large probability, but it will always remain between 0 and 1.
  * Make the resulting probabilities between 0 and 1.
  * Make the sum of the resulting probabilities equal to 1.

 ![]( {{site.assets}}/s/softmax_function.png ){: width="100%}

 See also [S], [Argmax Function], [Feedforward Neural Network]


# Softplus Activation Function

 Pros:
  * ...

 Cons:
  * ...

 See also [S], [Activation Function], [Exploding Gradient Problem], [Vanishing Gradient Problem]


# Software 1.0

 The “classical stack” of Software 1.0 is what we’re all familiar with — it is written in languages such as Python, C++, etc. It consists of explicit instructions to the computer written by a programmer. By writing each line of code, the programmer identifies a specific point in program space with some desirable behavior.

 See also [S], [Software 2.0]


# Software 2.0

 Software 2.0 is written in much more abstract, human unfriendly language, such as the weights of a neural network. No human is involved in writing this code because there are a lot of weights (typical networks might have millions), and coding directly in weights is kind of hard (I tried). Instead, our approach is to specify some goal on the behavior of a desirable program (e.g., “satisfy a dataset of input output pairs of examples”, or “win a game of Go”), write a rough skeleton of the code (i.e. a neural net architecture) that identifies a subset of program space to search, and use the computational resources at our disposal to search this space for a program that works. In the case of neural networks, we restrict the search to a continuous subset of the program space where the search process can be made (somewhat surprisingly) efficient with backpropagation and stochastic gradient descent.

 ![]( {{site.assets}}/s/software_2_0.png ){: width="100%}

 More at:
  * [https://karpathy.medium.com/software-2-0-a64152b37c35](https://karpathy.medium.com/software-2-0-a64152b37c35)

 See also [S], [Software 1.0]


# Sophia Robot

 Sophia is a social humanoid robot developed by the Hong Kong-based company Hanson Robotics. Sophia was activated on February 14, 2016, and made its first public appearance in mid-March 2016 at South by Southwest (SXSW) in Austin, Texas, United States. Sophia is marketed as a "social robot" that can mimic social behavior and induce feelings of love in humans.

 {% youtube "https://www.youtube.com/watch?v=BhU9hOo5Cuc" %}

 More at:
  * [https://en.wikipedia.org/wiki/Sophia_(robot)](https://en.wikipedia.org/wiki/Sophia_(robot)) 

 See also [S], [Hanson Robotics Company], [Robot]

# Sound Analysis

 See also [S], ...


# Spam Detection

 The spam filtering in your email inbox assigns a percentage of the incoming emails to the spam folder, using NLP to evaluate which emails look suspicious.

 See also [S], [Natural Language Processing]


# Space

 Relates to how an object is represented. In pixel space, an image consist of pixels or pixel parameters (e.g RGB + position). In a latent space, images are encoded and represented by a different tensor. Each time an object is represented differently, that transition make be lossy or lossless.

 See also [S], [Latent Space], [Pixel Space]


# Sparrow Model

 Google's answer to chatGPT. To be released in mid 2023.

 More at:
  * [https://medium.com/@tokamalpathak/chatgpt-and-google-sparrow-the-future-of-ai-powered-communication-5febb200f5ab](https://medium.com/@tokamalpathak/chatgpt-and-google-sparrow-the-future-of-ai-powered-communication-5febb200f5ab)

 See also [S], [ChatGPT Model], [Google Company]


# Sparse Activation

 We have many different parts of our brain that are specialized for different tasks, yet we only call upon the relevant pieces for a given situation. There are close to a hundred billion neurons in your brain, but you rely on a small fraction of them to interpret this sentence. AI can work the same way. We can build a single model that is “sparsely” activated, which means only small pathways through the network are called into action as needed. In fact, the model dynamically learns which parts of the network are good at which tasks -- it learns how to route tasks through the most relevant parts of the model. A big benefit to this kind of architecture is that it not only has a larger capacity to learn a variety of tasks, but it’s also faster and much more energy efficient, because we don’t activate the entire network for every task.

 More at:
  * pathways - [https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

 See also [S], [Dense Model], [Pathways Model Architecture]


# Sparse Matrix

 See also [S], [Sparse Vector]


# Sparse Tensor

 See also [S], [Sparse Vector]


# Sparse Vector

 A vector/list that has mostly the 0 element.

 See also [S], [Vector]


# Specificity

 ~ true negative rate. = probability of a negative test given the patient is doing well. Specificity refers to the probability of a negative test, conditioned on truly being negative. Examples:
  * How many healthy people were CORRECTLY identified as not having the condition.

 See also [S], [Confusion Matrix], [Sensitivity]


# Spectrogram Image

 Turn sound into frequencies represented in an image. That image can then be processed by CycleGAN or other!

 See also [S], [CycleGAN]


# SQuAD Benchmark

 A benchmark for NLP models for question answering.

 See also [S], [NLP Benchmark], [Question Answering]


# Stanford Autonomous Helicopter

 {% youtube "https://www.youtube.com/watch?v=0JL04JJjocc" %}

 See also [S], [Apprentice Learning]


# Stanford Natural Language Inference

# SNLI

 Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), is the task of determining the inference relation between two (short, ordered) texts: entailment, contradiction, or neutral.
 
 ```
Text                                                                    Judgments                 Hypothesis

A man inspects the uniform of a figure in some East Asian country.	C C C C C	The man is sleeping
                                                                       contradiction

An older and younger man smiling.                                       N N E N N       Two men are smiling and laughing at the cats playing on the floor.
                                                                       	neutral

A black race car starts up in front of a crowd of people.               C C C C C       A man is driving down a lonely road.
                                                                       contradiction

A soccer game with multiple males playing.                              E E E E E       Some men are playing a sport.
                                                                        entailment

A smiling costumed woman is holding an umbrella.                        N N E C N       A happy woman in a fairy costume holds an umbrella.
                                                                         neutral
 ```

 See also [S], [NLP Benchmark]


# State

 In reinforcement learning, links the agent and the environment. 
 For every state the agent takes a decision to reach its goal. 

 Beware:
  * Sometimes you do not know the state, but can make an observation that is strongly correlated to the state.

 See also [S], [Observation], [Reinforcement Learning]


# State Model

 When you go from one state to the other.

 See also [S], [Adversarial Model], [Hidden Markov Model], [Markov Decision Process], [Model Type], [Search Problem]


# State Of The Art

# SOTA

 The model that scored the highest based on the available benchmarks.

 See also [S], [Model Benchmark]


# Stereo Vision

 See also [S], [Autonomous Vehicle]


# Stochastic Gradient Descent

# SGD

 There are a few downsides of the gradient descent algorithm. We need to take a closer look at the amount of computation we make for each iteration of the algorithm. Say we have 10,000 data points and 10 features. The sum of squared residuals consists of as many terms as there are data points, so 10000 terms in our case. We need to compute the derivative of this function with respect to each of the features, so in effect we will be doing 10000 * 10 = 100,000 computations per iteration. It is common to take 1000 iterations, in effect we have 100,000 * 1000 = 100000000 computations to complete the algorithm. That is pretty much an overhead and hence gradient descent is slow on huge data. Stochastic gradient descent comes to our rescue! “Stochastic”, in plain terms means “random”. SGD randomly picks ONE data point (if more than one = mini-batch or batch Gradient Descent!) from the whole data set at each iteration/step to reduce the computations enormously.

 ![]( {{site.assets}}/s/stochastic_gradient_descent.png ){: width="100%}

 See also [S], [Batch Gradient Descent], [Gradient Descent], [Mini-Batch Gradient Descent]


# Stochastic Node

 Input = mean + variance, i.e a distribution, but output = a sample of that distribution. Different from a deterministic node.

 See also [S], [Deterministic Node], [Variational Autoencoder Reparametrization Trick]


# Streamlit

 Morea t:
  * [https://streamlit.io/gallery](https://streamlit.io/gallery)
  * [https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145](https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145)

 SEe also [S], ...


# Strong Artificial Intelligence

# Strong AI

 Searle identified a philosophical position he calls "strong AI":
   * The appropriately programmed computer with the right inputs and outputs would thereby have a mind in exactly the same sense human beings have minds.[b]
 The definition depends on the distinction between simulating a mind and actually having a mind. Searle writes that "according to Strong AI, the correct simulation really is a mind. According to Weak AI, the correct simulation is a model of the mind."

 More at:
  * [https://en.wikipedia.org/wiki/Chinese_room#Strong_AI](https://en.wikipedia.org/wiki/Chinese_room#Strong_AI)

 See also [S], [Weak AI]


# Style GAN

 Other GANs focused on improving the discriminator in this case we improve the generator. This GAN generates by taking a reference picture.

 ![]( {{site.assets}}/s/style_gan.jpeg ){: width="100%}

 See also [S], [Generative Adversarial Network]


# Subsampling

 See also [S], [Convolutional Neural Network]


# Super Resolution GAN

# SRGAN

 The main purpose of this type of GAN is to make a low resolution picture into a more detailed picture. This is one of the most researched problems in Computer vision. The architecture of the SRGAN is given below

 ![]( {{site.assets}}/s/super_resolution_gan.jpeg ){: width="100%}

 As given in the above figure we observe that the Generator network and Discriminator both make use of Convolutional layers , the Generator make use of the Parametric Rectified Linear Unit (ReLU) as the activation function whereas the Discriminator uses the Leaky-ReLU.

 See also [S], [Generative Adversarial Network], [Rectified Linear Unit Activation Function]


# SuperGLUE Benchmark

 In the last year, new models and methods for pretraining and transfer learning have driven striking performance improvements across a range of language understanding tasks. The GLUE benchmark, introduced one year ago, offered a single-number metric that summarizes progress on a diverse set of such tasks, but performance on the benchmark has recently come close to the level of non-expert humans, suggesting limited headroom for further research.

 We take into account the lessons learnt from original GLUE benchmark and present SuperGLUE, a new benchmark styled after GLUE with a new set of more difficult language understanding tasks, improved resources, and a new public leaderboard.

 {% pdf "{{site.assets}}/s/superglue_paper.pdf %}

 More at:
  * [https://super.gluebenchmark.com/](https://super.gluebenchmark.com/)

 See also [S], [GLUE Benchmark], [NLP Benchmark]


# Supervised Fine-Tuning

# SFT

 `A way to turn a generalist [pre-trained model] into a "fine-tuned" expert model, aka domain-specific model` Normally done with [supervised learning] to minimize the number of samples required and be less compute intensive and be more compute friendly?

 See also [S], [Domain-Specific Model], [Red Teaming], [Transfer Learning]


# Supervised Learning

 `provide labelled training data` (picture of a dog, with label this is a dog!). Ex:
   * Regression (predicting the price that a house will sell for)
   * Classification (Cat or not cat?)
   * Random forest.
 
 Teaches the model by providing a dataset with example inputs and outputs. Human teacher's expertise is used to tell the model which outputs are correct. Input --> Model --> output/prediction. Further grouped into classification and regression. Learn the relationship between the input parameters and the output. For example: route calls to the correct agent-skills (hence recording of calls to be reviewed by supervisor/teacher). REQUIREMENTS: model should already be functioning and easy to observe! If that is not the case, maybe look at unsupervised learning! 

 See also [S], [Classification], [Mechanical Turk], [Random Forest], [Regression], [Reinforcement Learning], [Self-Supervised Learning], [Semi-Supervised Learning], [Supervisor], [Unsupervised Learning]


# Supervisor

 A teacher! Beware that the teach is not necessarily a 'person', it can also be a machine, the environment, etc. For example, historical data to predict future sales. For example, predict future earthquakes, the teacher is nature herself!

 See also [S], [Supervised Learning]


# Support Vector

 Name of the type of points that are closed to the final SVM boundary and usd for the computation of the SMV boundary!

 See also [S], [Support Vector Machine]


# Support Vector Machine

# SVM

 `Find a linear/planar/x-D/hyperplane to use as a decision boundary in real, transformed, or latent space!` To find the boundary, only use the distance from the points/samples/support-vectors to the boundary and make sure it is as large/wide as possible to maximize the chance of success of the classification (i.e. minimize false positive). Mostly used for classification, but occasionally for regression.

 ![]( {{site.assets}}/s/support_vector_machine.png ){: width="100%}

  {% youtube "https://www.youtube.com/watch?v=FB5EdxAGxQg" %}

  {% youtube "https://www.youtube.com/watch?v=T9UcK-TxQGw" %}

 Beware:
  * SVM can be used for binary classification (but also with a trick can be used for multi-class classification)

 See [Classification], [Decision Boundary], [Kernel Trick], [Hyperplane], [Regression], [Support Vector]


# Surrogate Model

 Can be done in second or minute. X: hyperparameter configuration, Y= model quality, no gradient.

 See also [S], [HPO]


# Switch Transformer

 Based on the T5 model uses sparse activation +

 More at:
  * [https://analyticsindiamag.com/a-deep-dive-into-switch-transformer-architecture/](https://analyticsindiamag.com/a-deep-dive-into-switch-transformer-architecture/)
  * paper - [https://arxiv.org/pdf/2101.03961.pdf](https://arxiv.org/pdf/2101.03961.pdf)

 See also [S], [Google Company], [Mixture Of Local Expect], [Sparse Activation], [T5 Model]


# Synapse

 A synapse is the connection between nodes, or neurons, in an artificial neural network (ANN). Similar to biological brains, the connection is controlled by the strength or amplitude of a connection between both nodes, also called the synaptic weight. Multiple synapses can connect the same neurons, with each synapse having a different level of influence (trigger) on whether that neuron is “fired” and activates the next neuron.

 ![]( {{site.assets}}/s/synapse.png ){: width="100%}

 See also [S], [Activation Function], [Artificial Neuron], [Artificial Neural Network], [Neural Network]


# Synthesia Company

 {% youtube "https://www.youtube.com/watch?v=G-7jbNPQ0TQ" %}

 {% youtube "https://www.youtube.com/watch?v=UVNUCBUrHL0" %}

 More at:
  * [https://www.synthesia.io/about](https://www.synthesia.io/about)

 See also [S], [Company], [AI Avatar]


# Synthesized Variable

 EX: cube on paper --> visualize it as a 2-d object. Move from n --> K dimensions (called eigenvectors).

 ![]( {{site.assets}}/s/synthesized_variable.png ){: width="100%}

 Use the centre of gravity, i.e regression line for the projection. --> transpose to the projection on the 'best' regression line! The best line is used with the gradient descent. We use the 'best' line to project to the new dimension by losing the minimum quantity of information. (miximize the adjacent line of the triangle and minimize the T line. ... Q: How much information have we lost? ...
 See also [S], [Eigenvalue], [Principal Component Analysis]


# Synthetic Data

 Synthetic data is a tool that addresses many data challenges, particularly artificial intelligence and analytics issues such as privacy protection, regulatory compliance, accessibility, data scarcity, and bias, as well as data sharing and time to data (and therefore time to market).

 {% pdf "{{site.assets}}/s/synthetic_data_booklet.pdf" %}

 See also [S], ...
