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

 Part of AI, AI with first a learning phase! A subset of [Artificial Intelligence (AI)][AI]. `Field of study that gives computers the ability to learn without being explicitly programmed` 1 or more layers of data, includes but not limited to neural networks. Unsupervised, supervised (classification + regression) , reinforcement. `Data --> Model --> Prediction`.

 ![]( {{site.assets}}/m/machine_learning.png ){: width="100%"}

 ```
A company is using a rule-based system to classify the credit card transactions as fraud or not fraud. Do you think this a machine learning solution?
No! (but why? because it is a rule-based system?)

A company receives thousands of calls everyday to route to the right agent. The routing requires several hops and given the number of calls is expensive.
What is a machine learning solution?
Predict what are the required agent skills based some input parameters
==> That's a multiclass classification problem!
 ```

 See also [M], [Deep Learning], [Machine Learning Framework], [Machine Learning Pipeline], [Neural Network], [Prediction]


# Machine Learning Framework

  * [pytorch]
  * [tensorflow]
  * [JAX]

 Watch for:
  * eager mode (execute like a python script, from top to bottom)
  * graph format and execution engine natively has no need for Python, and TensorFlow Lite and TensorFlow Serving address mobile and serving considerations respectively.

 More at :
  * [https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/](https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/)

 See also [M], ...


# Machine Learning Pipeline

 1. ML problem framing (aka discovery phase)
 1. data collection and integration
  * :warning: [dataset] can be incomplete and therefore misleading
  * :warning: data in dataset can be irrelevant
  * :warning: bias in dataset?
  * :warning: dataset may be too small (needs to be at least 10 times the number of features!)
  * :warning: needs to be cleaned?
 1. [data preparation]
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
 1. [data visualization] and analysis
  * understand the relationships in a dataset
  * find outliers (use histograms)
  * find groups
  * use imputation to complete data
  * pie charts, histogram, scatter plots for correlation
  * try to extract noise from data, noise causes overfitting and reduce accuracy of predictions
 1. [feature selection] and [feature engineering]
  * you want a minimum correlation between the features, but the maximum correlation between the feature and the output
  * do the feature I use make sense?
  * very time consuming step!
  * ex: what was the last item purchased by the customer?
  * ex: when was the last purchase of the customer?
  * ex: do the customer owns a kindle? (then do not expect questions related to kindle)
 1. model training
  * model selection
  * model evaluation
    * confusion matrix for classification
    * [overfitting] ?
 1. deployment for prediction / inference

 See also [M], [Discovery Phase], [Machine Learning], [Machine Learning Framework]


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

 Google Translate, DeepL, and other machine translation programs use [Natural Language Processing (NLP)][Natural Language Processing]  to evaluate millions of sentences translated by human speakers of different language pairs.

 Paradigms:
  * [Example-Based Machine Translation]
  * [Statistical Machine Translation]
  * ...
  * [Neural Machine Translation]

 See also [M], [Natural Language Processing]


# Magi Model

 New search engine based on an AI model built by [Google] ?

 More at:
  * [https://blog.google/technology/developers/google-io-2023-100-announcements/](https://blog.google/technology/developers/google-io-2023-100-announcements/)

 See also [M], ...


# Majority Vote Algorithm

 When you crowdsource a labeling task, how can you be certain that the label is correct? Have several people label the same image/entry and apply this algorithm! An alternative is to use Dawid-Skene algorithm.

 See also [M], [Dawid-Skene Algorithm], [Labeling Service]


# Make-A-Video Model

 A state-of-the-art AI system that generates videos from text built by [Meta].

 {% pdf "{{site.assets}}/m/make_a_video_model_paper.pdf" %}

 More at:
  * [https://makeavideo.studio/](https://makeavideo.studio/)
  * [https://arxiv.org/abs/2209.14792](https://arxiv.org/abs/2209.14792)

 See also [M], ...


# MapReduce Process

> Move the processing and not the data + process in parallel and combine

 At [Google], MapReduce was used to completely regenerate Google's index of the World Wide Web. It replaced the old ad hoc programs that updated the index and ran the various analyses.

 MapReduce is a programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster.

 MapReduce is a framework for processing parallelizable problems across large datasets using a large number of computers (nodes), collectively referred to as a cluster (if all nodes are on the same local network and use similar hardware) or a grid (if the nodes are shared across geographically and administratively distributed systems, and use more heterogeneous hardware). Processing can occur on data stored either in a filesystem (unstructured) or in a database (structured). MapReduce can take advantage of the locality of data, processing it near the place it is stored in order to minimize communication overhead.

 More at:
  * [https://en.wikipedia.org/wiki/MapReduce](https://en.wikipedia.org/wiki/MapReduce) 

 See also [M], [Big Data]


# Markov Chain

 * stochastic model
 * describe sequence of possible events
 * what happens next depends on present state
 * memory-less
 * transition matrix

 More at:
  * [https://setosa.io/ev/markov-chains/](https://setosa.io/ev/markov-chains/)

 See also [M], ...


# Markov Decision Process

# MDP

 Markov Decision Process is a [Reinforcement Learning] algorithm that gives us a way to formalize sequential decision making. This formalization is the basis to the problems that are solved by Reinforcement Learning. The components involved in a Markov Decision Process (MDP) is a decision maker called an [agent] that interacts with the [environment] it is placed in. These interactions occur sequentially overtime. In each timestamp, the [agent] will get some representation of the environment state. Given this representation, the agent selects an [action] to make. The environment is then transitioned into some new state and the agent is given a reward as a consequence of its previous [action]. The process of selecting an [action] from a given state, transitioning to a new state and receiving a reward happens sequentially over and over again. This creates something called a trajectory that shows the sequence of [states], [actions] and [rewards]. Throughout the process, it is the responsibility of the reinforcement learning [agent] to maximize the total amount of [rewards] that it received from taking actions in given states of environments. `The agent not only wants to maximize the immediate rewards but the [cumulative reward] it receives in the whole process.`

 {% youtube "https://www.youtube.com/watch?v=2GwBez0D20A" %}

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


# Matplotlib Python Module

 A [python module] used for visualization

 ```
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
 ```

 More at:
  * [https://matplotlib.org/](https://matplotlib.org/)
  * examples - [https://matplotlib.org/stable/plot_types/index.html](https://matplotlib.org/stable/plot_types/index.html)

 See also [M], ...


# Matrix

 There is no definition of matrix, but only the interpretations of matrices
  * For an engineer who study circuits, matrix are just a way to solve system of equations
  * For a statistician, matrices are a representation of a Markovian process. 
  * For a data scientist, a matrix is the representation of a table and facilitate data analysis
  * For deep learning folks, a matrix is another python function which takes a vector as input and outputs a vector

 Special (linear-transformation) matrices
  * Identity Matrix: A transformation that results in the same output as the input
  * Scalar Matrix:
  * Identity matrix, but off by one:
  * Diagonal matrix: Stretch or scale each axis :warning: - sign creates a reflection!
  * Square zero matrix: All outputs is ZERO regardless of the input matrix
  * Shear matrix
  * Orthogonal matrix: Produce an improper rotation (pure rotation with or without reflection = no vector is stretched longer or shorter)
  * Projection matrix: You project a vector in a lower dimension space :warning: you lose info, you cannot invert those matrix!
  * Invertible matrix: To undo a linear transformation

 Special matrices
  * Invertible matrix: To undo a linear transformation
  * Transpose matrix: For square matrices only? No. Row becomes columns. Transpose of a MxN matrix is of NxM dimension. Top-bottom diagonal stays the same!
  * Square matrix: Number of columns = number of rows
  * Square symmetric matrix: Aij = Aji . Symmetric based on the top-bottom diagonal. S = St ! Eigenvectors are perpendicular/orthogonal!
  * Non-square (i.e. rectangular or Mcol x Nrow) matrices: projection in higher or lower dimensions. Mcol x Nrow . Ncol = Mrow
  * Orthogonal matrix: Produce an improper rotation. :warning: Q orthogonal => Qt = Qinv or inverse rotation and therefore orthogonal as well!!
  * Matrix of any dimension Mrow x Ncol
    * A . At = square symmetric matrix (with dimension Mrow x Mrow )!!! = Sleft
    * At . A = square symmetric matrix (with dimension Ncol x Ncol )!!! = Sright

 ```
 Special Non-square matrices

  * Dimension eraser: Keep X and Y, but remove Z     | 1 0 0 |
                                                     | 0 1 0 |

  * Dimension adder: Append zero as Z value          | 1 0 |
                                                     | 0 1 |
			                             | 0 0 |
 ```
  

 Special vectors:
  * [Eigenvectors][Eigenvector]: While every other vector deviates from their initial direction, the eigne vectors stay on their original lines despite the distortion from the matrix.

 Special values:
  * [Eigenvalues][Eigenvalue]: By how much the eigenvector is transformed on its original direction by the transformation

 See also [M], [Linear Transformation]


# Matrix Composition

 Multiplication of some matrices ==(Easy)==> one matrix
 
 Matrix Multiplication is not a multiplication at all! It is a composition of transformations.
 N multiplications of matrices = N linear transformations = 1 complex linear transformation (That's what defines linear transformations!)
 That is why Matrix multiplication is not commutative!

 {% youtube "https://www.youtube.com/watch?v=7Gtxd-ew4lk" %}

 {% youtube "https://www.youtube.com/watch?v=wciU07gPqUE" %}

 See also [M], [Matrix Decomposition]


# Matrix Decomposition

 One matrix ==(Difficult)===> multiplication of some matrices

 If a matrix is a linear transformation, its decomposition consists in turning that "complex" matrix into "simpler" matrices (or a succession of simpler linear transformations)

 Special decompositions:
  * Square symmetric matrices ==> [Spectral decomposition][Matrix Spectral Decomposition]
  * Jordan Decomposition
  * QR decomposition
  * Other matrices (rectangular matrices) ==> [Singular value decomposition]

 See also [M], [Matrix Composition]


# Matrix Null Space

 From this definition, the null space of A is the set of all vectors such that A.V=0 .
 Obviously V=[0,0,0,...,0] is part of the null space, so it is always non-empty.

 See also [M], [Matrix]


# Matrix QR Decomposition

 In linear algebra, a QR decomposition, also known as a QR factorization or QU factorization, is a [decomposition][Matrix Decomposition] of a matrix A into a product A = QR of an orthonormal matrix Q and an upper triangular matrix R. QR decomposition is often used to solve the linear least squares problem and is the basis for a particular [eigenvalue] algorithm, the QR algorithm.

 More at:
  * [https://en.wikipedia.org/wiki/QR_decomposition](https://en.wikipedia.org/wiki/QR_decomposition)

 See also [M], ...


# Matrix Rank

 The rank of the matrix is related to the [range][Matrix Range]. It denotes how many columns of 𝐴
 are actually "relevant" in determining its range. You may think that removing a column from a matrix will dramatically affect which vectors it can reach, but consider:

 ```
 | 1 2 0 |                     | 1 |
 | 1 2 0 | ~~(same range as)~~ | 1 |
 | 1 2 0 |                     | 1 |
 ```

 You can try to reason (to yourself), that the left matrix can reach the same space of vectors as the right matrix (Why?)

 See also [M], ...


# Matrix Range

 In the simplest terms, the range of a matrix is literally the "range" of it. The crux of this definition is essentially

 > Given some matrix A , which vectors can be expressed as a linear combination of its columns?

 Range (another word for column space) is what is meant by this. If you give me some matrix A that is MxN, the column space is the set of all vectors such that there exists a1,a2,....,an so that a1A1+a2A2+...anAn = V for some vector V .

 ```
 | 1 0 0 |  | a1 |     | 5 |
 | 0 1 0 |  | a2 |  =  | 5 |
 | 0 0 1 |  | a3 |     | 5 |
 ```
 Then V is in the range of A since a1=a2=a3=5 . A better example is when it's not, like:

 ```
 | 1 0 3 |   | a1 |     | 5 |
 | 1 1 2 |   | a2 |  =  | 5 |
 | 0 0 0 |   | a3 |     | 5 |
 ```
 Now it's not... since no a1,a2,a3 will satisfy the condition that V is a linear combination of the columns of A ...I mean, we will always have 0 in the third entry of any linear combination!

 More at:
  * [https://math.stackexchange.com/questions/2037602/what-is-range-of-a-matrix](https://math.stackexchange.com/questions/2037602/what-is-range-of-a-matrix)

 See also [M], ...


# Matrix Spectral Decomposition

 A special [decomposition][Matrix Decomposition] for square symmetric [matrix]

 ```
S = Q D Qt 
# where
# Q orthogonal with eigenvectors as columns
# D is diagonal with eigenvalues = stretch X axis by EV1, Y axis by EV2, etc.
# Qt also orthogonal with eigenvectors as rows (transpose)
# REACTION ==> improper rotation + stretching + improper rotation

<!> improper rotation can be turned into pure rotation if sign of eigenvectors is cleverly picked
<!> -1 * eigenvectors is also an eigenvector with same eigenvalue (?)
```

 ![]( {{site.assets}}/m/matrix_spectral_decomposition.png )

 More at:
  * [https://www.youtube.com/watch?v=mhy-ZKSARxI](https://www.youtube.com/watch?v=mhy-ZKSARxI)

 See also [M], ...

# Matrix Determinant

 {% youtube "https://www.youtube.com/watch?v=Ip3X9LOh2dk" %}

 See also [M], [Matrix]


# Matrix Multiplication

  See [Matrix Composition]

# Max Pooling Layer

 Downsample the feature map (take a thumbnail/smaller-size of the image with some feature still present). while a convoluted layer extract features. In the illustration below, we take a 2x2 kernel and pool the maximum value. Benefits with this approach are
  * discovered features in the previous convoluted Layer are preserved as we are keeping the max!
  * image is smaller <== A stack of images becomes a smaller stack of images
  * because of down sampling, the position of the exact match is not as important/sensitive


 ![]( {{site.assets}}/m/max_pooling_layer.png ){: width="100%"}

 See also [M], [Convoluted Layer], [Convoluted Neural Network], [Fully Connected Layer]


# Mean Absolute Error Loss Function

# MAE Loss Function

 The Mean Absolute Error (MAE) [loss function] is only slightly different in definition from the [MSE], but interestingly provides almost exactly opposite properties! To calculate the MAE, you take the difference between your model’s predictions and the [ground truth], apply the absolute value to that difference, and then average it out across the whole [dataset]. The MAE, like the [Mean Square Error (MSE)][MSE], will never be negative since in this case we are always taking the absolute value of the errors. The MAE is formally defined by the following equation:

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
  * Advantage: The beauty of the MAE is that its advantage directly covers the [MSE] disadvantage. Since we are taking the absolute value, all of the errors will be weighted on the same linear scale. Thus, unlike the [MSE], we won’t be putting too much weight on our [outliers] and our [loss function] provides a generic and even measure of how well our model is performing.
  * Disadvantage: If we do in fact care about the [outlier] predictions of our model, then the MAE won’t be as effective. The large errors coming from the [outliers] end up being weighted the exact same as lower errors. This might results in our model being great most of the time, but making a few very poor predictions every so-often.

 See also [M], [Huber Loss Function]


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

 See also [M], [Nvidia]


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

 A [company] previously known as [Facebook]

 Models:
  * [CICERO][CICERO Model]: Strategy game with multiplayer interaction
  * [ESMFold][ESMFold Model]: Protein folding
  * [LLaMA][LLaMA Model]: Large Language Model open-sourced
  * [Make-A-Video][Make-A-Video Model]: Text to video model
  * [Pluribus][Pluribus Model]: Plays poker better than humans
  * [RoBERTa][RoBERTa Model]: Optimized version of BERT
  * [Segment Anything][Segment Anything Model]: Instance segmentation in images
  * [Fairseq Toolkit] - Facebook AI Research Toolkit
    * [Wav2Vec] - For [Automatic Speech Recognition (ASR)][ASR]

 More at:
  * research on github - [https://github.com/facebookresearch](https://github.com/facebookresearch)

 See also [M], ...


# Meta Model

 A ML model to find the best hyperparameters. Ex: Gaussian process regression models object metric as a function of hyperparameters (beware assume smoothness, works with low data, confidence estimates) + bayesian optimization decides where to search next (explore and exploit and gradient free) .

 See also [M], [Hyperparameter Optimization], [Meta Learning]


# Meta-Learning

 Learn how to quickly adapt to new tasks within similar domains.

 Is a [sample efficient RL algorithm]

 Normally you take the x to predict the y and optimize parameters to get as close to y as possible. Here you take the x and y to generate a theta parameter to fit in another model... then use the loss of the aggregate model to

 ![]( {{site.assets}}/m/meta_learning.png ){: width="100%"}

 See also [M], [Meta Model], [Model Agnostic Meta Learning], [Reinforcement Learning], [Transfer Learning]


# Metaverse

 Think of it as the internet brought to life, or at least rendered in 3D. Zuckerberg has described it as a "virtual environment" you can go inside of — instead of just looking at on a screen. Essentially, it's a world of endless, interconnected virtual communities where people can meet, work and play, using virtual reality headsets, augmented reality glasses, smartphone apps or other devices.

 More at:
  * [https://www.npr.org/2021/10/28/1050280500/what-metaverse-is-and-how-it-will-work](https://www.npr.org/2021/10/28/1050280500/what-metaverse-is-and-how-it-will-work)

 See also [M], [The Matrix Movie], [Virtual Reality]


# METEOR Score

 See also [M], [MSFT COCO Caption Dataset]


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

 {% pdf "https://arxiv.org/pdf/1405.0312.pdf" %}

 More at :
  * home - [https://cocodataset.org/#home](https://cocodataset.org/#home)
  * paper - [https://arxiv.org/abs/1405.0312](https://arxiv.org/abs/1405.0312)

 See also [M], [BLEU Score], [CIDEr Score], [Dataset], [METEOR Score], [ROUGE Score]


# Microsoft Company

# MSFT Company

 See also [M], [Company], [DeepSpeed Project], [OpenAI]


# Midjourney Model

 {% youtube "https://www.youtube.com/watch?v=onjfu3Uh2vI" %}

 More at:
  * [https://promptomania.com/midjourney-prompt-builder/](https://promptomania.com/midjourney-prompt-builder/)

 See also [M], [Diffusion Model]


# Milvus Vector Database

 An open-source [vector database] that is highly flexible, reliable, and blazing fast.

 ![]( {{site.assets}}/m/milvus_database.png ){: width="100%"}

 More at:
  * site - [https://milvus.io/](https://milvus.io/)

 See also [M], [Vector Database]


# MindPong Game

 {% youtube "https://www.youtube.com/watch?v=rsCul1sp4hQ" %}

 More at:
  * ...

 See also [M], [Neuralink]


# Minerva Model

 Can AI change mathematics?

 More at:
  * [https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html)
  * [https://www.nature.com/articles/d41586-023-00487-2](https://www.nature.com/articles/d41586-023-00487-2)

 See also [M], [Google]


# Mini-Batch

 Take your training samples. Randomized the samples. Take the first 100 of them (the mini-batch).

 See also [M], [Mini-Batch Gradient Descent]


# Mini-Batch Gradient Descent Algorithm

 ~ [stochastic gradient descent][SGD Algorithm] with more than 1 sample that use matrix optimization in computation. A compromise between computing the true gradient (all samples) and the gradient at a single example (aka stochastic gradient descent), is to compute the gradient against more than one training example (called a "mini-batch") at each step. This can perform significantly better than true stochastic gradient descent because the code can make use of vectorization libraries rather than computing each step separately. It may also result in smoother convergence, as the gradient computed at each step uses more training examples.

 ![]( {{site.assets}}/m/mini_batch_gradient_descent.png ){: width="100%"}

 So, after creating the mini-batches of fixed size, we do the following steps in one epoch:
  * Pick a mini-batch
  * Feed it to Neural Network
  * Calculate the mean gradient of the mini-batch
  * Use the mean gradient we calculated in step 3 to update the weights
  * Repeat steps 1–4 for the mini-batches we created

 Just like SGD, the average cost over the [epochs] in mini-batch gradient descent fluctuates because we are averaging a small number of examples at a time. But the mini-batch gradient descent is a good (the best) approximation. Also note that the path is less optimized than the "complete" gradient descent where you take all sample into consideration for one step, but it is a good approximation and the destination/convergence is the same = the local minima is not changed by the path taken to reach it.

 See also [M], [Batch Gradient Descent Algorithm], [Gradient Descent Algorithm], [Mini-Batch]


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

 MLOps = People + Technology + Process

 ![]( {{site.assets}}/m/ml_mlops.png ){: width="100%"}

 See also [M], [DevOps]


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

 {% pdf "https://arxiv.org/pdf/1810.03993.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/1810.03993](https://arxiv.org/abs/1810.03993)
  * model cards at google - [https://modelcards.withgoogle.com/about](https://modelcards.withgoogle.com/about)
  * facebook LLaMa model card - [https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)

 See also [M], [Model Data Sheet]


# Model Checkpoint

 A snapshot of a model with its weights.
 Is usually used for [model evaluation] before being released.

 See also [M], ...


# Model Complexity

 ![]( {{site.assets}}/m/model_complexity.jpeg ){: width="100%"}

 See also [M], [Bias], [Variance]


# Model Convergence

 When we say a machine learning model has converged, it generally means that the model has stabilized during training and additional training is not producing significant changes or improvements in its parameters or performance on the training set. Here are some key points about model convergence:

  * During training, the model parameters are updated iteratively to minimize a loss function. As this loss gets lower, the model fits the training data better.
  * In the initial stages of training, the loss decreases rapidly with each update as the model rapidly learns. But over time, the rate of improvement slows and eventually plateaus.
  * We say the model has converged when the loss flattens out and further parameter updates lead to diminishing or negligible improvements. The model has fit the training data as well as it can.
  * Convergence happens when the learning rate drops to near zero. The updates to parameters become very small. The model is no longer improving meaningfully.
  * Checking for convergence helps decide when to stop training. Continuing training post-convergence is wasteful.

 In the context of [RL], with more experience, the agent gets better and eventually is able to reach the destination reliably. Depending on the exploration-exploitation strategy, the vehicle may still have a small probability of taking random actions to explore the environment.

 ![]( {{site.assets}}/m/model_convergence_reinforcement_learning.png ){: width="100%"}

 :warning: Well, this is not true, because Convergence does not necessarily mean the model has achieved optimal performance. The model may converge to a local minimum. Regularization and hyperparameter tuning is still important.

 In the case of [RL[, the model may have converge, but if the reward function is not programmed correctly, we may see a converged ehavior, but not the desired behavior!

 More at:
  * ...

 See also [M], ...


# Model Cost

 More at:
  * estimates for claude model - [https://orenleung.com/anthropic-claude-next-cost](https://orenleung.com/anthropic-claude-next-cost)


# Model Drift

 Model performance changes over time.

 See also [M], [Confusion Matrix]


# Model Data Sheet

 More at:
  * [https://www.microsoft.com/en-us/research/project/datasheets-for-datasets/](https://www.microsoft.com/en-us/research/project/datasheets-for-datasets/)

 See also [M], [Model Card]


# Model Evaluation

 Use a [model checkpoint] with a benchmark tool or process for evaluation.

 In the case of [RL], the benchmark process is a simulation in a virtual environment.

 ![]( {{site.assets}}/m/model_evaluation_reinforcement_learning.png ){: width="100%"}

 More at:
  * ...

 See also [M], ...


# Model Format

 * [CoreML Format] - Apple format
 * [ONNX Format] - Opensource format
 * [TorchScript Format] - A PyTorch format with no dependency on Python

 See also [M], ...

# Model Hub

 See also [M], ...


# Model Performance Metrics

  * percentage of correct predictions
  * sensitivity, recall, hit rate or TPR
  * precision
  * F1 score = wiegthed harmonic mean of precision and recall = ( 2 X precision * recall ) / (Precision + recall)
  * FPR, FNR

  [Hyperparameter Tuning]

 See also [M], ...

# Model Tuning

 done with gradient descent?


# Model Type

 ![]( {{site.assets}}/m/model_type.png ){: width="100%"}

 More at :
  * [https://youtu.be/J8Eh7RqggsU?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX&t=2568](https://youtu.be/J8Eh7RqggsU?list=PLoROMvodv4rO1NB9TD4iUZ3qghGEGtqNX&t=2568)

 See also [M], [Reflex Model], [Logic Model], [State Model], [Variable Model]


# Model Uplift

 Improvement of the model performance due to the usage of [synthetic data].

 Comparison of the performance of a model trained and tested with real data vs the performance of the same model trained with real and synthesize data and tested on real data.

 :warning: You do not test on synthetic data!

 See also [M], ...


# Model-Based Reinforcement Learning
# Model-Based RL

 Model-based RL approaches explicitly learn a model of [state transitions]. [Model-free RL] methods like [Q-learning] learn without modeling transitions.

 In [Reinforcement Learning], you learns model of [environment] transitions & [rewards], then optimizes [policy] through planning. e.g. [Dyna], [AlphaGo].

 Given a state and an action, the model might predict the next state and next reward. Models are used for deciding on a course of actions, by taking into account possible future situations before they are actually experienced.

 Model-based reinforcement learning has an agent try to understand the world and create a model to represent it. Here the model is trying to capture 2 functions, the transition function from states 𝑇 and the reward function 𝑅. From this model, the agent has a reference and can plan accordingly. A simple check to see if an RL algorithm is model-based or model-free is:
  * If, after learning, the agent can make predictions about what the next state and reward will be before it takes each action, it's a model-based RL algorithm.
  * If it can't, then it’s a model-free algorithm.

 {% youtube "https://www.youtube.com/watch?v=vfpZu0R1s1Y" %}

 More at:
  * ...

 See also [M], ...


# Model-Based Reinforcement Learning Algorithm
# Model-Based RL Algorithm

 More at:
  * ...

 See also [M], ...


# Model-Free Reinforcement Learning
# Model-Free RL

 [Model-based RL] approaches explicitly learn a model of [state transitions]. Model-free RL methods like [Q-learning] learn without modeling transitions.
 The agents here are explicitly trial-and-error learners. (?)

 Model-free learning is a category of [reinforcement learning (RL)][RL] algorithms that do not require a complete model of the environment to make decisions. In model-free learning, the agent learns directly from interacting with the environment, without explicitly building or using a model that represents the environment's dynamics or transition probabilities.

 Instead of modeling the environment, model-free learning algorithms focus on estimating value functions or directly learning policies through trial-and-error interactions. The two primary components in model-free learning are the [policy] and the value function:

 * Policy: The policy determines the agent's behavior by mapping states or state-action pairs to actions. The policy can be deterministic (selecting a single action) or stochastic (selecting actions based on probabilities). Model-free learning algorithms aim to find an optimal policy that maximizes the expected cumulative reward over time.

 * Value Function: The value function estimates the expected long-term return or cumulative reward associated with being in a particular state or taking a specific action in a given state. It represents the desirability or utility of states or actions. Value functions can be estimated through various techniques such as Monte Carlo methods, temporal difference learning, or function approximation.

 * Model-free learning algorithms learn by iteratively updating the value function or policy based on the observed [rewards] and [states] during interactions with the environment. These updates are typically driven by optimization principles, such as maximizing cumulative rewards or minimizing the difference between estimated and observed values.

 Model-free learning is suitable in scenarios where it is difficult or impractical to obtain a complete model of the environment, and the [agent] must learn directly from experience.

 A model-free RL algorithm can be thought of as an "explicit" trial-and-error algorithm. A simple check to see if an RL algorithm is model-based or model-free is:
  * If, after learning, the agent can make predictions about what the next state and reward will be before it takes each action, it's a model-based RL algorithm.
  * If it can't, then it’s a model-free algorithm.

 Algorithms that purely sample from experience such as Monte Carlo Control, [SARSA], [Q-learning], Actor-Critic are "model free" RL algorithms. They rely on real samples from the [environment] and never use generated predictions of next [state] and next [reward] to alter behaviour (although they might sample from experience memory, which is close to being a model).

 More at:
   * ...

 See also [M], ...


# Model-Free Reinforcement Learning Algorithm
# Model-Free RL Algorithm

 Examples of model-free learning algorithms include:
  * [Q-learning], 
  * [State-Action-Reward-State-Action (SARSA)][SARSA],
  * and [REINFORCE].

 ![]( {{site.assets}}/m/model_free_learning_algorithm.png ){: width="100%"}

 See also [M], ...


# Modified National Institute of Standards and Technology
# MNIST Dataset

 Pictures of numbers written by college student taken by the post office to be able to sort the zip codes. Conversion from every image to matrix was done by hand.

 First approach to solve the problem was to use One of the first [neural networks], 
  * layer 1: 200 neurons
  * 2nd layer: 100
  * 3rd: 60
  * 4th: 30
  * 5th: 10
  * with [sigmoid activation function]
 The [artificial neural network] has a inverted tree like structure. Unfortunately the [accuracy] of 92% could be improved!

 A better model found later was the [Convoluted Neural Network (CNN)][CNN]

 The first part of the [CNN] is called [feature extraction], the second part is the [classification].

 ![]( {{site.assets}}/m/mnist_dataset.png ){: width="100%"}

 See also [M], [Dataset]


# Modular Reasoning Knowledge and Language

# MRLK

 {% pdf "https://arxiv.org/pdf/2205.00445.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2205.00445](https://arxiv.org/abs/2205.00445)

 See also [M], [ReACT Prompting]


# Monte Carlo Policy Gradient Algorithm

 See [REINFORCE Algorithm]


# MuLan Model

 Similar to the [CLIP model], but for music! Music to text model!

 {% pdf "https://arxiv.org/pdf/2208.12415.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2208.12415](https://arxiv.org/abs/2208.12415)
  * code - [https://github.com/lucidrains/musiclm-pytorch](https://github.com/lucidrains/musiclm-pytorch)

 See also [M], ...


# Multi-Agent Environment

 An [environment] where several agent can coexist?

 See also [M], [Game Theory]


# Multi-Agent Reinforcement Learning
# Multi-Agent RL

 Learns policies for multiple interacting agents. Emergent coordination & competition.

 More at:
  * ...

 See also [M], ...


# Multi-Head Self-Attention

 ~ the brains of the Transformer and responsible for performance through parallelism. Multi-Head Attention consists of several attention layers running in parallel. The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value (aka Q,K,V). All three parameters are similar in structure, with each word in the sequence represented by a vector. In transformers is used for encoder and decoder.

 ![]( {{site.assets}}/m/multi_head_attention.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=g2BRIuln4uc" %}

 {% youtube "https://www.youtube.com/watch?v=PFczJ6NR5rY" %}

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


# Multiclass Classification

 ML for classification in more than 2 categories.

 See also [M], [Binary Classification], [Classification]


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


# Multinomial Naive Bayes Classifier

 {% youtube "https://www.youtube.com/watch?v=O2L2Uv9pdDA" %}

 {% youtube "https://www.youtube.com/watch?v=YbHbsaJhhKM" %}

 See also [M], [Naive Bayes Theorem]


# Multiple Linear Regression

 A linear regression with multiple input / independent variable.
 ```
Y = a + b.X1 + c.X2
 ```

 More at:
  * sample paper - [https://www.mdpi.com/2073-4395/11/5/885](https://www.mdpi.com/2073-4395/11/5/885)

 See also [M], [Linear Regression]


# MUM Model
 
 MUM was developed by [Google], uses the [T5][T5 Model] text-to-text framework, and is 1,000 times more powerful than [BERT][BERT Model].

 T5 uses transformer-based architecture, just like BERT, but instead uses a text-to-text approach. What that means is with T5, the input (query) and output (result) are always text strings, in contrast to BERT-style models that can only output either a classification label or the span of the input into a question and answer format. This means that the output with BERT, while undeniably impressive, was in comparison to MUM still rather abstract.

The T5 text-to-text process includes more in-depth machine translation, document summarization, question answering, and classification tasks (e.g., sentiment analysis).

 More at:
  * [https://blog.google/products/search/introducing-mum/](https://blog.google/products/search/introducing-mum/)
  * [https://www.upbuild.io/blog/what-is-google-mum/](https://www.upbuild.io/blog/what-is-google-mum/)

 See also [M], ...


# Muse Model

 Muse is a fast, state-of-the-art text-to-image generation and editing model built by [Google].

 We present Muse, a text-to-image Transformer model that achieves state-of-the-art image generation performance while being significantly more efficient than diffusion or autoregressive models. Muse is trained on a masked modeling task in discrete token space: given the text embedding extracted from a pre-trained large language model (LLM), Muse is trained to predict randomly masked image tokens. Compared to pixel-space diffusion models, such as [Imagen][Imagen Model] and [DALL-E 2][DALL-E Model], Muse is significantly more efficient due to the use of discrete tokens and requiring fewer sampling iterations; compared to autoregressive models, such as [Parti][Parti Model], Muse is more efficient due to the use of parallel decoding. The use of a pre-trained LLM enables fine-grained language understanding, translating to high-fidelity image generation and the understanding of visual concepts such as objects, their spatial relationships, pose, cardinality, etc. Our 900M parameter model achieves a new SOTA on CC3M, with an FID score of 6.06. The Muse 3B parameter model achieves an FID of 7.88 on zero-shot COCO evaluation, along with a CLIP score of 0.32. Muse also directly enables a number of image editing applications without the need to fine-tune or invert the model: inpainting, outpainting, and mask-free editing.

 {% pdf "{{site.assets}}/m/muse_model_paper.pdf" %}

 More at:
  * [https://muse-model.github.io/](https://muse-model.github.io/)
  * paper [https://arxiv.org/abs/2301.00704](https://arxiv.org/abs/2301.00704)
  * [https://venturebeat.com/ai/googles-muse-model-could-be-the-next-big-thing-for-generative-ai/](https://venturebeat.com/ai/googles-muse-model-could-be-the-next-big-thing-for-generative-ai/)

 See also [M], ...


# MuseGAN

 4 inputs: (1) Chords input (with temporal network) (2) style input (3) melody input (temporal network) (4) groove input

 See also [M], ...


# Music ABC Notation

 More at:
  * [https://en.wikipedia.org/wiki/ABC_notation](https://en.wikipedia.org/wiki/ABC_notation)
  * [https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177](https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177)

 See also [M], ...


# Music Generator

 More at:
  * with transformer - [https://towardsdatascience.com/creating-a-pop-music-generator-with-the-transformer-5867511b382a](https://towardsdatascience.com/creating-a-pop-music-generator-with-the-transformer-5867511b382a)
  * with LSTM - [https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177](https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177)
  * with RNN - [https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177](https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177)
  * with transformer -

 See also [M], ...


# MusicLM Model

 An impressive new AI system from [Google] can generate music in any genre given a text description. Called MusicLM, Google’s certainly isn’t the first generative artificial intelligence system for song. There have been other attempts, including Riffusion, an AI that composes music by visualizing it, as well as Dance Diffusion, Google’s own AudioML and OpenAI’s Jukebox. But owing to technical limitations and limited training data, none have been able to produce songs particularly complex in composition or high-fidelity. MusicLM is perhaps the first that can.

 {% pdf "{{site.assets}}/m/musiclm_model_paper.pdf" %}

 More at:
  * twitter - [https://twitter.com/keunwoochoi/status/1618809167573286912](https://twitter.com/keunwoochoi/status/1618809167573286912)
  * paper - [https://arxiv.org/abs/2301.11325](https://arxiv.org/abs/2301.11325)
  * example - [https://google-research.github.io/seanet/musiclm/examples/](https://google-research.github.io/seanet/musiclm/examples/)
  * dataset - [https://www.kaggle.com/datasets/googleai/musiccaps](https://www.kaggle.com/datasets/googleai/musiccaps)
  * techcrunch article - [https://techcrunch.com/2023/01/27/google-created-an-ai-that-can-generate-music-from-text-descriptions-but-wont-release-it/](https://techcrunch.com/2023/01/27/google-created-an-ai-that-can-generate-music-from-text-descriptions-but-wont-release-it/)

 See also [M], [Jukebox Model], [Riffusion Model]


# MuZero Model

 A model built by [DeepMind]

 ![]( {{site.assets}}/m/muzero_model.jpeg ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=vt5jOSy7cz8" %}

 {% youtube "https://www.youtube.com/watch?v=L0A86LmH7Yw" %}

 {% youtube "https://www.youtube.com/watch?v=pgZhGavMHcU" %}

 More at:
  * nature article - [https://www.nature.com/articles/s41586-020-03051-4.epdf](https://www.nature.com/articles/s41586-020-03051-4.epdf)

 See also [M], ...


# MXNET

 See also [M], [Deep Learning Framework]
