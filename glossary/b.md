---
title: B
permalink: /b/

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


# Backpropagation

 = a brute force approach, where you pick random weight and you iterate on them until they arrive at a stable solution. This is a `widely used algorithm for training feedforward neural networks and other ANN`. Approach discovered in 1986 that re-stimulated AI. Help a model learn from its mistakes by leveraging the chain rule of derivatives. The backpropagation algorithm consists in modifying the weight and bias of each cell in each layer based on their impact on the estimated output, or loss function (diff between estimated output and real output).

 {% youtube "https://www.youtube.com/watch?v=Ilg3gGewQ5U" %}

 {% youtube "https://www.youtube.com/watch?v=tIeHLnjs5U8" %}

 Backpropagation can find the
  1. weights + biases (?)
  1. kernel filters in a CCN
 Beware:
  * If you only use 2 in training sample, you may have a model where all images are recognized as 2, which is not correct. ==> the weights need to be computed with all the samples (i,e, an epoch or a mini-batch)!
  * If your ANN is multi-layered, deep, and use activation functions, backpropagation may not be able to compute all the weights, an issue that is known as the vanishing gradient problem.
  * In a variational autoencoder, you cannot run backpropagation because of the sampling between the encoder and decoder. The solution here is to us the "VAE reparametrization trick"

 See also [B], [Activation Function], [Derivative Chain Rule], [Feedforward Neural Network], [Kernel Filter], [Loss Function], [Neural Network], [Vanishing Gradient Problem], [Variational Autoencoder Reparametrization Trick]

# Bag Of Words

 A technique for natural language processing that extracts the words (features) used in a sentence, document, website, etc. and classifies them by frequency of use. This technique can also be applied to image processing. In NLP deprecated by RNN, which take into consideration the word order.

 See also [B], [Natural Language Processing], [Recurrent Neural Network], [Word2Vec]

# Bagging

 Bagging, also known as Bootstrap Aggregating. Build random sets by drawing random points from the dataset (with replacement). Train a different model on each of the sets. These models are the weak learners. The strong learner is then formed as a combination of the weak models, and the prediction is done by voting (if it is a classification model) or averaging the predictions (if it is a regression model). It is used to improve accuracy and make the model more generalize by reducing the variance, i.e., avoiding overfitting. In this, we take multiple subsets of the training dataset. For each subset, we take a model with the same learning algorithms like Decision tree, Logistic regression, etc., to predict the output for the same set of test data. Once we predict each model, we use a model averaging technique to get the final prediction output. One of the famous techniques used in Bagging is Random Forest. In the Random forest, we use multiple decision trees.

 See also [B], [Boosting], [Decision Tree], [Ensemble Method], [Random Forest], [Weak Learner]

# Balanced Fitting

 Good generalization for other data.

 ![]( {{site.assets}}/b/underfitting_overfitting_balanced.png ){: width="100%"}

 See also [B], [Overfitting], [Underfitting]

# Batch 

 When the training dataset is large, it needs to be broken into chunks called batch. Each batch is then used to build the model.

 ![]( {{site.assets}}/b/epoch_batch_iteration.png ){: width="100%"}

 See also [B], [Batch Size], [Epoch], [Iteration]

# Batch Gradient Descent

 ~ standard gradient descent. In Batch Gradient Descent, all the training data is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters. So that???s just one step of gradient descent in one epoch. Batch Gradient Descent is great for convex or relatively smooth error manifolds. In this case, we move somewhat directly towards an optimum solution.

 ![]( {{site.assets}}/b/batch_gradient_descent.png ){: width=20%}

 See also [B], [Gradient Descent], [Mini-Batch Gradient Descent]

# Batch Normalization

 Batch normalization layers can potentially resolve the vanishing gradient problem. Indeed, this problem arises when a large input space is mapped to a small one, causing the derivatives to disappear. In Image 1, this is most clearly seen at when |x| is big. Batch normalization reduces this problem by simply normalizing the input so |x| doesn???t reach the outer edges of the sigmoid function. As seen in diagram, it normalizes the input so that most of it falls in the green region, where the derivative isn???t too small.

 ![]( {{site.assets}}/b/batch_normalization.png ){: width=30%}

 See also [B], [Exploding Gradient Problem], [Sigmoid Activation Function], [Vanishing Gradient Problem]

# Batch Size

 The number of samples (rows) in a batch.

 See also [B], [Batch]


# Bayes Theorem

 Bayes theorem is used to find the reverse probabilities p(A|B) if we know the conditional probability of an event, i.e p(A) and p(B), and p(B|A)

 ```
# Probability of having feature A and B
# is equal to
# Probability of having B knowing A
# multiplied by
# probability of feature A

p(A,B) = p(B|A) * p(A)
         = p(A|B) * p(B)

# therefore
           p(B|A) * p(A)
p(A|B) = ----------------
               p(B)

Where P(A) and P(B) are the probabilities of events A and B.
P(A|B) is the probability of event A given B
P(B|A) is the probability of event B given A.
 ```

 See also [B], [Bayes Inference], [Naive Bayes]

# Bayesian Inference

 How confident are you in the result? A method of statistical learning - using a small amount of historical data and combining it with new data
 ```
          P(B|A) x P (A)
P(A|B) = ----------------
            P(B)
 ```

 See also [B], [Bayesian Network], [Bayes Theorem]

# Bayesian Network

 Bayesian networks are graphical models that use Bayesian inference to represent variables and their conditional dependencies. The goal of Bayesian networks is to model likely causation (conditional dependence), by representing these conditional dependencies as connections between nodes in a directed acyclic graph (DAG). The graph???s nodes are just the model???s variables, whether observable quantities, latent variables, unknown parameters or subjective hypotheses. Once graphed, researchers can then fairly simply calculate the probability tables for each node and find the joint probability effect of even independent, random variables on the model???s final outcome.

 ![]( {{site.assets}}/b/bayesian_network.png ){: width="100%"}

 See also [B], [Bayesian Inference]

 
# Bayesian Optimization Sampling Method

 Use ML to optimize your model. Given N samples, what would be the best next step to for my sample (given that I am looking for a local maxima) . This optimization method is an INFORMED method where the search DOES use previous results to pick the next input values to try.  `The concept is to limit evals of the objective function * which is time consuming/expensive * by spending more time choosing the next values to try.` (Think dichotomy, + awareness of correlation between parameters, etc? ==> `from which next sample will I learn the most?`)

 Beware:
  * To use when
   * getting a sample is expensive ==> smart sampling required!
   * observations are noisy (?)
   * function is black box, with no closed form or gradient (?)
  * you are looking for a minima and do not care about the distribution (?)

 More
  * [https://scikit-optimize.github.io/notebooks/hyperparameter-optimization.html](https://scikit-optimize.github.io/notebooks/hyperparameter-optimization.html)
  * [https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0)

 See also [B], [Active Learning], [Grid Search], [Hyperparameter], [Random Search], [Surrogate Model]

# Bayesian Search

 Searching for a value using the bayesian optimization sampling method. See also [B], [Bayesian Optimization Sampling Method], [Hyperparameter Tuning]

# Behavioural Cloning

 ~ Trying to duplicate the behavior of an expect. Early approaches to imitation learning seek to learn a policy as a machine learning model that maps environment observations to (optimal) actions taken by the expert using supervised learning. The method is called Behavioral Cloning (BC), but it has a drawback: BC has loose, or no, guarantees that the model will generalize to unseen environmental observations. A key issue is that when the agent ends up in an situation that is unlike any of the expert trajectories, BC is prone to failures.

 ![]( {{site.assets}}/b/behavioural_cloning.png ){: width="100%"}

 For example, in the figure above, the car agent doesn???t know what to do if it goes away from the expert trajectory and it crashes. To avoid making a mistake, BC requires expert data on all possible trajectories in the environment, making it a heavily data-inefficient approach.

 See also [B], [Imitation Learning]

# Bernoulli Distribution

 the discrete probability distribution of a random variable which takes the value 1 with probability P and the value 0 with probability Q=1-P. Less formally, it can be thought of as a model for the set of possible outcomes of any single experiment that asks a yes???no question. Such questions lead to outcomes that are boolean-valued: a single bit whose value is success/yes/true/one with probability p and failure/no/false/zero with probability Q. It can be used to represent a (possibly biased) coin toss where 1 and 0 would represent "heads" and "tails", respectively, and P would be the probability of the coin landing on heads (or vice versa where 1 would represent tails and P would be the probability of tails). In particular, unfair coins would have P =/= 1/2. The Bernoulli distribution is a special case of the binomial distribution where a single trial is conducted (so n would be 1 for such a binomial distribution). It is also a special case of the two-point distribution, for which the possible outcomes need not be 0 and 1.

 More at:
   * [https://en.wikipedia.org/wiki/Bernoulli_distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)

# BERT Classification


 ![]( {{site.assets}}/b/bert_classification.png ){: width="100%"}

  * E = word embeddings ?     <=== Maybe wrong use of embedding, rather token, i.e. tokenized?
    * Embedding is an integer from a tokenizer? No!
    * Sparse vector (of tokenised sentence) fed to word2vec (or similar) ? No!
  * R = representation ( token after transformation by encoder stack )
    * Representation is a matrix/tensor (of square dimension 768?)
    * "What is the cost?" and "Is it expensive?" have almost the same SEP_representation !
 See also [B], [Bidirectional Encoder Representations from Transformers]

# Bias

  1. statistics ==> The gap between the prediction and the actual value. Where is bias coming from? Issues with the data sampling?
  1. data sample ==> data that is used for learning is biased, ex: all nurse are female ==> implies unwanted correlation in data
  1. neural network learning ==> When using bias in the connect of activation function, it is an integer that represent a threshold the weighted input should exceed to trigger the neuron. There is a bias at each node of the ANN. The node weighted input is = sum(aL . wL) + bias.

 See also [B], [Activation Function], [Balanced Fitting], [Bias-Variance Trade-Off], [Overfitting], [Underfitting], [Variance]

# Bias-Variance Trade-Off

 Ideally, a model will have both low bias and variance, but efforts to decrease one will frequently increase the other. This is known as the bias-variance trade-off.

# Bidirectional Encoder Representation from Transformer Model

# BERT Model


 A NLP model that was built by Google in 2017. It is an Open-Source project by Google AI researchers with a great power of understanding the context of sentence (language) showing high performance in various nlp tasks such as question-answer system , Named-entity-recognition, Machine Translation and many more.
  * BIDIRECTIONAL = use words before and after the [MASK] to predict the Masked word. This is different from unidirectional (used by GPT) such as predicting what the next word is.
  * Can be extended, i.e. FinBERT.
 Trained using
  * Masked Language Modeling (MLM)     <== pre-train work embedding and contextual understanding using [MASK]
  * and next sentence prediction (NSP).  <== pre-train the [CLS] token (used to perform sequence/sentence-wide task)
    * Note that the representation of the [CLS] token include both the sentences, the one before and the one after the [SEP] token (separation token) (?)

 {% pdf {{site.assets}}/b/bert_paper.pdf %}

 ![]( {{site.assets}}/b/bert_embeddings.png ){: width=35%}

 More at:
   * embeddings (token + segment + position) - [https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a](https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a)
   * [https://medium.com/@mromerocalvo/6dcf5360b07f](https://medium.com/@mromerocalvo/6dcf5360b07f)
   * [https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)
   * [https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

 See also [B], [Attention Score], [Attention-Based Model], [BERT Classification], [Masked Language Modeling], [Name Entity Recognition], [Next Sentence Prediction], [Tokenizer], [Transformer Model]


# Bidirectional Recurrent Neural Network

# BRNN

 Bidirectional recurrent neural networks (BRNN) connect two hidden layers running in opposite directions to a single output, allowing them to receive information from both past and future states. This generative deep learning technique is more common in supervised learning approaches, rather than unsupervised or semi-supervised because how difficult it is to calculate a reliable probabilistic model.

 ![]( {{site.assets}}/b/bidirectional_recurrent_neural_network.png ){: width=30%}

 See also [B], [Recurrent Neural Network]

# Big Data 

 See also [B], [Deep Learning], [Expert System], [Machine Learning]

# Bilingual Evaluation Understudy Score

# BLEU Score

 This is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" ??? this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics. Scores are calculated for individual translated segments???generally sentences???by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness are not taken into account. BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional reference translations will increase the BLEU score.

 See also [B], [NLP Benchmark]

# Binary Classification

 `Answer a question with Yes or No with a confidence level`. Ex: is this shape a square? The simplest case of classification algorithm. In the case of the support-vector-machine, binary classification can be done with the creation of a hyperplane as a decision boundary in a real, transformed, or latent space.

 See also [B], [Binary Cross-Entropy Loss Function], [Classification], [Multiclass Classification], [Support Vector Machine]

# Binary Cross-Entropy Loss Function

 If you are training a binary classifier, chances are you are using binary cross-entropy / log loss as your loss function.
 ```
cross-entropy loss = c = sum(0, n, Pi * log (1/Qi)

# And in the case of binary classification problem where we have only two classes, we name it as binary cross-entropy loss and above formula becomes:
binary cross-entropy loss = c = sum(0, 1, Pi * log (1/Qi) = Po * log(1/Qo) + (1-Po) * log(1/Q1)
 ```

 More at :
   * [https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

 See also [B], [Binary Classifier], [Cross-Entropy Loss Function], [Entropy], [Loss Function]

# BLIP Model

 Create a a caption for an image using an encoder-decoder model (unlike the CLIP model, does not use the same embedding space?).

 See also [B], [CLIP Model], [Image Reconstruction], [Multimodal Translation], [Text Reconstruction]

# Boltzmann Machine

 an unsupervised DL model in which every node is connected to every other node. That is, unlike the ANNs, CNNs, RNNs and SOMs, the Boltzmann Machines are undirected (or the connections are bidirectional). Boltzmann Machine is not a deterministic DL model but a stochastic or generative DL model. It is rather a representation of a certain system. There are two types of nodes in the Boltzmann Machine ??? Visible nodes ??? those nodes which we can and do measure, and the Hidden nodes ??? those nodes which we cannot or do not measure. Although the node types are different, the Boltzmann machine considers them as the same and everything works as one single system. The training data is fed into the Boltzmann Machine and the weights of the system are adjusted accordingly. Boltzmann machines help us understand abnormalities by learning about the working of the system in normal conditions.

 ![]( {{site.assets}}/b/boltzmann_machine.jpeg ){: width="100%"}

 More at:
   * [https://www.geeksforgeeks.org/types-of-boltzmann-machines/](https://www.geeksforgeeks.org/types-of-boltzmann-machines/)

 See also [B], [Deep Belief Network], [Restricted Boltzmann Machine], [Unsupervised Deep Learning Model], [Unsupervised Learning]

# Boosting

 Sequentially combine weak predictors (such as decision trees) to get a strong predictor! Start by training a random model, which is the first weak learner. Evaluate it on the entire dataset. Shrink the points that have good predictions, and enlarge the points that have poor predictions. Train a second weak learner on this modified dataset. We continue in this fashion until we build several models. The way to combine them into a strong learner is the same way as with bagging, namely, by voting or by averaging the predictions of the weak learner. More specifically, if the learners are classifiers, the strong learner predicts the most common class predicted by the weak learners (thus the term voting), and if there are ties, by choosing randomly among them. If the learners are regressors, the strong learner predicts the average of the predictions given by the weak learners. Boosting is primarily used to reduce the bias and variance in a supervised learning technique. It refers to the family of an algorithm that converts weak learners (base learner) to strong learners. The weak learner is the classifiers that are correct only up to a small extent with the actual classification, while the strong learners are the classifiers that are well correlated with the actual classification.

 Few famous techniques of Boosting are:
  * !AdaBoost,
  * Gradient boosting
  * XgBOOST (Extreme Gradient Boosting).

 So now we know what bagging and boosting are and what are their roles in Machine Learning.

 See also [B], [AdaBoost], [Bagging], [Boosting Step Size], [Gradient Boosting], [Weak Learner], [XGBoost]

# Boosting Step Size


 See also [B], [Boosting], [Hyperparameter]

# Bootstrap Sampling Method

 A large number of samples are drawn randomly with replacement from the original dataset, and the model is trained and tested on these samples. This method is used to estimate the variability of a model's performance and the uncertainty of its predictions. The main concept behind bootstrap sampling is train the same model multiple times on multiple samples taken with replacement from the target population. Bootstrapping is the most popular resampling method today. It uses sampling with replacement to estimate the sampling distribution for a desired estimator. The main purpose for this particular method is to evaluate the variance of an estimator. It does have many other applications, including:
  * Estimating confidence intervals and standard errors for the estimator (e.g. the standard error for the mean),
  * Estimating precision for an estimator ??,
  * Dealing with non-normally distributed data,
  * Calculating sample sizes for experiments.
 Bootstrapping has been shown to be an excellent method to estimate many distributions for statistics, sometimes giving better results than traditional normal approximation. It also works well with small samples. It doesn???t perform very well when the model isn???t smooth, is not a good choice for dependent data, missing data, censoring, or data with outliers.

 More at:
  * [https://www.datasciencecentral.com/resampling-methods-comparison/](https://www.datasciencecentral.com/resampling-methods-comparison/)

 See also [B], [Resampling Method]

# Boston Dynamics Company

 More at:
  * website - [https://www.bostondynamics.com/](https://www.bostondynamics.com/)

 See also [B], [Company]
