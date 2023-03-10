---
title: H
permalink: /h/

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

# Hebbian Learning

 "Neurons that fire together, wire together" Linked to learning and neural mechanism. Repeated or persistent firing changes synaptic weight due to increased efficiency. Synaptic modifications can be hebbian, anti-hebbian, or non-hebbian.

 More at :
  * [https://anthony-tan.com/Supervised-Hebbian-Learning/](https://anthony-tan.com/Supervised-Hebbian-Learning/)

 See also [H], 
 
# Hidden Layer

 See also [H], [Neural Network]

# Hidden Markov Model

# HMM

 More at:
  * [https://en.wikipedia.org/wiki/Hidden_Markov_model ](https://en.wikipedia.org/wiki/Hidden_Markov_model )

 See also [H], [Generative Classified], [Hidden State]


# Hidden State

  * Memory from previous RNN stages = representation of previous input
  * In Encoder-decoder = latent state/space?

 See also [H], [H], [Encoder-Decoder Model], [Hidden Markov Model], [Recurrent Neural Network]

# Hinge Loss Function

 The use of hinge loss is very common in binary classification problems where we want to separate a group of data points from those from another group. It also leads to a powerful machine learning algorithm called Support Vector Machines (SVMs) Let’s have a look at the mathematical definition of this function.

 More at:
  * [https://www.baeldung.com/cs/hinge-loss-vs-logistic-loss](https://www.baeldung.com/cs/hinge-loss-vs-logistic-loss)

 See also [H], [Loss Function], [Support Vector Machine], 

# Holdout Fold

 See also [H], [Cross-Validation Sampling Method]

# Huber Loss Function

 Now we know that the Mean Square Error (MSE) is great for learning outliers while the Mean Absolute Error (MAE) is great for ignoring them. But what about something in the middle? Consider an example where we have a dataset of 100 values we would like our model to be trained to predict. Out of all that data, 25% of the expected values are 5 while the other 75% are 10. An MSE loss wouldn’t quite do the trick, since we don’t really have “outliers”; 25% is by no means a small fraction. On the other hand we don’t necessarily want to weight that 25% too low with an MAE. Those values of 5 aren’t close to the median (10 — since 75% of the points have a value of 10), but they’re also not really outliers. Our solution? The Huber Loss Function. The Huber Loss offers the best of both worlds by balancing the MSE and MAE together. We can define it using the following piecewise function:

 ![]( {{site.assets}}/h/huber_loss_function_formula.png ){: width="100%"}

 ![]( {{site.assets}}/h/huber_loss_function_graph.png ){: width="100%"}

 What this equation essentially says is: for loss values less than delta, use the MSE; for loss values greater than delta, use the MAE. This effectively combines the best of both worlds from the two loss functions! Using the MAE for larger loss values mitigates the weight that we put on outliers so that we still get a well-rounded model. At the same time we use the MSE for the smaller loss values to maintain a quadratic function near the centre. This has the effect of magnifying the loss values as long as they are greater than 1. Once the loss for those data points dips below 1, the quadratic function down-weights them to focus the training on the higher-error data points.

 ```
import numpy as np

def huber_loss(y_pred, y, delta=1.0):
    huber_mse = 0.5*(y-y_pred)**2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)
 ```

 Pros and Cons:
  * Advantages : Best of the MSE and the MAE ?
  * Disadvantages
   * For cases where outliers are very important to you, use the MSE! 
   * For cases where you don’t care at all about the outliers, use the MAE!

 See also [H], [Loss Function], [Mean Absolute Error Loss Function], [Mean Square Error Loss Function]

# Human-Centered AI

  * Understand human intelligence more deeply and more broadly
  * Connect to neuroscience, cognitive psychology, etc
  * Collaborates with humans
  * Enhances, not replaces humans; gives humans appropriate control
  * Aware of human preferences (value discovery/alignment)
  * Aware of human abilities and limitations
  * Accountable, explainable, understandable, and trustworthy
  * Focused on what is good for humanity (health, environment)
  * Bridges to policy world, other academic disciplines, industry
  * Respects ethics (animal-centered AI? Earth-centered AI?) 

 More at:
  * stanford HAI - [https://hai.stanford.edu/](https://hai.stanford.edu/)

 See also [H], [Artificial Intelligence]

# Hyperparameter

 `~ parameters to tune the performance of the ML model`. Any decision the algorithm author can't make for you. In machine learning, we use the term hyperparameter to distinguish from standard model parameters. So, it is worth to first understand what those are. A machine learning model is the definition of a mathematical formula with a number of parameters that need to be learned from the data. That is the crux of machine learning: fitting a model to the data. This is done through a process known as model training. In other words, by training a model with existing data, we are able to fit the model parameters. `However, there is another kind of parameters that cannot be directly learned from the regular training process`. These parameters express “higher-level” properties of the model such as its complexity or how fast it should learn. They are called hyperparameters. Hyperparameters are usually fixed before the actual training process begins. So, how are hyperparameters decided? That is probably beyond the scope of this question, but suffice to say that, broadly speaking, this is done by setting different values for those hyperparameters, training different models, and deciding which ones work best by testing them.

 So, to summarize. Hyperparameters:
  * Define higher level concepts about the model such as complexity, or capacity to learn.
  * Cannot be learned directly from the data in the standard model training process and need to be predefined.
  * Can be decided by setting different values, training different models, and choosing the values that test better

 Some examples of hyperparameters:
  * Number of leaves or depth of a tree
  * Number of trees
  * Number of latent factors in a matrix factorization
  * Learning rate (in many models)
  * Number of hidden layers in a deep neural network
  * Number of hidden nodes in network layers
  * Number of clusters in a k-means clustering
  * Drop out
  * Regularization
  * Boosting step size
  * Initialization of clustering algorithm

 Beware often you have 1 hyperparameter that is more impactful than the other. Also beware of correlation between parameters. Hyperparameters can take a continuous, integer, or categorical value (ex learning rate 0.1, epochs:20, optimizer: sgd). 

 See also [H], [Algorithmic], [Boosting Step Size], [Complexity], [Data Handling], [Drop Out], [Expressiveness], [Hyperparameter Optimization], [Learning Rate], [Parameter], [Regularization], [XGBoost]


# Hyperparameter Optimization

# HPO

 Tuning strategies:
  * (1) Trial and Error, defaults, guess, experience, intuition, heuristics 
  * (2) Try everything! Random, Grid search, Sobol :warning: High dimensional grid search, = the curse of dimensionality
  * (3) Meta model ... Required to avoid over-fitting and under-fitting.

 See also [H], [CPU], [Curse of Dimensionality], [F1 Score], [GPU], [Grid Search], [Hyperparameter], [Meta Model], [Overfitting], [Random Search], [Sobol], [Underfitting]

# Hyperparameter Tuning

 Same as HPO? Process to tune the hyperparameters to get the best prediction? See also [H], [AutoML]

# Hyperplane

 The boundary between TWO classification classes? Yes, in a real or latent dimension! For example in a 3D space, a 2-D plane could be an hyperplane where on one side you have the elements of class A and on the other side you have the elements of class B. Used as a decision boundary. A hyperplane is a subspace that has one dimension less than the ambient space that contains it. In simple linear regression, there is one dimension for the response variable and another dimension for the explanatory variable, for a total of two dimensions. The regression hyperplane thus has one dimension; a hyperplane with one dimension is a line. In mathematics, a hyperplane H is a linear subspace of a vector space V such that the basis of H has cardinality one less than the cardinality of the basis for V.  In other words, if V is an n-dimensional vector space than H is an (n-1)-dimensional subspace.  Examples of hyperplanes in 2 dimensions are any straight line through the origin. In 3 dimensions, any plane containing the origin.  In higher dimensions, it is useful to think of a hyperplane as member of an affine family of (n-1)-dimensional subspaces (affine spaces look and behavior very similar to linear spaces but they are not required to contain the origin), such that the entire space is partitioned into these affine subspaces. This family will be stacked along the unique vector (up to sign) that is perpendicular to the original hyperplane.  This "visualization" allows one to easily understand that a hyperplane always divides the parent vector space into two regions.

 ![]( {{site.assets}}/h/hyperplane.png ){: width="100%"}

 See also [H], [Classification], [Decision Boundary], [Support Vector Machine]
