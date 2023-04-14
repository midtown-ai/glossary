---
title: K
permalink: /k/

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


# K-Fold Cross Validation

 Used in 2 cases:
  1. to evaluate a model's performance by estimating its [generalized RMSE]
    * :warning: The generalized RMSE is computed by averaging its sampled RMSE on sample batches?
  1. to compare models during  [hyperparameter optimization]
    * To obviously find the hyperparameters that leads to the best model with the best RMSE

 K-fold is not specific to the model's algorithm. But for k-fold to be interesting you need to have a lot of data, not just a little so the folds are big enough!

 The general process of k-fold cross-validation for evaluating a model’s performance is:
  * The whole dataset is randomly split into independent k-folds without replacement.
  * k-1 folds are used for the model training and one fold is used for performance evaluation.
  * This procedure is repeated k times (iterations) so that we obtain k number of performance estimates (e.g. MSE) for each iteration.
  * Then we get the mean of k number of performance estimates (e.g. MSE).

 ![]( {{site.assets}}/k/k_fold_cross_validation_model_performance.webp ){: width="100%"}

 Remarks:
  * The splitting process is done without replacement. So, each observation will be used for training and validation exactly once.
  * Good standard values for k in k-fold cross-validation are 5 and 10. However, the value of k depends on the size of the dataset. For small datasets, we can use higher values for k. However, larger values of k will also increase the runtime of the cross-validation algorithm and the computational cost.
  * When k=5, 20% of the test set is held back each time. When k=10, 10% of the test set is held back each time and so on…
  * A special case of k-fold cross-validation is the [Leave-one-out cross-validation (LOOCV)][LOOCV] method in which we set k=n (number of observations in the dataset). Only one training sample is used for testing during each iteration. This method is very useful when working with very small datasets.

 Using k-fold cross-validation for hyperparameter tuning

 Using k-fold cross-validation in combination with [grid search] is a very useful strategy to improve the performance of a machine learning model by tuning the model hyperparameters.

 In grid search, we can set values for multiple hyperparameters. Let’s say we have two hyperparameters each having three different values. So, we have 9 (3 x 3) different combinations of hyperparameter values. The space in which all those combinations contain is called the hyperparameter space. When there are two hyperparameters, space is two dimensional.

 In grid search, the algorithm takes one combination of hyperparameter values at a time from the hyperparameter space that we have specified. Then it trains the model using those hyperparameter values and evaluates it through k-fold cross-validation. It stores the performance estimate (e.g. MSE). Then the algorithm takes another combination of hyperparameter values and does the same. After taking all the combinations, the algorithm stores all the performance estimates. Out of those estimates, it selects the best one. The combination of hyperparameter values that yields the best performance estimate is the optimal combination of hyperparameter values. The best model includes those hyperparameter values.

 ![]( {{site.assets}}/k/k_fold_cross_validation_hyperparameter_tuning.webp ){: width="100%"}

 The reason behind fitting the best model to the whole training set after k-fold cross-validation is to provide more training samples to the learning algorithm of the best model. This usually results in a more accurate and robust model.

 More at:
  * [https://towardsdatascience.com/k-fold-cross-validation-explained-in-plain-english-659e33c0bc0](https://towardsdatascience.com/k-fold-cross-validation-explained-in-plain-english-659e33c0bc0)
  * [https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f](https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f)
  * [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)


 See also [K], ...


# K-Mean Algorithm

 Recurrence algorithm. Give a number of clusters (eg 3), then you try to group your samples to the 3 clusters. A cluster is defined by its cluster centre. For each sample , measure the distance to the centre-1, centre-2, centre-3 if 3 clusters. The point/sample belongs to the cluster whose centre is the closest. then move the centre to the place where those that belongs to the cluster error is minimum (i.e. move the centre of the cluster to a new location). Then recur with the new position for the cluster centres. At some point, the cluster centres will not move and the recurrence will therefore stop.

 See also [K], [K-Mean Cluster], [K-Mean Failure]


# K-Mean Cluster

  When using K-Mean algorithm, how to find the number of clusters? Do this iteratively (2, 3, ...) and for each scenario plot the sum of the squared error. Then look at the elbow, where the squared error drops significantly. Where that happens, you have found the number of clusters. Increasing the number of clusters beyond that number only has a marginal effect.

  See also [K], [K-Mean Algorithm]


# K-Mean Failure

 There are scenarios in which k-means fails. Here the 2 groups of samples are not correctly identified. In that particular case, k-mean does not work correctly.

 ![]( {{site.assets}}/k/kmean_failure.png ){: width="100%"}

 A possible solution is DBSCAN.

 See also [K], [DBSCAN], [K-Mean Algorithm]


# K-Nearest Neighbor

# KNN

 KNN can be used for both classification and regression predictive problems. However, it is more widely used in classification problems in the industry. It is commonly used for its easy of interpretation and low calculation time. In the example below, K=3 and given that the 3 nearest neighbor are in the same class, we are fairly confident, the new sample is in the same class.

 ![]( {{site.assets}}/k/k_nearest_neighbor.png ){: width="100%"}

 More at:
  * [https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)

 See also [K], [Model Evaluation], [Semisupervised Learning]


# Keras

 A python module ...

 See also [K], ...


# Kernel Trick

 ![]( {{site.assets}}/k/kernel_trick_table.png ){: width="100%"}
 ![]( {{site.assets}}/k/kernel_trick_curve.png ){: width="100%"}

 More at:
  * [https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace](https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace)
  * video - [https://www.youtube.com/watch?v=IpGxLWOIZy4](https://www.youtube.com/watch?v=IpGxLWOIZy4)

 See also [K], [Support Vector Machine]


# Knowledge

 ```
 Data < Information < Knowledge
 ```

 See also [K], [Data], [Information]


# Knowledge Corpus

 Queried right after LLM prompt by a [Neural Retriever] for the purpose of [Knowledge Retrieval]

 See also [K], ...


# Knowledge Graph

# KG

 A knowledge graph is a directed labeled graph in which we have associated domain specific meanings with nodes and edges. Anything can act as a node, for example, people, company, computer, etc. An edge label captures the relationship of interest between the nodes, for example, a friendship relationship between two people, a customer relationship between a company and person, or a network connection between two computers, etc. The directed labeled graph representation is used in a variety of ways depending on the needs of an application. A directed labeled graph such as the one in which the nodes are people, and the edges capture the parent relationship is also known as a data graph. A directed labeled graph in which the nodes are classes of objects (e.g., Book, Textbook, etc.), and the edges capture the subclass relationship, is also known as a taxonomy. In some data models, given a triple (A,B,C), we refer to A, B, C as the subject, the predicate, and the object of the triple respectively. A knowledge graph serves as a data structure in which an application stores information. The information could be added to the knowledge graph through a combination of human input, automated and semi-automated methods. Regardless of the method of knowledge entry, it is expected that the recorded information can be easily understood and verified by humans. Many interesting computations over a graph can be reduced to navigating it. For example, in a friendship KG, to calculate the friends of friends of a person A, we can navigate the graph from A to all nodes B connected to it by a relation labeled as friend, and then recursively to all nodes C connected by the friend relation to each B.

 More at:
  * [https://ai.stanford.edu/blog/introduction-to-knowledge-graphs/](https://ai.stanford.edu/blog/introduction-to-knowledge-graphs/)

 See also [K], [Graph Neural Network]


# Knowledge Representation

 To store what a computer knows or hears!

 See also [K], ...


# Knowledge Retrieval

 A solution to the [hallucinations][Hallucination] of [Large Language Model] ?

 Possible thanks to a [Information Retriever] that fronts the model
  * ...
  * [Neural Retriever]

 More at:
  * [https://venturebeat.com/ai/whats-next-in-large-language-model-llm-research-heres-whats-coming-down-the-ml-pike/](https://venturebeat.com/ai/whats-next-in-large-language-model-llm-research-heres-whats-coming-down-the-ml-pike/)

 See also [K], ...


# Kullback-Liebler Distance

 See [Kullback-Liebler Divergence]


# Kullback-Leibler Divergence

# KL Divergence

 calculates a score that measures the divergence of one probability distribution from another. The Kullback–Leibler divergence (KL divergence), aka relative entropy, is the difference between cross-entropy of two distributions and their own entropy. For everyone else, imagine drawing out the two distributions, and wherever they do not overlap will be an area proportional to the KL divergence. Appears in the loss function of a variational autoencoder! This term stay close to normal(mean=0,stdev=1) !!

 More at:
  * [https://www.youtube.com/watch?v=rZufA635dq4&t=1091s](https://www.youtube.com/watch?v=rZufA635dq4&t=1091s)
  * [https://machinelearningmastery.com/divergence-between-probability-distributions/](https://machinelearningmastery.com/divergence-between-probability-distributions/)

 See also [K], [Cross-Entropy Loss Function], [Disentangled Variational Autoencoder], [Entropy], [Variational Autoencoder]
