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

# Kernel Filter

 A small matrix which you can use to multiply to the pixel area of the same size. The Kernel is applied to the same image for every area possible. AS represented below, we can use a kernel of 2x2, but we recommend a 3x3 kernel.

 ![]( {{site.assets}}/k/kernel_filter.gif ){: width="100%"}

 Where is the 'design' of the kernels coming from? Just like weights in a neural network, it comes from Backpropagation !

 More at:
  * [https://medium.com/codex/kernels-filters-in-convolutional-neural-network-cnn-lets-talk-about-them-ee4e94f3319](https://medium.com/codex/kernels-filters-in-convolutional-neural-network-cnn-lets-talk-about-them-ee4e94f3319)

 See also [K], [Convolutional Layer], [Convolutional Neural Network]

# Kernel Trick

 ![]( {{site.assets}}/k/kernel_trick_table.png ){: width="100%"}
 ![]( {{site.assets}}/k/kernel_trick_curve.png ){: width="100%"}

 More at:
  * [https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace](https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace)
  * video - [https://www.youtube.com/watch?v=IpGxLWOIZy4](https://www.youtube.com/watch?v=IpGxLWOIZy4)

 See also [K], [Support Vector Machine]

# Knowledge Graph

# KG

 A knowledge graph is a directed labeled graph in which we have associated domain specific meanings with nodes and edges. Anything can act as a node, for example, people, company, computer, etc. An edge label captures the relationship of interest between the nodes, for example, a friendship relationship between two people, a customer relationship between a company and person, or a network connection between two computers, etc. The directed labeled graph representation is used in a variety of ways depending on the needs of an application. A directed labeled graph such as the one in which the nodes are people, and the edges capture the parent relationship is also known as a data graph. A directed labeled graph in which the nodes are classes of objects (e.g., Book, Textbook, etc.), and the edges capture the subclass relationship, is also known as a taxonomy. In some data models, given a triple (A,B,C), we refer to A, B, C as the subject, the predicate, and the object of the triple respectively. A knowledge graph serves as a data structure in which an application stores information. The information could be added to the knowledge graph through a combination of human input, automated and semi-automated methods. Regardless of the method of knowledge entry, it is expected that the recorded information can be easily understood and verified by humans. Many interesting computations over a graph can be reduced to navigating it. For example, in a friendship KG, to calculate the friends of friends of a person A, we can navigate the graph from A to all nodes B connected to it by a relation labeled as friend, and then recursively to all nodes C connected by the friend relation to each B.

 More at:
  * [https://ai.stanford.edu/blog/introduction-to-knowledge-graphs/](https://ai.stanford.edu/blog/introduction-to-knowledge-graphs/)

 See also [K], [Graph Neural Network]

# Kullback-Liebler Distance

 See [Kullback-Liebler Divergence]

# Kullback-Leibler Divergence

# KL Divergence

 calculates a score that measures the divergence of one probability distribution from another. The Kullback–Leibler divergence (KL divergence), aka relative entropy, is the difference between cross-entropy of two distributions and their own entropy. For everyone else, imagine drawing out the two distributions, and wherever they do not overlap will be an area proportional to the KL divergence. Appears in the loss function of a variational autoencoder! This term stay close to normal(mean=0,stdev=1) !!

 More at:
  * [https://www.youtube.com/watch?v=rZufA635dq4&t=1091s](https://www.youtube.com/watch?v=rZufA635dq4&t=1091s)
  * [https://machinelearningmastery.com/divergence-between-probability-distributions/](https://machinelearningmastery.com/divergence-between-probability-distributions/)

 See also [K], [Cross-Entropy Loss Function], [Disentangled Variational Autoencoder], [Entropy], [Variational Autoencoder]
