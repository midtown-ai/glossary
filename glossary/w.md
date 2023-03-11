---
title: W
permalink: /w/

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

 
# Wave2Vec

 Used for speech.

 See also [W], ...

 
# Weak AI

 Searle identified a philosophical position he calls "strong AI":
  * The appropriately programmed computer with the right inputs and outputs would thereby have a mind in exactly the same sense human beings have minds.[b]
 The definition depends on the distinction between simulating a mind and actually having a mind. Searle writes that "according to Strong AI, the correct simulation really is a mind. According to Weak AI, the correct simulation is a model of the mind."

 More at:
  * [https://en.wikipedia.org/wiki/Chinese_room#Strong_AI](https://en.wikipedia.org/wiki/Chinese_room#Strong_AI)

 See also [W], [Strong AI]

 
# Weak Learner

 anything just better than random guessing! That is basically the only requirement for a weak learner. So long as you can consistently beat random guessing, any true boosting algorithm will be able to increase the accuracy of the final ensemble. What weak learner you should choose is then a trade off between 3 factors:
  1. The bias of the model. A lower bias is almost always better, but you don't want to pick something that will overfit (yes, boosting can and does overfit)
  1. The training time for the weak learner. Generally we want to be able to learn a weak learner quickly, as we are going to be building a few hundred (or thousand) of them.
  1. The prediction time for our weak learner. If we use a model that has a slow prediction rate, our ensemble of them is going to be a few hundred times slower!
 The classic weak learner is a decision tree. By changing the maximum depth of the tree, you can control all 3 factors. This makes them incredibly popular for boosting. What you should be using depends on your individual problem, but decision trees is a good starting point.

 See also [W], [Decision Tree], [Gradient Bagging], [Gradient Boosting]

 
# Weak-Supervised Learning

 Data augmentation of supervised learning = augment data when the provided labeled data is small (can be augmented!)!

 See also [W], [Learning Method]

 
# Weak Supervision Labeling Function

 ~ greatly augment data with supervised data.

 See also [W], [Snorkel Program], [Weak-Supervised Learning]

 
# Weighted Gini Impurity Index

 See also [W], [Gini Impurity Index], [Forest Of Stumps]


# Whisper Model

 Open-sourcing a neural net called Whisper that approaches human level robustness and accuracy on English speech recognition.

 More at:
  * [https://openai.com/blog/whisper/](https://openai.com/blog/whisper/)

 See also [W], [O], [OpenAI Company]

 
# Word Embedding

 ~ `Take a sparse vector as input to a word2vec process and turn into a point in the embedding space, where 2 close related words (in meaning) are close (at a small euclidian distance)`. TFM and TFIDF are numerical representations of text documents that only consider frequency and weighted frequencies to represent text documents. By contrast, word embeddings can capture the context of a word in a document (e.g. "bank" in bank account and river bank have different embeddings). With the word context, embeddings can quantify the similarity between words, which in turn allows us to do arithmetic with words. Word2Vec is a method based on neural nets that maps words in a corpus to a numerical vector. We can then use these vectors to find synonyms, perform arithmetic operations with words, or to represent text documents (by taking the mean of all the word vectors in a document). For example, let’s assume that we use a sufficiently big corpus of text documents to estimate word embeddings. Let’s also assume that the words king, queen, man and woman are part of the corpus. Let say that vector(‘word’) is the numerical vector that represents the word ‘word’. To estimate vector(‘woman’), we can perform the arithmetic operation with vectors:
 
 ```
vector(‘king’) + vector(‘woman’) — vector(‘man’) ~ vector(‘queen’)
 ```

 Word representations allow finding similarities between words by computing the cosine similarity between the vector representation of two words. The cosine similarity measures the angle between two vectors. We compute word embeddings using machine learning methods, but that’s often a pre-step to applying a machine learning algorithm on top. For instance, suppose we have access to the tweets of several thousand Twitter users. Also suppose that we know which of these Twitter users bought a house. To predict the probability of a new Twitter user buying a house, we can combine Word2Vec with a logistic regression. You can train word embeddings yourself or get a pre-trained (transfer learning) set of word vectors. To download pre-trained word vectors in 157 different languages, take a look at !FastText.

 {% youtube "https://www.youtube.com/watch?v=bof9EdygMSo" %}

 More at:
  * [https://towardsdatascience.com/introduction-to-word-embeddings-4cf857b12edc](https://towardsdatascience.com/introduction-to-word-embeddings-4cf857b12edc)

 See also [W], [NLP], [Word2Vec]

 
# Word Embedding Space

 In natural language processing, word embeddings are numerical representations of words so that similar words have close representations. So, word embeddings lie in a latent space where every word is encoded into a low-dimensional semantic vector. There are many algorithms for learning word embeddings like Word2Vec or GloVe (which are both context-free). Other more advanced models are Contextual models, which  instead generate a representation of each word that is based on the other words in the sentence (e.g. "bank" in bank account and river bank have different embeddings) . In the image below, we can see an illustration of the topology of the word embeddings in the latent space:

 ![]( {{site.assets}}/w/word_embedding_space.png ){: width="100%"}

 As expected, semantically similar words like the word ‘toilet’ and the word ‘bathroom’ have close word embeddings in the latent space.

 More at:
  * [https://towardsdatascience.com/introduction-to-word-embeddings-4cf857b12edc](https://towardsdatascience.com/introduction-to-word-embeddings-4cf857b12edc)

 See also [W], [Input Space], [Latent Space]

 
# Word2Vec

 Context-free models such as word2vec or GloVe generate a single "word embedding" representation for each word in the vocabulary, so bank would have the same representation in bank deposit and river bank. Do not take into consideration the context on the right or on the left of the word. ~ Bag of Words. Deprecated by RNN?

 See also [W], [Bag Of Words], [Recurrent Neural Network], [Word Embedding]

 
# WordNet Dataset

 WordNet® is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations. The resulting network of meaningfully related words and concepts can be navigated with the browser. WordNet is also freely and publicly available for download. WordNet's structure makes it a useful tool for computational linguistics and natural language processing.

 More at:
  * [https://wordnet.princeton.edu/](https://wordnet.princeton.edu/)

 See also [W], [Dataset], [ImageNet Dataset], [Transfer Learning]
