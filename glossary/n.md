---
title: N
permalink: /n/

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


# N-Gram

 An N-Gram is a connected string of N items from a sample of text or speech. The N-Gram could be comprised of large blocks of words, or smaller sets of syllables. N-Grams are used as the basis for functioning N-Gram models, which are instrumental in natural language processing as a way of predicting upcoming text or speech.

 ![]( {{site.assets}}/n/ngram.png ){: width="100%"}

 N-Gram models are uses in natural language processing as a tool for modeling probable upcoming sequences of characters, also known as trigrams or 3-grams. An example is the phrase, "Good Afternoon," which breaks down to the trigrams "Goo","d A", "fte", etc. In machine translation models, however, N-Gram models are usually used in conjunction with Bayesian inference, leading to a more accurate prediction.

 See also [N], [N-Gram Model], [Natural Language Programming]


# N-Gram Model

 See also [N], [N-Gram]


# Naive Bayes

 uses the Bayes’ Theorem and assumes that all predictors are independent. In other words, this classifier assumes that the presence of one particular feature in a class doesn’t affect the presence of another one.
 
 ```
p(X1,X2) = p(X2 | X1) * p(X1)
In Naive Bayes, p(X2) is independent from p(X1), in other words p(X2|X1) = p(X2).
Therefore
p(X1,X2) = p(X2) * p(X1)

If we generalize
p(X1,X2, ..., Xn) = p(X1) * p(X2) * ... * p(Xn)
 ```

 See also [N], [Bayes Theorem], [Feature]


# Name Entity Recognition

 A standard NLP problem which involves spotting named entities (people, places, organizations etc.) from a chunk of text, and classifying them into a predefined set of categories. Some of the practical applications of NER include:
  * Scanning news articles for the people, organizations and locations reported.
  * Providing concise features for search optimization: instead of searching the entire content, one may simply search for the major entities involved.
  * Quickly retrieving geographical locations talked about in Twitter posts.

 See also [N], [Entity Extraction], [NLP Benchmark]


# Nash Equilibrium

 In [Game Theory], ...

 More at:
  * [https://www.deepmind.com/blog/game-theory-as-an-engine-for-large-scale-data-analysis](https://www.deepmind.com/blog/game-theory-as-an-engine-for-large-scale-data-analysis)
  * [https://en.wikipedia.org/wiki/Nash_equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium)

 See also [N], ...


# Natural Intelligence

 See also [N], [Artificial Intelligence]


# Natural Language Generation

 See also [N], [Casual Language Modeling], [Decoder], [GPT Model]


# Natural Language Processing

# NLP

 A huge percentage of the world’s data and knowledge is in some form of human language. Can you imagine being able to read and comprehend thousands of books, articles and blogs in seconds? Obviously, computers can’t yet fully understand human text but we can train them to do certain tasks. For example, we can train our phones to autocomplete our text messages or to correct misspelled words. We can even teach a machine to have a simple conversation with a human. Natural Language Processing (NLP) is not a machine learning method per se, but rather a widely used technique to prepare text for machine learning. Think of tons of text documents in a variety of formats (word, online blogs, ….). Most of these text documents will be full of typos, missing characters and other words that needed to be filtered out. NLP applications includes:
  * Machine Translation (!Seq2Seq Models)
  * Question Answering
  * Semantic Search
  * Sentiment Analysis
  * Spam filtering
  * Text Summarization
  * Virtual Assistants (i.e chatbots)
 But also :
  * Classification ...
  * Language modeling (prediction analysis)
  * Topics ...
 Sample applications
  * A bot that converse with you in another language
  * A bot that correct your grammar
  * A/B testing for the content of call to action on website
  * Spam filtering

 See also [N], [BERT Model], [Chain Of Thought Prompting], [GPT Model], [Machine Translation], [NLTK], [Question Answering], [Sentiment Analysis], [Seq2Seq Model], [Spam Detection], [Sentiment Analysis], [Spam Filtering], [Text Summarization], [Virtual Assistant]


# Natural Language Programming Benchmark

# NLP Benchmark

  * [BLEU Benchmark] : for machine translation
  * Coref :  Links pronouns to antecedents. Also capable to take the perspective of a speak, e.g. I, you, my sister, etc refers to different people function of who said it.
  * [GLUE Benchmark] :
  * NER : Named Entity Recognition (place, people, date, etc)
  * Language Parser : Identitfy which group of words go together (as phrase) and which words are the subject or object of a verb.
  * SNLI : relation between 2 statements (contradict, neutral, or entailment)
  * [SQuAD Benchmark] : Question and answering
  * [SuperGLUE Benchmark] :
  * SRL : Semantic understanding (machine translation, information extraction, text summarization, question answering)
  * SST-5 : Sentiment analysis - https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained

  All of those can be summarize in the [HELM Benchmark]

 See also [N], [Benchmark], [Coreference], [Entity Extraction], [Language Parsing], [Model Benchmark], [Named Entity Recognition], [Question Answering], [Semantic Understanding], [Sentiment Analysis], [SNLI]


# Natural Language ToolKit

# NLTK

 At the moment, the most popular package for processing text is NLTK (Natural Language ToolKit), created by researchers at Stanford. The simplest way to map text into a numerical representation is to compute the frequency of each word within each text document. Think of a matrix of integers where each row represents a text document and each column represents a word. This matrix representation of the word frequencies is commonly called Term Frequency Matrix (TFM). From there, we can create another popular matrix representation of a text document by dividing each entry on the matrix by a weight of how important each word is within the entire corpus of documents. We call this method Term Frequency Inverse Document Frequency (TFIDF) and it typically works better for machine learning tasks.

 See also [N], [NLP], [TF], [TF-IDF], [TFM]


# Natural Language Interpretation

# NLI

 See [Natural Language Understanding]


# Natural Language Inference

# NLI

 See [Natural Language Understanding]


# Natural Language Supervision

# NLS

 See also [N], [CLIP Model]


# Natural Language Understanding

# NLU

 Natural-language understanding (NLU) is a subtopic of natural-language processing in artificial intelligence that deals with machine reading comprehension (Intent, slots ~ Alexa). Natural-language understanding is considered an AI-hard problem. There is considerable commercial interest in the field because of its application to automated reasoning, machine translation, question answering, news-gathering, text categorization, voice-activation, archiving, and large-scale content analysis. A popular json libary for this is snip-NLU.
 
 ```
"What will be the weather in paris at 9pm?"                     # Utterance = Input question to SNIP assistant ~ Alexa

# Transformed into
{
   "intent": {
      "intentName": "searchWeatherForecast",                    # NLU intent = sequence label or sequence classification (Name of the backend app/function on Alexa?)
      "probability": 0.95
   },
   "slots": [                                                   # Slot = NLU entities + NLU variables ?
      {
         "value": "paris",
         "entity": "locality",
         "slotName": "forecastLocality"
      },
      {
         "value": {
            "kind": "InstantTime",
            "value": "2018-02-08 20:00:00 +00:00"
         },
         "entity": "snips/datetime",                            # Entity ?
         "slotName": "forecastStartDatetime"
      }
   ]
}
 ```

 See also [N], [Autoencoding Model], [Entity Extraction]


# Negative Attribute

 Attribute to which the object needs to be the furthest away. Ex if you are trying to find a cat in a set of images, use a white noise image to describe negative attributes.

 See also [N], [Attribute], [Positive Attribute]


# Neptune AI Company

 An AI [Company]

 More at:
  * [https://neptune.ai/blog](https://neptune.ai/blog) 

 See also [N], ...


# Netflix Company

 See also [N], [Netflix Prize]


# Netflix Prize

 The Netflix Prize was an open competition for the best collaborative filtering algorithm to predict user ratings for films, based on previous ratings without any other information about the users or films, i.e. without the users being identified except by numbers assigned for the contest.

 The competition was held by Netflix, an online DVD-rental and video streaming service, and was open to anyone who is neither connected with Netflix (current and former employees, agents, close relatives of Netflix employees, etc.) nor a resident of certain blocked countries (such as Cuba or North Korea). On September 21, 2009, the grand prize of US$1,000,000 was given to the BellKor's Pragmatic Chaos team which bested Netflix's own algorithm for predicting ratings by 10.06%.

 {% youtube "https://www.youtube.com/watch?v=ImpV70uLxyw" %}

 More at:
  * [https://en.wikipedia.org/wiki/Netflix_Prize](https://en.wikipedia.org/wiki/Netflix_Prize)

 See also [N], [Netflix Company]

 
# Neural Architecture Search

 Find the best neural network architecture to use for the model.

 See also [N], [AutoML], [Neural Network]


# Neural Machine Translation

# NMT

 Neural machine translation (NMT) is an approach to machine translation that uses an [artificial neural network] to predict the likelihood of a sequence of words, typically modeling entire sentences in a single integrated model.

 They require only a fraction of the memory needed by traditional [statistical machine translation (SMT)][Statistical Machine Translation] models.

 Its main departure is the use of [vector] representations ("[embeddings][Word Embedding]", "continuous space representations") for words and internal states. 

 More at:
  * [https://en.wikipedia.org/wiki/Neural_machine_translation](https://en.wikipedia.org/wiki/Neural_machine_translation)

 See also [N], ...


# Neural Network

 ~ `Can discover and approximate a(ny?) function given fixed(-count?) inputs and fixed(-count?) outputs! = universal function approximator` A multi-layer perceptron. Also known as Artificial Neural Network (ANN). Can be used in supervised or unsupervised learning.
 
 ![]( {{site.assets}}/n/neural_network_example.png ){: width="100%"}

 `For a young person, the university he/she is coming from is important. Compare that to someone with experience, whose university may not matter so much. but experience and field are important. ==> the weight of university is function of the age ==> That's what the second layer correct for!`
 
 ![]( {{site.assets}}/n/neural_networks.png ){: width="100%"}

 More at:
  * playground - [https://playground.tensorflow.org/](https://playground.tensorflow.org/)

 See also [N], [Artificial Neural Network], [Early Stopping], [Feedforward Neural Network], [Hidden Layer], [Input Layer], [Output Layer], [Perceptron], [Recurrent Neural Network], [Universal Function Approximator]


# Neural Network Interpretability

 There is a growing sense that neural networks need to be interpretable to humans. The field of neural network interpretability has formed in response to these concerns. As it matures, two major threads of research have begun to coalesce: [feature visualization] and [attribution][Feature Attribution].

 See also [N], ...


# Neural Retriever

 {% pdf "https://arxiv.org/pdf/2205.16005.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2205.16005]

 See also [N], ...


# Neural Style Transfer

# NST

 Use for generating images. * Only 2 images, the base and the style image, with pre-trained VGG. * Perform back-propagation base image pixels, updating transfer style. * 3 loss functions: content, style, total variance.


# Neural Topic Modeling

# NTM

 An unsupervised learning algorithm that is used to organize a corpus of documents into topics that contain word groupings based on their statistical distribution. Documents that contain frequent occurrences of words such as "bike", "car", "train", "mileage", and "speed" are likely to share a topic on "transportation" for example. Topic modeling can be used to classify or summarize documents based on the topics detected or to retrieve information or recommend content based on topic similarities. The topics from documents that NTM learns are characterized as a latent representation because the topics are inferred from the observed word distributions in the corpus. The semantics of topics are usually inferred by examining the top ranking words they contain. Because the method is unsupervised, only the number of topics, not the topics themselves, are prespecified. In addition, the topics are not guaranteed to align with how a human might naturally categorize documents. // Topic modeling provides a way to visualize the contents of a large document corpus in terms of the learned topics. Documents relevant to each topic might be indexed or searched for based on their soft topic labels. The latent representations of documents might also be used to find similar documents in the topic space. You can also use the latent representations of documents that the topic model learns for input to another supervised algorithm such as a document classifier. Because the latent representations of documents are expected to capture the semantics of the underlying documents, algorithms based in part on these representations are expected to perform better than those based on lexical features alone. // Although you can use both the Amazon SageMaker NTM and LDA algorithms for topic modeling, they are distinct algorithms and can be expected to produce different results on the same input data.

 See also [N], [LDA], [Unsupervised Learning]


# Neural Winter

 A period between 1998 and 2007, where research in deep learning dropped off due to limitations on data and compute. Before 1998 = academic research. After 2007 GPUs are broadly available (CUDA, etc).


# Neuralink Company

 {% youtube "https://www.youtube.com/watch?v=VlDx6yzomEg" %}

 See also [N], [Company] [MindPong Game]


# Next Sentence Prediction

# NSP

 Pretrain a [CLS] token in BERT by performing a classification task. Did sentence B come directly after sentence A? Yes or No ? A classification problem with softmax function on is_next and not_next (sum of probabilities = 1).
 
 ```
A: Istanbul is a great city to visit
B: I was just there
 ```

 See also [N], [BERT Model], [Softmax Function]


# Next Word Prediction

 See also [N], [Self-Supervised Learning]


# Node

 See [Artificial Neuron]


# Noise

 Add to an input to turn the same input going through a given model into different output.

 See also [N], [Noise Vector]


# Noise Vector

 A latent noise vector is also passed in as an input and this is responsible for ensuring that there is a flavor to each output generated by the generator, even when the same input is provided.

 See also [N], [Noise], [U-Net Generator]


# Non-Linear Regression

 More at:
  * [https://medium.com/@toprak.mhmt/non-linear-regression-4de80afca347](https://medium.com/@toprak.mhmt/non-linear-regression-4de80afca347)

 See also [N], [Linear Regression], [Polynomial Regression], [Regression]


# Nvidia Company

 {% youtube "https://www.youtube.com/watch?w=Gn_IMIPrX9s" %}

 {% youtube "https://www.youtube.com/watch?w=erL77suOVPg" %}

 {% youtube "https://www.youtube.com/watch?w=GuV-HyslPxk" %}


 Models
  * [Isaac Gym] : environment for RL
  * [Megatron Model] : NLP large language model
  * [RIVA Model] : text-to-speech model
  * [VIMA Model] : multi-modal ? model for robots?

 Metaverse
  * [Omniverse] : The metaverse by Nvidia!

 More at :
  * [https://blogs.nvidia.com/blog/2022/12/16/top-five-nvidia-ai-videos/](https://blogs.nvidia.com/blog/2022/12/16/top-five-nvidia-ai-videos/)

 See also [N], [Company]
