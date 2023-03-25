---
title: T
permalink: /t/

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


# T-Distribution

 ~ normal distribution with fatter tails!

 See also [T], ...


# t-Stochastic Neighbor Embedding

# t-SNE

 Another popular method is t-Stochastic Neighbor Embedding (t-SNE), which does non-linear dimensionality reduction. People typically use t-SNE for data visualization, but you can also use it for machine learning tasks like reducing the feature space and clustering, to mention just a few. The next plot shows an analysis of the MNIST database of handwritten digits. MNIST contains thousands of images of digits from 0 to 9, which researchers use to test their clustering and classification algorithms. Each row of the dataset is a vectorized version of the original image (size 28 x 28 = 784) and a label for each image (zero, one, two, three, …, nine). Note that we’re therefore reducing the dimensionality from 784 (pixels) to 2 (dimensions in our visualization). Projecting to two dimensions allows us to visualize the high-dimensional original dataset.

 ![]( {{site.assets}}/t/tsne.gif ){: width="100%"}
 
 See also [T], [Dimensionality Reduction]


# Tanh Activation Function

 Pros:
  * Regulate the values to be always between -1 and 1. Used in RNN.
  * solve exploding gradient problem

 Cons:
  * vanishing gradient problem.

 ![]( {{site.assets}}/t/tanh_activation_function.png ){: width="100%"}

 See also [T], [Activation Function], [Exploding Gradient Problem], [Recurrent Neural Network], [Vanishing Gradient Problem]


# T5 Model

 Trained with C4. A Transformer based architecture that uses a text-to-text approach. Every task – including translation, question answering, and classification – is cast as feeding the model text as input and training it to generate some target text. This allows for the use of the same model, loss function, hyperparameters, etc. across our diverse set of tasks. The changes compared to BERT include:
   * adding a causal decoder to the bidirectional architecture.
   * replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.
 T5 claims the state of the art on more than twenty established NLP tasks. It’s extremely rare for a single method to yield consistent progress across so many tasks. That list includes most of the tasks in the GLUE and SuperGLUE benchmarks, which have caught on as one of the main measures of progress for applied language understanding work of this kind (and which my group helped to create). On many of these task datasets, T5 is doing as well as human crowdworkers, which suggests that it may be reaching the upper bound on how well it is possible to do on our metrics.

 ![]( {{site.assets}}/t/t5_model.jpeg ){: width="100%"}

 More at:
   * [https://paperswithcode.com/method/t5](https://paperswithcode.com/method/t5)
   * code - [https://github.com/google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
   * blog article - [https://medium.com/syncedreview/google-t5-explores-the-limits-of-transfer-learning-a87afbf2615b](https://medium.com/syncedreview/google-t5-explores-the-limits-of-transfer-learning-a87afbf2615b)
   * [https://paperswithcode.com/method/t5#:~:text=T5%2C%20or%20Text%2Dto%2D,to%20generate%20some%20target%20text.](https://paperswithcode.com/method/t5#:~:text=T5%2C%20or%20Text%2Dto%2D,to%20generate%20some%20target%20text.)

 See also [T], [BERT Model], [Colossal Clean Crawled Corpus], [Google Company], [Switch Transformer], [Transformer Architecture], [Transformer Model]


# Target Attribute

 This is the attribute that we want the XGBoost to predict. In unsupervised training, corresponds to a label in supervised training.

 See also [T], [Feature], [Unsupervised Learning], [XGBoost]


# Task

 To discern a task:
  * Will the activity engage learners’ interest?
  * Is there a primary focus on meaning?
  * Is there a goal or an outcome?
  * Is success judged in terms of the result?
  * Is completion a priority?
  * Does the activity relate to real-world activities?

 If your answer is yes to all the questions, you can be sure that the classroom activity you have in mind is task-like.

 More at:
  * [https://www.teacheracademy.eu/blog/task-based-learning/](https://www.teacheracademy.eu/blog/task-based-learning/)

 See also [T], [Task-Based Learning]


# Task-Based Learning

# TBL

 Focus on completing the task, but use all your skills (and develop new ones) on the way. Example: Start a company? Start an AI club? Identify problem, opportunities, and improve + find new tools along the way.

 More at:
   * [https://www.teacheracademy.eu/blog/task-based-learning/](https://www.teacheracademy.eu/blog/task-based-learning/)

 See also [T], [Learning Method], [Task]


# Taxonomy

 See also [T], ...


# Tensor

 A matrix (not a vector) of inputs. Ex an image is converted to a tensor and fed to the input of a convolutional neural network.

 See also [T], [Convolutional Neural Network], [Vector]


# Tensor Processing Unit

# TPU

 See also [T], [Google Company], [Tensor]


# TensorFlow ML Framework

 See also [T], [Deep Learning Framework], [Distributed Training], [Google Company], [Machine Learning Framework]


# Terminator Movie

 The Terminator is a 1984 American science fiction action film directed by James Cameron. It stars Arnold Schwarzenegger as the Terminator, a cyborg assassin sent back in time from 2029 to 1984 to kill Sarah Connor (Linda Hamilton), whose unborn son will one day save mankind from extinction by Skynet, a hostile artificial intelligence in a post-apocalyptic future.

 {% youtube "https://www.youtube.com/watch?v=k64P4l2Wmeg" %}

 {% youtube "https://www.youtube.com/watch?v=piPIckK_R0o" %}

 More at:
  * [https://en.wikipedia.org/wiki/The_Terminator](https://en.wikipedia.org/wiki/The_Terminator)

 See also [T], [AI Movie]


# Term Frequency

# TF

 measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
 
 ```
 TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
 ```

 See also [T], [TF-IDF]


# Term Frequency Inverse Document Frequency

# TF-IDF

 TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model. Tf-idf can be successfully used for stop-words filtering in various subject fields including text summarization and classification. Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.

 ![]( {{site.assets}}/t/tfidf.jpeg ){: width="100%"}

 More at:
  * [http://tfidf.com/](http://tfidf.com/)

 See also [T], [NLP], [Term Frequency Matrix]


# Term Frequency Matrix

# TFM

 The simplest way to map text into a numerical representation is to compute the frequency of each word within each text document. Think of a matrix of integers where each row represents a text document and each column represents a word. This matrix representation of the word frequencies is commonly called Term Frequency Matrix (TFM).

 See also [T], [NLP], [Term Frequency Inverse Document Frequency]


# Test Subset

 Use to see how the model built with the training subset and the development subset performs on new data. The performance of the model will show issues related to overfitting, etc. This subset includes only the features since we want to predict the labels. The performance we see on the test subset is what we can reasonably see in production. :warning: The test subset cannot be used at any time in the training or post-training phase (i.e model auto tuning, eg over vs underfitting).

 ![]( {{site.assets}}/t/train_test_subsets.png ){: width="100%"}

 See also [T], [Dataset], [Development Subset], [Overfitting], [Training Subset]


# Text Embedding

 See [Word Embedding]


# Text Reconstruction

 ![]( {{site.assets}}/t/text_reconstruction.png ){: width="100%"}

 Above is a pipeline for text reconstruction. The input text is fed to DALL-E/SD to generate an image, whcih is fed to Flamingo/BLIP to generate a caption, which is fed to DALL-E/SD to reconstruct a text caption. The generated text-caption is compared with the input text using the CLIP text encoder in the embedding space.

 See also [T], [BLIP Model], [CLIP Text Encoder], [Image Reconstruction]


# Text Summarization

 Summarizing a text involves reducing its size while keeping key information and the essential meaning. Some everyday examples of text summarization are news headlines, movie previews, newsletter production, financial research, legal contract analysis, and email summaries, as well as applications delivering news feeds, reports, and emails.

 See also [T], [Natural Language Processing]


# Text-To-Speech Model

# TTS Model

 Models such as
  * [Riva][Riva Model] by [Nvidia][Nvidia Company]
  * [WaveNet][WaveNet Model] by [DeepMind][DeepMind Company]
  * [Whisper][Whisper Model] by [OpenAI][OpenAI Company]

 See also [T], [Sequence To Sequence Model]


# Text-To-Text Transfer Transformer Model

# T5 Model

 The T5 model, pre-trained on C4, achieves state-of-the-art results on many NLP benchmarks while being flexible enough to be fine-tuned to a variety of important downstream tasks.

 ![]( {{site.assets}}/t/t5_model.gif ){: width="100%"}

 See also [T], [Colossal Clean Crawled Corpus], [Google Company]


# The Matrix Movie

 The Matrix is a 1999 science fiction action film written and directed by the Wachowskis. It is the first installment in the Matrix film series, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano, and depicts a dystopian future in which humanity is unknowingly trapped inside the Matrix, a simulated reality that intelligent machines have created to distract humans while using their bodies as an energy source.

 {% youtube "https://www.youtube.com/watch?v=nUEQNVV3Gfs" %}

 More at:
  * [https://en.wikipedia.org/wiki/The_Matrix](https://en.wikipedia.org/wiki/The_Matrix)

 See also [M], [AI Movie], [Metaverse]


# Theano

 See also [T], ...


# Theory Of Mind

# ToM

 Theory of mind (ToM), or the ability to impute unobservable mental states to others, is central to human social interactions, communication, empathy, self-consciousness, and morality. We administer classic false-belief tasks, widely used to test ToM in humans, to several language models, without any examples or pre-training. Our results show that models published before 2022 show virtually no ability to solve ToM tasks. Yet, the January 2022 version of GPT-3 (davinci-002) solved 70% of ToM tasks, a performance comparable with that of seven-year-old children. Moreover, its November 2022 version (ChatGPT/davinci-003), solved 93% of ToM tasks, a performance comparable with that of nine-year-old children. These findings suggest that ToM-like ability (thus far considered to be uniquely human) may have spontaneously emerged as a byproduct of language models’ improving language skills.

 For example, to correctly interpret the sentence “Virginie believes that Floriane thinks that Akasha is happy,” one needs to understand the concept of the mental states (e.g., “Virginie believes” or “Floriane thinks”); that protagonists may have different mental states; and that their mental states do not necessarily represent reality (e.g., Akasha may not be happy, or Floriane may not really think that).

 Beware:
  * :warning: abilities that rely on ToM ==> empathy, moral judgment, or self-consciousness.

 More at:
  * philosophy - [https://iep.utm.edu/theomind/](https://iep.utm.edu/theomind/)
  * colab - [https://colab.research.google.com/drive/1zQKSDEhqEFcLCf5LuW--A-TGcAhF19hT](https://colab.research.google.com/drive/1zQKSDEhqEFcLCf5LuW--A-TGcAhF19hT)
  * paper - [https://arxiv.org/abs/2302.02083](https://arxiv.org/abs/2302.02083)

 See also [T], [Emergent Ability], [GPT Model], [Large Language Model]


# Threat Model

 See also [T], [Adversarial Attack], [Adversarial Policy]


# Time-Series Predictive Analysis

 `~ look at a sequence of elements/images, find the next element/image = time series representation`. For example, music can be represented with a time series. In this approach, music is represented as time-series data, where each note is based on the previous notes. 

 See also [T], [Autoregressive Model]


# Tokenization

 Tokenization is the first step in any NLP pipeline. It has an important effect on the rest of your pipeline. A tokenizer breaks unstructured data and natural language text into chunks of information that can be considered as discrete elements. The token occurrences in a document can be used directly as a vector representing that document. Tokenization can separate sentences, words, characters, or subwords. When we split the text into sentences, we call it sentence tokenization. For words, we call it word tokenization.

 See also [T], [Tokenizer]


# Tokenizer

 :warning: Pass the tokens and their positions (index in the list!) :warning: The tokens are then coded in number / ~ line number of token in file :warning: Prefix and suffix may be added to token for multi-input processing (e.g. "[CLS]" or "[SEP]" )
 Two terms we see a lot when working with tokenization is uncased and cased (Note this has little to do with the BERT architecture, just tokenization!).
  * uncased --> removes accents, lower-case the input :warning: Usually better for most situation as case does NOT contribute to context
  * cased --> does nothing to input :warning:  recommended where case does matter, such as Name Entity Recognition
 and more
  * clean_text : remove control characters and replace all whitespace with spaces
  * handle_chinese_chars : includes spaces around Chinese characters (if found in the dataset)
 
 ```
                           Hope, is the only thing string than fear! #Hope #Amal.M
# Space tokenizer (split)  ['Hope,', 'is', 'the', 'only', 'thing', 'string', 'can', 'fear!', '#hope', '#Amal.M']
# Word tokenizer           ['Hope', ',', 'is', 'the',  'only', 'thing', ',string', 'than', 'fear', '!',  '#', 'Hope', '#', 'Amal.M']
# Sentence tokenizer       ['Hope, is the only thing string than fear!', '#Hope #Amal.M']
# Word-Punct tokenizer     ['Hope', ',', 'is', 'the',  'only', 'thing', ',string', 'than', 'fear', '!',  '#', 'Hope', '#', 'Amal', '.', 'M']

                           What you don't want to be done to yourself, don't do to others...
# Treebank word tokenizer  ['What', 'you', 'do', "n't", 'want', 'to', 'be', 'done', 'to', 'yourself', ',', 'do', "n't", 'do', 'to', 'others', '...']
 ```

 
 ```
# Wordpiece tokenizer :
   * It works by splitting words either into the full forms (e.g., one word becomes one token) or into word pieces — where one word can be broken into multiple tokens.
   * the original BERT uses.
Word	        Token(s)
surf	        ['surf']
surfing	        ['surf', '##ing']
surfboarding	['surf', '##board', '##ing']
surfboard	['surf', '##board']
snowboard	['snow', '##board']
snowboarding	['snow', '##board', '##ing']
snow	        ['snow']
snowing	        ['snow', '##ing']
 ```

 
 ```
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./bert-it')
tokenizer('ciao! come va?')  # hi! how are you?
{
 'input_ids': [2, 13884, 5, 2095, 2281, 35, 3],               # Tokenized input with padding/prefix/suffix
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0],                     # 0 or 1 or ? : Belongs to sentence #0, #1, #?
 'attention_mask': [1, 1, 1, 1, 1, 1, 1]                      # 0 or 1 : 0 if token is padding
}


with open('./bert-it/vocab.txt', 'r') as fp:
    vocab = fp.read().split('\n')
vocab[2], vocab[13884], vocab[5], \
    vocab[2095], vocab[2281], vocab[35], \
        vocab[3]
('[CLS]', 'ciao', '!', 'come', 'va', '?', '[SEP]')
 ```

 See also [T], [BERT Model], [Tokenization]


# Torch

 at the origin of pytorch?

 See also [T], [PyTorch]


# Training Subset

 Use with the development subset to build the model.

 See also [T], [Cross-validation Sampling Method], [Dataset], [Development Subset], [Overfitting], [Test Subset]


# Transfer Learning

 ~ Learning on one use-case can be reused for another case. Benefits:
  * training cost is reduced
  * the way human work!
  * Training when not enough data? --> reuse previous learning to build new model and change only a delta

 Approach:
  * select a source model from a model repository (ex: huggingface)
  * reuse and train model

 Example:
   * BERT + financial data --> FinBERT
   * BERT + classification layer --> BERT for classification !!!!!

 Let’s pretend that you’re a data scientist working in the retail industry. You’ve spent months training a high-quality model to classify images as shirts, t-shirts and polos. Your new task is to build a similar model to classify images of dresses as jeans, cargo, casual, and dress pants. Can you transfer the knowledge built into the first model and apply it to the second model? Yes, you can, using Transfer Learning. `Transfer Learning refers to re-using part of a previously trained neural net and adapting it to a new but similar task` Specifically, once you train a neural net using data for a task, you can transfer a fraction of the trained layers and combine them with a few new layers that you can train using the data of the new task. By adding a few layers, the new neural net can learn and adapt quickly to the new task. The main advantage of transfer learning is that you need less data to train the neural net, which is particularly important because training for deep learning algorithms is expensive in terms of both time and money (computational resources) — and of course it’s often very difficult to find enough labeled data for the training. Let’s return to our example and assume that for the shirt model you use a neural net with 20 hidden layers. After running a few experiments, you realize that you can transfer 18 of the shirt model layers and combine them with one new layer of parameters to train on the images of pants. The pants model would therefore have 19 hidden layers. The inputs and outputs of the two tasks are different but the re-usable layers may be summarizing information that is relevant to both, for example aspects of cloth. Transfer learning has become more and more popular and there are now many solid pre-trained models available for common deep learning tasks like image and text classification.

 Transfer learning is one of the most useful discoveries to come out of the computer vision community. Stated simply, transfer learning allows one model that was trained on different types of images, e.g. dogs vs cats, to be used for a different set of images, e.g. planes vs trains, while reducing the training time dramatically. When Google released !ImageNet, they stated it took them over 14 days to train the model on some of the most powerful GPUs available at the time. Now, with transfer learning, we will train an, albeit smaller, model in less than 5 minutes.

 ![]( {{site.assets}}/t/transfer_learning.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=BqqfQnyjmgg" %}

 To execute transfer learning, transfer the weights of the trained model to the new one. Those weights can be retrained entirely, partially (layering), or not at all (prepend a new process such as classification on an encoder output!) :warning: :warning: :warning: BEWARE: When using transfer learning, you transfer the bias for the pretrained model

 See also [T], [BERT Model], [GPT Model], [ImageNet Dataset], [Insufficient Data Algorithm], [Pre-Trained Model]


# Transform Function

# TF

 A function to transform the input dataset. For ex: rotate image in the right position.

 See also [T], [Labeling Function], [Slicing Function], [Snorkel Program]


# Transformer Architecture

 The Transformer is a recent deep learning model for use with sequential data such as text, time series, music, and genomes. Whereas older sequence models such as recurrent neural networks (RNNs) or long short-term memory networks (LSTMs) process data sequentially, the Transformer processes data in parallel (can therefore be parallelised on machines in the cloud!). This allows them to process massive amounts of available training data by using powerful GPU-based compute resources. Furthermore, traditional RNNs and LSTMs can have difficulty modeling the long-term dependencies of a sequence because they can forget earlier parts of the sequence. Transformers use an attention mechanism to overcome this memory shortcoming by directing each step of the output sequence to pay “attention” to relevant parts of the input sequence. For example, when a Transformer-based conversational AI model is asked “How is the weather now?” and the model replies “It is warm and sunny today,” the attention mechanism guides the model to focus on the word “weather” when answering with “warm” and “sunny,” and to focus on “now” when answering with “today.” This is different from traditional RNNs and LSTMs, which process sentences from left to right and forget the context of each word as the distance between the words increases.
  * word positioning (feed the work and its position in the sentence)
  * Attention
   * self-attention (link pronouns, subject to verbs, adjectives to nouns, adverbs)
   * cross-attention (positioning of words between languages, i.e. input and output)
 
 {% pdf "{{site.assets}}/t/transformer_paper.pdf" %}
 
 ![]( {{site.assets}}/t/transformer_model_architecture.png ){: width="100%"}

 ![]( {{site.assets}}/t/transformer_model_architecture_overview.png ){: width="100%"}
 
 The transformer is a current-state of the art NLP model. It relies almost entirely on self-attention to model the relationship between tokens in a sentence rather than relying on recursion like RNNs and LSTMs do.
 
 ```
“Adept’s technology sounds plausible in theory, [but] talking about Transformers needing to be ‘able to act’ feels a bit like misdirection to me,” Mike Cook, an AI researcher at the Knives & Paintbrushes research collective, which is unaffiliated with Adept, told TechCrunch via email. “Transformers are designed to predict the next items in a sequence of things, that’s all. To a Transformer, it doesn’t make any difference whether that prediction is a letter in some text, a pixel in an image, or an API call in a bit of code. So this innovation doesn’t feel any more likely to lead to artificial general intelligence than anything else, but it might produce an AI that is better suited to assisting in simple tasks.”
# https://techcrunch.com/2022/04/26/2304039/
 ```

 More at:
  * [https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)

 See also [T], [Action Transformer], [Attention Score], [Attention-Based Model], [Autoregressive], [Generative Model], [Long Short Term Memory Network], [Masked Self-Attention], [Multi-Head Attention], [Recurrent Neural Network], [Self-Attention]


# Transformer Model

 Models that are based on transformers are:
   * BERT models : use the encoder side of the transformer
   * GPT models : use the decoder side of the transformer
   * T5 models : use the encode-decoder, the whole transformer !

 :warning: Beware:
   * As far as I can tell, transformer are the only models that can do transfer learning. Is this true?

 See also [T], [BERT Model], [GPT Model], [T5 Model]


# Translation

 See also [T], [Emergent Ability]


# Tree Parzen Estimators

# TPE

 See also [T], [Gaussian Process], [Random Forest]


# Triton

 A low level framework to compile code on any GPU. A major step toward bypassing CUDA and the NVIDIA lock in!

 More at :
  * [https://openai.com/blog/triton/](https://openai.com/blog/triton/)
  * [https://www.semianalysis.com/p/nvidiaopenaitritonpytorch](https://www.semianalysis.com/p/nvidiaopenaitritonpytorch)

 See also [T], [Nvidia Company], [OpenAI Company]


# Truth

 Can sometimes be discovered by observation and inductive reasoning, but not always!

 See also [T], [Inductive Reasoning]


# Tuning Parameter

 See [Hyperparameter]
