---
title: G
permalink: /g/

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


# Galactica Large Language Model

# Galactica LLM

 Galactica is “a large language model that can store, combine and reason about scientific knowledge,” according to a paper published by Meta AI. It is a transformer model that has been trained on a carefully curated dataset of 48 million papers, textbooks and lecture notes, millions of compounds and proteins, scientific websites, encyclopedias, and more. Galactica was supposed to help scientists navigate the ton of published scientific information. Its developers presented it as being able to find citations, summarize academic literature, solve math problems, and perform other tasks that help scientists in research and writing papers.

 {% pdf "{{site.assets}}/g/galactica_model_paper.pdf" %}

 More at:
  * what happened to galactica? - [https://www.louisbouchard.ai/galactica/](https://www.louisbouchard.ai/galactica/)
  * take aways - [https://bdtechtalks.com/2022/11/21/meta-ai-galactica](https://bdtechtalks.com/2022/11/21/meta-ai-galactica)
  * site - [http://galactica.org](http://galactica.org)

 See also [G], [Large Language Model], [Meta Company]


# Game Theory

 Used in model architecture, such as [GAN] and used for decision making process, such as in [Pluribus Model] and [Multi-Agent Model]

 More at:
  * ...

 See also [G], [Nash Equilibrium], [Shapley Value]


# Gated Recurrent Unit Cell

# GRU Cell

 Cell or module that can be used in the RNN chain of a Long Short Term Memory, or LSTM Network. A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models and therefore less compute intensive. This cell has been growing increasingly popular.

 ![]( {{site.assets}}/g/gated_recurrent_unit_cell2.png ){: width="100%"}

 ![]( {{site.assets}}/g/gated_recurrent_unit_cell.png ){: width="100%"}

 More at:
  * [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

 See also [G], [Long Short Term Memory Network], 


# Gato Model

 A model developed by [DeepMind][DeepMind Company] that uses [Multi-Task Learning]

 ![]( {{site.assets}}/g/gato_model.png ){: width="100%"}

 {% pdf "{{site.assets}}/g/gato_model_paper.pdf" %}

 {% youtube "https://www.youtube.com/watch?v=wSQJZHfAg18" %}

 More at:
  * [https://www.deepmind.com/publications/a-generalist-agent](https://www.deepmind.com/publications/a-generalist-agent)
  * [https://www.deepmind.com/blog/a-generalist-agent](https://www.deepmind.com/blog/a-generalist-agent)

 See also [G], ...


# Gaussian Distribution

 The Gaussian distribution, normal distribution, or bell curve, is a probability distribution which accurately models a large number of phenomena in the world. Intuitively, it is the mathematical representation of the general truth that many measurable quantities, when taking in aggregate tend to be of the similar values with only a few outliers which is to say that many phenomena follow the central limit theorem.

 ![]( {{site.assets}}/g/gaussian_distribution.png ){: width="100%"}

 See also [G], [Central Limit Theorem], [Gaussian Process]


# Gaussian Process

 See also [G], [Random Forest], [Tree Parzen Estimators]


# Gen Model

 A text-to-video model built by the [Runway Company]

 {% youtube "https://www.youtube.com/watch?v=trXPfpV5iRQ" %}

 {% pdf "{{site.assets}}/g/gen2_model_paper.pdf" %}

 More at:
  * [https://research.runwayml.com/gen2](https://research.runwayml.com/gen2)
  * [https://arxiv.org/abs/2302.03011](https://arxiv.org/abs/2302.03011)

 See also [G], ...


# General Language Understanding Evaluation Benchmark

# GLUE Benchmark

 The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. GLUE consists of:
  * A benchmark of nine sentence- or sentence-pair language understanding tasks built on established existing datasets and selected to cover a diverse range of dataset sizes, text genres, and degrees of difficulty,
  * A diagnostic dataset designed to evaluate and analyze model performance with respect to a wide range of linguistic phenomena found in natural language, and
  * A public leaderboard for tracking performance on the benchmark and a dashboard for visualizing the performance of models on the diagnostic set.

 The format of the GLUE benchmark is model-agnostic, so any system capable of processing sentence and sentence pairs and producing corresponding predictions is eligible to participate. The benchmark tasks are selected so as to favor models that share information across tasks using parameter sharing or other transfer learning techniques. The ultimate goal of GLUE is to drive research in the development of general and robust natural language understanding systems.

 {% pdf "{{site.assets}}/g/glue_paper.pdf" %}

 More at:
  * [https://gluebenchmark.com/](https://gluebenchmark.com/)

 See also [G], [Benchmark], [SuperGLUE Benchmark]


# Generalized Additive 2 Model

# GA2M

 Use GA2Ms if they are significantly more accurate than [GAMs][GAM], especially if you believe from your domain knowledge that there are real feature interactions, but they are not too complex. This also gives the advantages of a [White Box Model], with more effort to interpret.
 
 More at:
  * constrained GA2M paper - [https://arxiv.org/abs/2106.02836](https://arxiv.org/abs/2106.02836)

 See also [G], ...


# Generalized Additive Model

# GAM

 Generalized Additive Models (GAMs) were developed in the 1990s by Hastie and Tibshirani. 
 
 More at:
  * [https://www.fiddler.ai/blog/a-gentle-introduction-to-ga2ms-a-white-box-model](https://www.fiddler.ai/blog/a-gentle-introduction-to-ga2ms-a-white-box-model)

 See also [G], ...


# Generative Adversarial Network

# GAN

 So why do we want a generative model? Well, it’s in the name! We wish to generate something an image, music, something! But what do we wish to generate? `Typically, we wish to generate data` (I know, not very specific). More than that though, it is likely that we wish to generate data that is never before seen, yet still fits into some data distribution (i.e. some pre-defined dataset that we have already set aside and that was used to build a discriminator). GANs, a generative AI technique, pit 2 networks against each other to generate new content. The algorithm consists of two competing networks: a generator and a discriminator. A generator is a convolutional neural network (CNN) that learns to create new data resembling the source data it was trained on. The discriminator is another convolutional neural network (CNN) that is trained to differentiate between real and synthetic data. The generator and the discriminator are trained in alternating cycles such that the generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.

 ![]( {{site.assets}}/g/vanilla_gan.png ){: width="100%"}

 There are different types of GAN, including:
  * Vanilla GAN
  * Cycle GAN
  * Conditional GAN
  * Deep Convoluted GAN
  * Style GAN
  * Super Resolution GAN (SRGAN)

 {% pdf "{{site.assets}}/g/generative_adversarial_networks_paper.pdf" %}

 More at:
  * [http://hunterheidenreich.com/blog/what-is-a-gan/](http://hunterheidenreich.com/blog/what-is-a-gan/)
  * GAN with Keras - [https://python.plainenglish.io/exploring-generative-adversarial-networks-gans-in-two-dimensional-space-922ee342b253](https://python.plainenglish.io/exploring-generative-adversarial-networks-gans-in-two-dimensional-space-922ee342b253)

 See also [G], [AR-CNN], [Convolutional Neural Network], [Conditional GAN], [Cycle GAN], [DeepComposer], [Discriminator], [Generative Model], [Generator]


# Generative Artificial Intelligence

# Generative AI

 Generative artificial intelligence (AI) describes algorithms (such as ChatGPT) that can be used to create new content, including audio, code, images, text, simulations, and videos. Recent new breakthroughs in the field have the potential to drastically change the way we approach content creation.

 More at:
  * https://www.mckinsey.com/featured-insights/mckinsey-explainers/what-is-generative-ai

 See also [G], [DALL-E Model], [ChatGPT Model]


# Generative Classifier

 Generative Classifiers tries to model class, i.e., what are the features of the class. In short, it models how a particular class would generate input data. When a new observation is given to these classifiers, it tries to predict which class would have most likely generated the given observation. Such methods try to learn about the environment. An example of such classifiers is Naive Bayes. Mathematically, generative models try to learn the joint probability distribution, `p(x,y)`, of the inputs x and label y, and make their prediction using Bayes rule to calculate the conditional probability, `p(y|x)`, and then picking a most likely label. Thus, it tries to learn the actual distribution of the class.

 ![]( {{site.assets}}/g/generative_classifier.png ){: width="100%"}

 See also [G], [Bayesian Network], [Hidden Markov Model], [Markov Random Field], [Naive Bayes]


# Generative Design

 [Generative AI] applied to architecture, design of parts, etc. everything that normally use CAD tools!

 {% youtube "https://www.youtube.com/watch?v=RrL4HshuzUw" %}

 More at:
  * [https://www.generativedesign.org/](https://www.generativedesign.org/)
  * [https://www.ptc.com/en/blogs/cad/beginner-guide-generative-design](https://www.ptc.com/en/blogs/cad/beginner-guide-generative-design)

 See also [G], ...


# Generative Model

 AI models that generate/create content. Examples of Generative AI techniques include:
  * [Diffusion Models][Diffusion Model]
  * [Generative Adversarial Networks (GAN)][Generative Adversarial Network]
  * [Variational autoencoders (VAEs)][Variational Autoencoder] = Hidden state is represented by a distribution, which is then sampled and decoded (Q: what is mean and variance?)
  * [Transformers][Transformer Model]
   * Decoders with masked self-attention

 ![]( {{site.assets}}/g/generative_model_1.png ){: width="100%"}

 ![]( {{site.assets}}/g/generative_model_2.png ){: width="100%"}

 More at :
  * Generative Modeling by Estimating Gradients of the Data Distribution - [https://yang-song.net/blog/2021/score/](https://yang-song.net/blog/2021/score/)

 See also [G], [Decoder], [Flow-Based Model], [Masked Self-Attention]
 

# Generative Pre-Trained Transformer Model

# GPT Model

# GPT-3

# GPT-4

 Before GPT-3 there was no general language model that could perform well on an array of NLP tasks. Language models were designed to perform one specific NLP task, such as text generation, summarization, or classification, using existing algorithms and architectures. GPT-3 has extraordinary capabilities as a general language model. GPT-3 is pre-trained on a corpus of text from five datasets: Common Crawl, !WebText2, Books1, Books2, and Wikipedia. 
  * By default, GPT-2 remembers the last 1024 words. That the max? length of the left-side context?
  * GPT-3 possesses 175 billion weights connecting the equivalent of 8.3 million [artificial neurons][Artificial Neuron] arranged 384 layers deep.
    * GPT-2 and GPT-3 have fundamentally the same architecture
    * But each generation of models ~ 10-100x increase in compute/size
    * The difference in using these models is qualitatively extremely different

 {% pdf "{{site.assets}}/g/gpt3_model_paper.pdf" %}

 GPT4 released on Tuesday 03/14/2023

 ![]( {{site.assets}}/g/gpt4_model_exams.png ){: width="100%"}

 {% pdf "{{site.assets}}/g/gpt4_model_paper.pdf" %}

 Early experiement with GPT-4 have shown sparks of [Artificial General Intelligence]!

 {% youtube "https://www.youtube.com/watch?v=outcGtbnMuQ" %}

 {% youtube "https://www.youtube.com/watch?v=PEjl7-7lZLA" %}

 {% youtube "https://www.youtube.com/watch?v=6Hewb1wlOlo" %}

 More at:
   * GPT-1 paper - [https://paperswithcode.com/paper/improving-language-understanding-by](https://paperswithcode.com/paper/improving-language-understanding-by)
   * GPT-2 paper - [https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (also attached)
   * GPT-3 paper -
   * GPT-4 paper - [https://cdn.openai.com/papers/gpt-4.pdf](https://cdn.openai.com/papers/gpt-4.pdf)
   * GPT-4 site - [https://openai.com/research/gpt-4](https://openai.com/research/gpt-4)
   * GPT fine-tuning - [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
   * gpt vs chatgpt vs instructgpt - [https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5](https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5)

 See also [G], [ChatGPT Model], [Digital Watermark], [Fine-Tuning], [InstructGPT Model], [Natural Language Processing], [Pre-Trained Model]


# Generator

 A neural network that generates music, image, something and is getting feedback from a Discriminator neural network. How does the generator generate images? Solution : The generator takes noise from latent space and further based on test and trial feedback , the generator keeps improvising on images. After a certain times of trial and error , the generator starts producing accurate images of a certain class which are quite difficult to differentiate from real images.

 See also [G], [DeepComposer], [Discriminator], [Generative Adversarial Network], [Generator Loss], [Latent Space]


# Generator Loss

 Typically when training any sort of model, it is a standard practice to monitor the value of the loss function throughout the duration of the training. The discriminator loss has been found to correlate well with sample quality. You should expect the discriminator loss to converge to zero and the generator loss to converge to some number which need not be zero. When the loss function plateaus, it is an indicator that the model is no longer learning. At this point, you can stop training the model. You can view these loss function graphs in the AWS DeepComposer console:

 ![]( {{site.assets}}/g/generator_loss.png ){: width="100%"}

 After 400 epochs of training, discriminator loss approaches near zero and the generator converges to a steady-state value. Loss is useful as an evaluation metric since the model will not improve as much or stop improving entirely when the loss plateaus.

 See also [G], [Discriminator Loss], [Loss Function], [Loss Graph]


# Genetic Programming

 Genetic programming is a technique to create algorithms that can program themselves by simulating biological breeding and Darwinian evolution. Instead of programming a model that can solve a particular problem, genetic programming only provides a general objective and lets the model figure out the details itself. The basic approach is to let the machine automatically test various simple evolutionary algorithms and then “breed” the most successful programs in new generations.


# Geoffrey Hinton Person

 Since 2013, he has divided his time working for Google (Google Brain) and the University of Toronto. In 2017, he co-founded and became the Chief Scientific Advisor of the Vector Institute in Toronto.

 With David Rumelhart and Ronald J. Williams, Hinton was co-author of a highly cited paper published in 1986 that popularised the [backpropagation] algorithm for training multi-layer neural networks, although they were not the first to propose the approach. Hinton is viewed as a leading figure in the deep learning community. The dramatic image-recognition milestone of the [AlexNet Model] designed in collaboration with his students [Alex Krizhevsky][Alex Krizhevsky Person] and [Ilya Sutskever][Ilya Sutskever Person] for the ImageNet challenge 2012 was a breakthrough in the field of computer vision.

 {% youtube "https://www.youtube.com/watch?v=qpoRO378qRY" %}

 More at:
  * [https://en.wikipedia.org/wiki/Geoffrey_Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton)

 See also [G], ...


# GINI Impurity Index

 A metrics to measure how diverse the data is in a dataset.
  * The more diverse the dataset is, the closer the GINI index is to 1 (but never equal to one)
  * The GINI index = 1 in the impossible case where all the elements in the dataset are different and the dataset is na infinite number of sample
  * The GINI index is 0 if all the samples in the dataset are the same (same label)

 {% youtube "https://www.youtube.com/watch?v=u4IxOk2ijSs" %}

 Questions:
  * Why is this important?

 See also [G], [Dataset], [Forest Of Stumps], [Weighted Gini Impurity Index]


# GitHub Company

 Offers code repositories. An acquisition of [Microsoft][Microsoft Company]

 See also [G], [GitHub Copilot]


# GitHub Copilot

 The [OpenAI Codex][Codex Model] integrated to [GitHub][GitHub Company] to suggest code and entire functions in real-time, right from your editor.

 Morea at:
  * [https://github.com/features/copilot](https://github.com/features/copilot)

 See also [G], ...

# GLIDE Model

 What adds the text conditioning to the diffusion model!?

 {% youtube "https://www.youtube.com/watch?v=lvv4N2nf-HU" %}

 {% pdf "{{site.assets}}/g/glide_model_paper.pdf" %}

 More at :
  * how does DALL-E work? - [https://www.assemblyai.com/blog/how-dall-e-2-actually-works/](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)
  * blog - [https://syncedreview.com/2021/12/24/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-173/](https://syncedreview.com/2021/12/24/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-173/)
  * paper - [https://arxiv.org/abs/2112.10741](https://arxiv.org/abs/2112.10741)
  * code - [https://github.com/openai/glide-text2im](https://github.com/openai/glide-text2im)
  * colab notebooks - [https://github.com/openai/glide-text2im/tree/main/notebooks](https://github.com/openai/glide-text2im/tree/main/notebooks)

 See also [G], [DALL-E Model]


# GloVe

 See also [G], [Word Embedding Space]


# Gluon

 This is ...


# Google Company

 Models developed by the company
  * [Bard][Bard Model]: A lightweight version of Lambda meant to compete against Bing + [ChatGPT Model]
  * [BERT][BERT Model]: 
  * [Google Lens][Google Lens]:
  * [Google Translate][Google Translate Model]:
  * [Gshard Model] :
  * [Imagen][Imagen Model]: A text-to-image diffusion model
  * [Imagen Video][Imagen Video Model]: A text-to-video model
  * [LaMBDA][LaMBDA Model]: A large language model
  * [Minerva Model] :
  * [MUM Model] :
  * [Muse Model] :
  * [MusicLM][MusicLM Model]: Generative model for music
  * [PaLM Model] :
  * [Pathways Model Architecture] :
  * [Switch Transformer] :
  * [T5][T5 Model]:
  * [Tensor Processing Unit] or TPU :
  * [Transformer Model] :

 Companies
  * [DeepMind][DeepMind Company] which built models such as [Chinchilla][Chinchilla Model], [Sparrow][Sparrow Model], [AlphaiFold][AlphaFold Model], and more ... 

 Projects
  * [Experiments](https://experiments.withgoogle.com/)
    * [AI Experiments](https://experiments.withgoogle.com/collection/ai)
      * [Quick Draw](https://quickdraw.withgoogle.com/)
  * [Teachable Machine](https://teachablemachine.withgoogle.com/)
    * Pose project - [https://medium.com/@warronbebster/teachable-machine-tutorial-head-tilt-f4f6116f491](https://medium.com/@warronbebster/teachable-machine-tutorial-head-tilt-f4f6116f491)
    * Image project - [https://medium.com/p/4bfffa765866](https://medium.com/p/4bfffa765866)
    * Audio project - [https://medium.com/p/4212fd7f3555(https://medium.com/p/4212fd7f3555)

 {% youtube "https://www.youtube.com/watch?v=X5iLF-cszu0&t=611s" %}

 See also [G], [Company]


# Google Lens

 Developed by [Google][Google Company], ...

 See also [G], ...


# Google Translate Model

 Google Translate is a multilingual neural machine translation service developed by Google to translate text, documents and websites from one language into another. It offers a website interface, a mobile app for Android and iOS, and an API that helps developers build browser extensions and software applications. As of April 2023, Google Translate supports 133 languages at various levels, and as of April 2016, claimed over 500 million total users, with more than 100 billion words translated daily, after the company stated in May 2013 that it served over 200 million people daily.

 Launched in April 2006 as a statistical machine translation service, it used United Nations and European Parliament documents and transcripts to gather linguistic data. Rather than translating languages directly, it first translates text to English and then pivots to the target language in most of the language combinations it posits in its grid, with a few exceptions including Catalan-Spanish. During a translation, it looks for patterns in millions of documents to help decide which words to choose and how to arrange them in the target language. Its accuracy, which has been criticized on several occasions, has been measured to vary greatly across languages. In November 2016, Google announced that Google Translate would switch to a neural machine translation engine – Google Neural Machine Translation (GNMT) – which translates "whole sentences at a time, rather than just piece by piece. It uses this broader context to help it figure out the most relevant translation, which it then rearranges and adjusts to be more like a human speaking with proper grammar".

 {% youtube "https://www.youtube.com/watch?v=_GdSC1Z1Kzs" %}

 {% youtube "https://www.youtube.com/watch?v=TIG2ckcCh1Y" %}

 More at:
  * site - [https://translate.google.com/](https://translate.google.com/)
  * wikipedia - [https://en.wikipedia.org/wiki/Google_Translate](https://en.wikipedia.org/wiki/Google_Translate)

 See also [G], [Google Company]


# Gopher Model

 NLP model developed by [DeepMind][DeepMind Company]
 ~ 280 parameters with a transformer architecture. Was later optimized and resulted in the [Chinchilla Model]

 {% youtube "https://www.youtube.com/watch?v=nO653U-Pb5c" %}

 {% pdf "{{site.assets}}/g/gopher_model_paper.pdf" %}

 More at :
  * [https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval)

 See also [G], ...


# GPU Instance

 P2 and P3 instances on AWS.

 See also [G], [Graphical Processing Unit]


# GPU Technology Conference

# GTC

 Annual conference organized by [Nvidia][Nvidia Company]

 The last conference was help on 2023/03/20 to 23.

 Here are the highlights and the keynote.

 {% youtube "https://www.youtube.com/watch?v=tg332P3IfOU" %}

 {% youtube "https://www.youtube.com/watch?v=DiGB5uAYKAg" %}

 More at:
  * [https://www.nvidia.com/gtc/](https://www.nvidia.com/gtc/)

 See also [G], ...


# Grade School Math Dataset

# GSM8K Dataset

 GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers. The dataset is segmented into 7.5K training problems and 1K test problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem. It can be used for multi-step mathematical reasoning.

 More at
  * research and samples - [https://openai.com/research/solving-math-word-problems](https://openai.com/research/solving-math-word-problems)
  * paper - [https://paperswithcode.com/paper/training-verifiers-to-solve-math-word](https://paperswithcode.com/paper/training-verifiers-to-solve-math-word)
  * site - [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
  * dataset - [https://paperswithcode.com/dataset/gsm8k](https://paperswithcode.com/dataset/gsm8k)

 See also [G], [Dataset], [OpenAI Company], [PaLM Model]


# Gradient

 A gradient is the direction and magnitude calculated during the training of a neural network it is used to teach the network weights in the right direction by the right amount.

 See also [G], [Gradient Descent]


# Gradient Boosting

 Use gradient descent to minimize the loss function.

 See also [G], [Boosting], [Ensemble Method], [Gradient Bagging], [Weak Learner]


# Gradient Clipping

 A technique to prevent exploding gradients in very deep networks, usually in recurrent neural networks. Gradient Clipping is a method where the error derivative is changed or clipped to a threshold during backward propagation through the network, and using the clipped gradients to update the weights.
 Gradient Clipping is implemented in two variants:
  * Clipping-by-value
  * Clipping-by-norm

 More at: 
  * [https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48)
  * gradient clipping accelerate learning? - [https://openreview.net/pdf?id=BJgnXpVYwS](https://openreview.net/pdf?id=BJgnXpVYwS)

 See also [G], [Exploding Gradient Problem], [Gradient Descent], [Recurring Neural Network]


# Gradient Descent

 One of the shining successes in machine learning is the gradient descent algorithm (and its modified counterpart, stochastic gradient descent). Gradient descent is an iterative method for finding the minimum of a function. In machine learning, that function is typically the loss (or cost) function. "Loss" is simply some metric that quantifies the cost of wrong predictions. Gradient descent calculates the loss achieved by a model with a given set of parameters, and then alters those parameters to reduce the loss. It repeats this process until that loss can't substantially be reduced further. The final set of parameters that minimize the loss now define your fitted model. 

 {% youtube "https://www.youtube.com/watch?v=OkmNXy7er84" %}

 Gradient descent
  * stochastic gradient descent
  * momentum gradient
  * nesterov accelerated gradient
  * ada gradient
  * adadelta gradient
  * adaptive momentum estimation
  * nag ?
  * rmsprop ?

 {% youtube "https://www.youtube.com/watch?v=nhqo0u1a6fw" %}

 {% youtube "https://www.youtube.com/watch?v=4F0_V_0OO2Q" %}

 ![]( {{site.assets}}/g/gradient_descent.png ){: width="100%"}

 `Using gradient descent, you can find the regression line that best fit the sample using a loss function` (distance of sample from line). To do that, you can start with any line and calculate the loss function, as you move the parameter, the loss function become smaller and smaller indicating in which direction the parameter should go.

 :warning: On huge datasets, There are downsides of the gradient descent algorithm. Indeed for huge datasets, we need to take a huge amount of computation we make for each iteration of the algorithm. To alleviate the problem, possible solutions are:
  * batch data for lighter partial processing (= standard gradient descent) :warning: compute intensive for large datasets
  * mini-batch ... = uses all the data in batches (= best of all proposed alternatives?)
  * data sampling or stochastic gradient descent. (= lower compute required since we use only a few sample, but faster iteration with slower convergence st each step, but faster overall?)

 See also [G], [Activation Function], [Batch Gradient Descent], [Gradient Perturbation], [Learning Rate], [Loss Function], [Mini-Batch Gradient Descent], [Parameter], [Prediction Error], [Stochastic Gradient Descent]


# Gradient Perturbation

 Works for neural network only or where gradient descent is used. Better than output perturbation in that the noise is built in the model, so you can share the model at will ;-) Gradient perturbation, widely used for differentially private optimization, injects noise at every iterative update to guarantee differential privacy. Noise is included in the gradient descent!

 More at:
  * [https://www.ijcai.org/Proceedings/2020/431}(https://www.ijcai.org/Proceedings/2020/431)
  * deep learning with differential privacy paper - [https://arxiv.org/abs/1607.00133](https://arxiv.org/abs/1607.00133)
  * paper - [https://www.microsoft.com/en-us/research/publication/gradient-perturbation-is-underrated-for-differentially-private-convex-optimization/](https://www.microsoft.com/en-us/research/publication/gradient-perturbation-is-underrated-for-differentially-private-convex-optimization/)

 See also [G], [Differential Privacy], [Gradient Clipping], [Output Perturbation]


# Graph Convolutional Network

# GCN

 More at 
  * [https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780)

 See also [G], ...


# Graph Neural Network

# GNN

 Use graph relationship to ... GNNs are used to train predictive models on datasets such as:
  * Social networks, where graphs show connections between related people,
  * Recommender systems, where graphs show interactions between customers and items,
  * Chemical analysis, where compounds are modeled as graphs of atoms and bonds,
  * Cybersecurity, where graphs describe connections between source and destination IP addresses,
  * And more!

 Ex: Most of the time, these datasets are extremely large and only partially labeled. Consider a fraud detection scenario where we would try to predict the likelihood that an individual is a fraudulent actor by analyzing his connections to known fraudsters. This problem could be defined as a semi-supervised learning task, where only a fraction of graph nodes would be labeled (‘fraudster’ or ‘legitimate’). This should be a better solution than trying to build a large hand-labeled dataset, and “linearizing” it to apply traditional machine learning algorithms.

 More at:
  * [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)

 See also [G], [Entity Extraction], [GNN Edge-Level Task], [GNN Graph-Level Task], [GNN Node-Level Task], [Insufficient Data Algorithm], [Knowledge Graph], [Relation Extraction], [Scene Graph]


# Graph Neural Network Edge-Level Task

# GNN Edge-Level Task

 One example of edge-level inference is in image scene understanding. Beyond identifying objects in an image, deep learning models can be used to predict the relationship between them. We can phrase this as an edge-level classification: given nodes that represent the objects in the image, we wish to predict which of these nodes share an edge or what the value of that edge is. If we wish to discover connections between entities, we could consider the graph fully connected and based on their predicted value prune edges to arrive at a sparse graph.

 ![]( {{site.assets}}/g/graph_neural_network_edge_level_task.png ){: width="100%"}

 In (b), above, the original image (a) has been segmented into five entities: each of the fighters, the referee, the audience and the mat. (C) shows the relationships between these entities.

 ![]( {{site.assets}}/g/graph_neural_network_edge_level_task2.png ){: width="100%"}

 See also [G], [Graph Neural Network]


# Graph Neural Network Graph Level Task

# GNN Graph-Level Task

 In a graph-level task, our goal is to predict the property of an entire graph. For example, for a molecule represented as a graph, we might want to predict what the molecule smells like, or whether it will bind to a receptor implicated in a disease.

 ![]( {{site.assets}}/g/graph_neural_network_graph_level_task.png ){: width="100%"}

 This is analogous to image classification problems with MNIST and CIFAR, where we want to associate a label to an entire image. With text, a similar problem is sentiment analysis where we want to identify the mood or emotion of an entire sentence at once.

 See also [G], [Graph Neural Network]


# Graph Neural Network Node-Level Task

# GNN Node-Level Task

 Node-level tasks are concerned with predicting the identity or role of each node within a graph. A classic example of a node-level prediction problem is Zach’s karate club. The dataset is a single social network graph made up of individuals that have sworn allegiance to one of two karate clubs after a political rift. As the story goes, a feud between Mr. Hi (Instructor) and John H (Administrator) creates a schism in the karate club. The nodes represent individual karate practitioners, and the edges represent interactions between these members outside of karate. The prediction problem is to classify whether a given member becomes loyal to either Mr. Hi or John H, after the feud. In this case, distance between a node to either the Instructor or Administrator is highly correlated to this label.

 ![]( {{site.assets}}/g/graph_neural_network_node_level_task.png ){: width="100%"}

 On the left we have the initial conditions of the problem, on the right we have a possible solution, where each node has been classified based on the alliance. The dataset can be used in other graph problems like unsupervised learning.

 Following the image analogy, node-level prediction problems are analogous to image segmentation, where we are trying to label the role of each pixel in an image. With text, a similar task would be predicting the parts-of-speech of each word in a sentence (e.g. noun, verb, adverb, etc).

 See also [G], [Graph Neural Network]


# Graphical Processing Unit

# GPU

 To accelerate processing of data.

 The move of AI/ML from CPU to GPU was done when [AlexNet Model] solved the [ImageNet Large Scale Visual Recognition Challenge] in 09/30/2012

 ![]( {{site.assets}}/g/graphical_processing_unit_scaling.webp ){: width="100%"}

 More at:
  * [https://www.nature.com/articles/s41586-021-04362-w](https://www.nature.com/articles/s41586-021-04362-w)

 See also [G], [CPU], [Cuda Core], [Hyperparameter Optimization], [TPU]


# Greg Brockman Person

 {% youtube "https://www.youtube.com/watch?v=YtJEfTTD_Y4" %}

 More at:
  * ...

 See also [G], ...

# Grid Search

 It’s tricky to find the optimal value for hyperparameters. The simplest solution is to try a bunch of combinations and see what works best. This idea of creating a “grid” of parameters and just trying out all the possible combinations is called a Grid Search.

 :warning: Beware of combinatorial growth or the [curse of dimensionality]

 ![]( {{site.assets}}/g/grid_search.webp){: width="100%"}

 More at:
  * [https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d](https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d)

 See also [G], [Hyperparameter Optimization], [Random Search]

# Gshard Model

  * [https://arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668)

 See also [G], [Google Company], [Sparse Activation]
