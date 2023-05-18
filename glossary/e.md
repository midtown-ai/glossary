---
title: E
permalink: /e/

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


# Early Stopping

 There is a challenge in training a [neural network] long enough for it to learn the mapping, but not so long that it overfits the training data. One way to accomplish this is to train on the training dataset, but to stop training at the point when performance on a validation dataset starts to degrade. In the world of training neural networks, this is what is known as “early stopping”. `A key challenge with [overfitting], and with machine learning in general, is that we can’t know how well our model will perform on new data until we actually test it"`.

 ![]( {{site.assets}}/e/early_stopping.png ){: width="100%"}

 See also [E], [AutoML], [Balanced Fitting], [Early Stopping Epoch], [Underfitting]


# Early Stopping Epoch

 ![]( {{site.assets}}/e/early_stopping_epoch.png ){: width="100%"}

 See also [E], [Early Stopping], [Overfitting], [Underfitting]


# Edge Detection

 Used for image [segmentation] and ...

 See also [E], ...


# Eigenvalue

 Each eigenvector has a eigenvalue

 The eigenvalue is the ratio by which the eigenvector is scaled during the linear transformation (matrix multiplication).

 If eigenvalue (v) is 
  * 0 < v : After transformation, the eigenvector keeps the same direction
  * v < 0 : After transformation, the eigenvector changes direction
  * 1 < v : After transformation, the eigenvector is stretched (elongated)
  * -1 < v < 1 : After transformation, the eigenvector is shrunken, scaled shorter

 ![]( {{site.assets}}/e/eigenvalue_definition.png ){: width="100%"}

 ![]( {{site.assets}}/e/eigenvalue_matrix.png ){: width="100%"}

 More at:
  * [https://setosa.io/ev/eigenvectors-and-eigenvalues/](https://setosa.io/ev/eigenvectors-and-eigenvalues/)

 See also [E], [Synthesized Variable]


# Eigenvector

 After a linear transformation (matrix multiplication), while every other vector deviates from their initial direction, the eigenvectors stay on the their original line despite the distortion from the matrix.

 :warning: but their length can be stretched or direction can be inverted (the opposite) (?), but the direction stays the same (?) before and after the transformation.

 :warning: All vectors on the same direction as the eigenvector is also an eigenvector, because their direction stays the same. The eigenvector is the one of unit length.

 :warning: Eigenvector for matrix A is probably not the eigenvector for matrix B

 :warning: A 2x2 matrix can have 0, 1, or 2 eigenvectors!

 During the transformation, each eigenvector is scaled during the linear transformation (matrix multiplication). That scaling factor is the [eigenvalue]!

 ![]( {{site.assets}}/e/eigenvector_definition.png ){: width="100%"}

 ![]( {{site.assets}}/e/eigenvector_matrix.png ){: width="100%"}

 More at:
  * [https://setosa.io/ev/eigenvectors-and-eigenvalues/](https://setosa.io/ev/eigenvectors-and-eigenvalues/)

 See also [E], [Eigenvalue], [Matrix], [Synthesized Variable]


# Elastic Net

 Used in [Regularization].

 {% pdf "https://hastie.su.domains/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf" %}

 More at:
  * paper - [https://hastie.su.domains/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf](https://hastie.su.domains/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf)
  * [https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/)
 
 See also [E], ...


# Electric Dreams Movie

 Electric Dreams is a 1984 science fiction romantic comedy film directed by Steve Barron (in his feature film directorial debut) and written by Rusty Lemorande. The film is set in San Francisco and depicts a love triangle among a man, a woman, and a personal computer. It stars Lenny Von Dohlen, Virginia Madsen, Maxwell Caulfield, and the voice of Bud Cort.

 {% youtube "https://www.youtube.com/watch?v=-aH39gQu-X8" %}

 {% youtube "https://www.youtube.com/watch?v=JhjyTE0f6tY" %}

 More at:
  * [https://en.wikipedia.org/wiki/Electric_Dreams_(film)](https://en.wikipedia.org/wiki/Electric_Dreams_(film))

 See also [E], [AI Movie]

# Elevenlabs AI Company

 An AI startup that lets anyone clone a target’s voice in a matter of seconds.

 {% youtube "https://www.youtube.com/watch?v=6eb3e9lJ-7s" %}

 Alternatives
  * [play.ht](https://play.ht/)
  * [coqui.ai](https://coqui.ai/about)

 More at:
  * [https://beta.elevenlabs.io/about](https://beta.elevenlabs.io/about)
  * [https://beta.elevenlabs.io/speech-synthesis](https://beta.elevenlabs.io/speech-synthesis)
  * [https://www.theverge.com/2023/1/31/23579289/ai-voice-clone-deepfake-abuse-4chan-elevenlabs](https://www.theverge.com/2023/1/31/23579289/ai-voice-clone-deepfake-abuse-4chan-elevenlabs)
  * [https://twitter.com/ramsri_goutham/status/1619620737509396483](https://twitter.com/ramsri_goutham/status/1619620737509396483)
  * [https://ramsrigoutham.medium.com/create-ai-powered-personalized-meditation-videos-d2f76fee03a5](https://ramsrigoutham.medium.com/create-ai-powered-personalized-meditation-videos-d2f76fee03a5)

 See also [E], ...


# Elon Musk Person

 Founder of Tesla, SpaceX, the Boring company, and early investor in [OpenAI]

 {% youtube "https://www.youtube.com/watch?v=iI_RNK4X2wg" %}

 See also [E], ...


# Embedding

 See [Word Embedding]


# Embedding Space

 A high dimensional semantic space.

 See also [E], [CLIP Model], [Embedding]


# Emergent Ability

> An ability is emergent if it is not present in smaller models but is present in larger models

 Emerging abilities include
  * [Theory Of Mind]
  * [Translation]
  * Word unscramble
  * Math word problems
  * Basic math operations
  * Instruction following

 Emergence is when quantitative changes in a system result in qualitative changes in behavior. An ability is emergent if it is not present in smaller models but is present in larger models. For example Theory of Mind would be an example of a spontaneous emergence of an ability in AI. As far as we know, OpenAI engineers did not deliberately implement ToM in GPT. Instead, ToM has emerged spontaneously as a byproduct of GPT being trained to achieve its task: Predict a next word in a sentence. This means that AI can develop surprising abilities without humans explicitly trying to design them. We should think about what abilities may come next! Finally, our study shows the usefulness of applying psychological methods to studying AI. AI models’ increasing complexity prevents us from understanding their functioning and deriving their capabilities directly from their design. This echoes the challenges faced by psychologists and neuroscientists in studying the original black box: the human brain. We hope that psychological science will help us to stay abreast of rapidly evolving AI.

 {% pdf "{{site.assets}}/e/emergent_abilities_of_large_language_models_paper.pdf" %}

 ![]( {{site.assets}}/e/emergent_abilities_of_large_language_models_table.png ){: width="100%"}

 See also [E], [GPT Model], [Large Language Model]


# Emotion

 Emotions are mental states brought on by neurophysiological changes, variously associated with thoughts, feelings, behavioral responses, and a degree of pleasure or displeasure. There is currently no scientific consensus on a definition. Emotions are often intertwined with mood, temperament, personality, disposition, or creativity.

Research on emotion has increased over the past two decades with many fields contributing including psychology, medicine, history, sociology of emotions, and computer science. The numerous attempts to explain the origin, function and other aspects of emotions have fostered intense research on this topic. Theorizing about the evolutionary origin and possible purpose of emotion dates back to Charles Darwin. Current areas of research include the neuroscience of emotion, using tools like PET and fMRI scans to study the affective picture processes in the brain.

 ![]( {{site.assets}}/e/emotion.png ){: width="100%"}

 More at:
  * [https://en.wikipedia.org/wiki/Emotion](https://en.wikipedia.org/wiki/Emotion)

 See also [E], [Emotional Intelligence], [Feeling]

# Emotional Intelligence

# EI

 Emotional intelligence (EI) is most often defined as the ability to perceive, use, understand, manage, and handle [emotions][Emotion]. People with high emotional intelligence can recognize their own emotions and those of others, use emotional information to guide thinking and behavior, discern between different feelings and label them appropriately, and adjust emotions to adapt to environments. Although the term first appeared in 1964, it gained popularity in the 1995 best-selling book Emotional Intelligence, written by science journalist Daniel Goleman.

 More at:
  * Book - [https://www.amazon.com/Emotional-Intelligence-Matter-More-Than/dp/055338371X](https://www.amazon.com/Emotional-Intelligence-Matter-More-Than/dp/055338371X)

 


 See also [E], [Affective Computing]


# Encoder
 
 ~ Let’s call encoder the process that produce the “new features” representation from the input or “old features” representation (by selection or by extraction) and decoder the reverse process. Dimensionality reduction can then be interpreted as data compression where the encoder compress the data (from the initial space to the encoded space, also called latent space) whereas the decoder decompress them. Of course, depending on the initial data distribution, the latent space dimension and the encoder definition, this compression/representation can be lossy, meaning that a part of the information is lost during the encoding process and cannot be recovered when decoding.

 ![]( {{site.assets}}/e/encoder_decoder.png ){: width="100%"}

 For each input, the encoder representation (hidden state) is up to the architecture of the model.

 {% youtube "https://www.youtube.com/watch?v=MUqNwgPjJvQ" %}

 Maybe the most famous encoder is BERT. To be useful, BERT needs to be matched with a classifier or a decoder.

 See also [E], [BERT Model], [Decoder], [Encoder Representation], [Hidden State], [Image Encoder], [One-Shot Learning], [Principal Component Analysis], [Encoder Representation], [Encoder Representation Space], [Image Encoder], [Similarity Function], [Variational Autoencoder], [Voice Encoder]


# Encoder Representation

 The output of the encoder given an input! The dimension on the output representation is smaller or equal to the dimension of the input. :warning: Same or smaller element count (i.e. length), but the dimension of output  elements can be larger than the dimension of an input element. Ex with NLP: input-element = word/token/integer ---> output-element = contextualised-word matrix/list (of length 768 in BERT) = meaning of word within the text .

 See also [E], [Encoder Representation Space]


# Encoder Representation Space

  * Latent Space
  * Word embedding space
  * Semantic space

 :warning: How do we chose the dimension in the representation space? WITH THE LOSS FUNCTION ! ===> |Y - Yest |, we set the ground truth of Y !!!!!

 See also [E], [Decoder Representation Space], [Latent Space], [Loss Function], [Semantic Space], [Word Embeddings Space]


# Encoder Stack
 
 A sequence of encoders when the output of one feeds on the following one.

 See also [E], [Decoder], [Decoder Stack], [Encoder], [Transformer Model]


# Encoder-Decoder Attention

 See also [E], [Attention-Based Model]


# Encoder-Decoder Model

 The best way to understand the concept of an encoder-decoder model is by playing Pictionary. The rules of the game are very simple, player 1 randomly picks a word from a list and needs to sketch the meaning in a drawing. The role of the second player in the team is to analyse the drawing and identify the word which it describes. In this example we have three important elements player 1(the person that converts the word into a drawing), the drawing (rabbit) and the person that guesses the word the drawing represents (player 2). This is all we need to understand an encoder decoder model.

 ![]( {{site.assets}}/e/encoder_decoder_model.jpeg ){: width="100%"}

 More at:
  * [https://towardsdatascience.com/what-is-an-encoder-decoder-model-86b3d57c5e1a](https://towardsdatascience.com/what-is-an-encoder-decoder-model-86b3d57c5e1a)

 See also [E], [Autoencoder], [Decoder], [Decoder Stack], [Encoder], [Encoder Stack], [Hidden State], [U-Net Architecture]


# Encoding

 Often used in data preparation to turn categorical features into numbers.

 Methods:
  * [One-Cold Encoding]
  * [One-Hot Encoding]
  * [Ordinal Encoding]

 See also [E], ...
# Endpoint

 After the model has been built, we create an endpoint in docker to make it available for queries. An endpoint has a URL which can be queried directly. `You don't have SSH access to the endpoint`.


# Engineered Arts Company

 The designer and manufacturer of the [Ameca Robot]

 Engineered Arts is an English engineering, designer and manufacturer of humanoid robots based in Cornwall, United Kingdom. It was founded in October 2004 by Will Jackson.

 More at:
  * [https://en.wikipedia.org/wiki/Engineered_Arts](https://en.wikipedia.org/wiki/Engineered_Arts)

 See also [E], ...


# Ensemble Method

 `Ensemble methods consist of joining several weak learners to build a strong learner.` ~ average the output of several models, such as decision trees?. Ex: aver Ensemble methods, meaning that they use a number of weak classifiers/learner to produce a strong classifier, which usually means better results. Imagine you’ve decided to build a bicycle because you are not feeling happy with the options available in stores and online. You might begin by finding the best of each part you need. Once you assemble all these great parts, the resulting bike will outshine all the other options.
 Ensemble methods use this same idea of combining several predictive models (supervised ML) to get higher quality predictions than each of the models could provide on its own. For example, the Random Forest algorithms is an ensemble method that combines many Decision Trees trained with different samples of the datasets. As a result, the quality of the predictions of a Random Forest is higher than the quality of the predictions estimated with a single Decision Tree. Think of ensemble methods as a way to reduce the variance and bias of a single machine learning model. That’s important because any given model may be accurate under certain conditions but inaccurate under other conditions. With another model, the relative accuracy might be reversed. By combining the two models, the quality of the predictions is balanced out. The great majority of top winners of Kaggle competitions use ensemble methods of some kind. The most popular ensemble algorithms are Random Forest, XGBoost and LightGBM. 

 ![]( {{site.assets}}/e/ensemble_method.png ){: width=20%}

 See also [E], [Gradient Bagging], [Gradient Boosting], [Isolation Forest], [LightGBM], [Random Forest], [Weak Leaner], [XGBoost]


# Entity

 A node in a knowledge graph.

 See also [E], [Knowledge Graph]


# Entity Extraction

 Extract entities from text or image to build a scene graph. Methods:
  * text input
   * Rule-based approach
   * Sequence labeling
   * Language models  <== recommended
  * image input
   * ? face detection ?
   * ? object detection?
   * ???

 See also [E], [Entity], [Name Entity Recognition], [Relation Extraction], [Scene Graph]


# Entropy

 . :warning: When a loss function hit the Shannon entropy, the model has learned everything there is to know, the model is predict everything as well as possible. So perfect algorithm and the model knows everything there is to know.

 Shannon entropy is a measure of the amount of uncertainty or randomness in a system. It was introduced by Claude Shannon in 1948 as a way to quantify the amount of information in a message or signal.

The entropy of a system is defined as the negative sum of the probabilities of each possible outcome multiplied by the logarithm of those probabilities. Mathematically, it can be expressed as:
 ```
H(X) = -∑(p(x) * log2 p(x))

# H(X) is the entropy of the system,
# p(x) is the probability of a particular outcome x,
# and log2 is the base-2 logarithm.
```

 The entropy is measured in bits, and it represents the minimum number of bits required to encode the information in the system. A system with high entropy has more uncertainty and randomness, and therefore requires more bits to encode the information. Conversely, a system with low entropy has less uncertainty and randomness, and requires fewer bits to encode the information.

 Shannon entropy has applications in various fields, including information theory, cryptography, and data compression. It is a fundamental concept in the study of communication and information processing.

 {% youtube "https://www.youtube.com/watch?v=YtebGVx-Fxw" %}

 More at:
  * [https://en.wikipedia.org/wiki/Entropy_(information_theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))

 See also [E], [Cross-Entropy], [Kullback-Liebler Divergence]


# Environment

 In [reinforcement learning], the environment provides states or observations of current state, and rewards/feeback.

 In a RL environment,
  * you cannot use [backpropagation] through an environment (Reward <-- Action <-- State) because too complicated and we cannot compute the derivative!
  * you cannot change its [parameters] as it is fixed!
  * but you can use the reward to signal (intensity and direction) to identify which  action is preferred over the others and update the agent's [policy] weights

 ![]( {{site.assets}}/e/environment.png ){: width="100%"}

 See also [E], [Isaac Gym], [PyBullet], [RobotSchool]


# Episode

 In [Reinforcement Learning], 

 we play a lot of episode to learn from the environment

  * Continuous task = no end
  * Episodic task = has at least one final state (time, goal, etc)

 See also [E], ...


# Epoch

 Epoch = Number of passes on the training dataset

 :warning: The parameters of the model are usually updated many times during each epoch. The model is updated after each batch of data. This is known as [batch training] or mini-batch training.

 One epoch means that every training sample has been fed through the model at least once. If your epochs are set to 50, for example, it means that the model you are training will work through the entire training dataset 50 times.

 `A complete pass on the dataset ~ 1 iteration!`. A complete dataset can be large in which case it is broken in [batches][Batch] and processed in many [iteration] (one per batch). Why use more than one Epoch? It may not look correct that passing the entire dataset through an ML algorithm or neural network is not enough, and we need to pass it multiple times to the same algorithm. So it needs to be kept in mind that to optimize the learning, we use gradient descent, an iterative process. Hence, it is not enough to update the weights with a single pass or one epoch. Moreover, one epoch may lead to [overfitting] in the model. In other words, when the training loop has passed through the entire training dataset once, we call that one epoch. Training for a higher number of epochs will mean your model will take longer to complete its training task, but it may produce better output if it has not yet converged.
  * Training over more epochs will take longer but can lead to a better output (e.g. sounding musical output)
  * Model training is a trade-off between the number of epochs (i.e. time) and the quality of sample output.

 Typically, the number of epochs is a hyperparameter that is set prior to training, and the model is trained for a fixed number of epochs. The optimal number of epochs depends on the complexity of the problem, the amount of training data, and other factors specific to the particular machine learning task.

 After training is complete, the model can be evaluated on a separate validation dataset to assess its performance. If the model is [overfitting] the training data, the validation loss will start to increase while the training loss continues to decrease, indicating that the model is starting to memorize the training data rather than learning to generalize to new data. In this case, [early stopping] or other regularization techniques can be used to prevent overfitting.

 See also [E], [Gradient Descent Algorithm]


# Eric Schmidt Person

 CEO of [Google] between 2001 and 2011, Chairman of Alphabet.

 {% youtube "https://www.youtube.com/watch?v=Sg3EchbCcA0" %}

 More at:
  * [https://en.wikipedia.org/wiki/Eric_Schmidt](https://en.wikipedia.org/wiki/Eric_Schmidt)

 See also [E], [People]


# Ernie Bot

 An alternative to [ChatGPT Model] developed by [Baidu]

 {% youtube "https://www.youtube.com/watch?v=ukvEUI3x0vI" %}

 More at:
  * [https://www.pcmag.com/news/openai-has-nothing-to-fear-from-chinas-chatgpt-rival-ernie-bot](https://www.pcmag.com/news/openai-has-nothing-to-fear-from-chinas-chatgpt-rival-ernie-bot)

 See also [E], ...


# Error

 See [Prediction Error]


# ESM Metagenomic Atlas

 The ESM Metagenomic Atlas will enable scientists to search and analyze the structures of [metagenomic proteins][Metagenomic Protein] at the scale of hundreds of millions of proteins. This can help researchers to identify structures that have not been characterized before, search for distant evolutionary relationships, and discover new proteins that can be useful in medicine and other applications.

 ![]( {{site.assets}}/e/esm_metagenomic_atlas.png ){: width="100%"}

 More at:
  * [https://ai.facebook.com/blog/protein-folding-esmfold-metagenomics/](https://ai.facebook.com/blog/protein-folding-esmfold-metagenomics/)

 See also [E], [AlphaFold Model], [OpenFold Model]


# Estimator

 ~ a model to draw estimation from. `Estimators predict a value based on observed data`. Estimation is a statistical term for finding some estimate of unknown parameter, given some data. Point Estimation is the attempt to provide the single best prediction of some quantity of interest. Quantity of interest can be: A single parameter, A vector of parameters — e.g., weights in linear regression, A whole function.

 See also [E], [Function Estimation], [Point Estimator]


# Ethical AI

 Examples of ethical problems with AI
  * About the model
    * Bias
  * About the industry
    * Concentration
  * About use of the output
    * Fake media
    * Deep fake

 5 Key ethical concepts in [Responsible AI] :

 1. Accountability
  * When one is accountable, then one has to account for something
  * An accountability framework refers to the formal and codified system for holding someone accountable within a specific social or organization context
  * It can include processes for managing or explaining decision making in an organization. It also involves establishing clear roles, responsibilities and transparency with respect to who is responsible for what and how to rectify mistakes

 1. Fairness
  * When something is fair, then people are given what they are due or deserve
  * In law, fairness generally entails goals like non-discrimination, equality before the law, and procedural fairness.
  * Statistical notions of fairness vary as well, and it is not possible to satisfy simultaneous all these definitions of fairness
  * Many efforts to promote fairness in AI address the issue of [bias]
  * In [Responsible AI], fairness is a socio-technical concept

 1. Understanding
  * To understand means to grasp the core meaning or to comprehend
  * In the field of [Responsible AI], understanding is closely connected to [explainability][Explainable AI]
  * [Explainability][Explainable AI] means being able to explain the internal mechanisms of a system in human terms. While [explainability][Explainable AI] is necessary for understanding, it is not sufficient
  * Understanding is audience-specific and contextual
  * Understanding matters because our stakeholders need to understand how and why AI is being used to make decisions that impact their lives

 1. Data stewardship
  * Stewardship is an ethical concept that means responsible management or care
  * Data stewardship necessitates that ethical data practices are in place to not only ensure data quality, integrity, and protection, but also to preserve customer's right to privacy and meaningful, informed consent
  * Machine learning is driven by data
  * Data is tangible asset in the information-based economy and ethical data practices are an important element of [Responsible AI]

 1. Safety
  * Being safe means being protected from harm or threats
  * Commonly, people think of physical safety, but there are other forms of safety, such as emotional, psychological, and financial safety
  * Safety is also a cluster-concept in [Responsible AI]
  * From a technical perspective, safety often means that an AI system's outputs are accurate, secure from hostile actors, robust, and reliable in different settings and over time
  * From a social and ethical perspective, safety is more robust concept that includes consideration of how an AI system will affect people's well-being, rights, and interests, as well as the broader safety of social institutions and the natural environment

 {% youtube "https://www.youtube.com/watch?v=fc-cHk9yFpg" %}

 {% pdf "{{site.assets}}/e/ethical_ai_cards.pdf" %}

 More at:
  * [https://hbr.org/2020/10/a-practical-guide-to-building-ethical-ai](https://hbr.org/2020/10/a-practical-guide-to-building-ethical-ai)
  * ethics card - [https://www.ideo.com/blog/ai-needs-an-ethical-compass-this-tool-can-help](https://www.ideo.com/blog/ai-needs-an-ethical-compass-this-tool-can-help)
  * [https://scottaaronson.blog/?p=6823](https://scottaaronson.blog/?p=6823)
  * comic and copyright - [https://aibusiness.com/nlp/ai-generated-comic-book-keeps-copyright-well-some-of-it](https://aibusiness.com/nlp/ai-generated-comic-book-keeps-copyright-well-some-of-it)
  * AAAI ethics and diversity - [https://aaai.org/about-aaai/ethics-and-diversity/](https://aaai.org/about-aaai/ethics-and-diversity/)

 See also [E], ...


# Evolutionary Scale Modeling

# ESM

 More at:
  * [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)

 See also [E], ...

# Ex Machina Movie

 Ex Machina is a 2014 science fiction film written and directed by Alex Garland in his directorial debut. 
 In the film, programmer Caleb Smith (Gleeson) is invited by his CEO (Isaac) to administer the Turing test to an intelligent humanoid robot (Vikander).

 {% youtube "https://www.youtube.com/watch?v=EoQuVnKhxaM" %}

 More at:
  * [https://en.wikipedia.org/wiki/Ex_Machina_(film)](https://en.wikipedia.org/wiki/Ex_Machina_(film)) 

 See also [E], [AI Movie]


# Example-Based Machine Translation

# EBMT

 Example-based machine translation (EBMT) is a method of [machine translation] often characterized by its use of a bilingual [corpus] with parallel texts as its main knowledge base at run-time. It is essentially a translation by analogy and can be viewed as an implementation of a [case-based reasoning] approach to [machine learning].

 More at:
  * [https://en.wikipedia.org/wiki/Example-based_machine_translation](https://en.wikipedia.org/wiki/Example-based_machine_translation)

 See also [E], ...


# Expected Value

 {% youtube "https://www.youtube.com/watch?v=KLs_7b7SKi4" %}

 See also [E], ...


# Experience Replay

 Experience replay, a common RL technique, used in [Deep Q-Networks] amongst others, is another in-between approach ([Offline learning] vs [Online Learning]). Although you could store all the experience necessary to fully train an agent in theory, typically you store a rolling history and sample from it. It's possible to argue semantics about this, but I view the approach as being a kind of "[buffered online learning]", as it requires low-level components that can work online (e.g. neural networks for [DQN]).

 More at:
  * [https://ai.stackexchange.com/questions/10474/what-is-the-relation-between-online-or-offline-learning-and-on-policy-or-off](https://ai.stackexchange.com/questions/10474/what-is-the-relation-between-online-or-offline-learning-and-on-policy-or-off)

 See also [E], [Actor-Critic with Experience Replay Algorithm]


# Experience Replay Buffer


 See also [E], ...


# Expert System

 In [artificial intelligence], an expert system is a computer system emulating the decision-making ability of a human expert. Expert systems are designed to solve complex problems by reasoning through bodies of knowledge, represented mainly as if–then rules rather than through conventional procedural code. The first expert systems were created in the 1970s and then proliferated in the 1980s. Expert systems were among the first truly successful forms of artificial intelligence (AI) software. An expert system is divided into two subsystems: the inference engine and the knowledge base. The knowledge base represents facts and rules. The inference engine applies the rules to the known facts to deduce new facts. Inference engines can also include explanation and debugging abilities.

 ![]( {{site.assets}}/e/expert_system_comparison.png ){: width="100%"}

 More at:
  * [https://en.wikipedia.org/wiki/Expert_system](https://en.wikipedia.org/wiki/Expert_system)

 See also [E], [Big Data], [Deep Learning], [Logical Reasoning], [Machine Learning], [Optimization], [Statistics]


# Explainable AI

# XAI

 As more and more companies embed AI and advanced analytics within a business process and automate decisions, there needs to have transparency into how these models make decisions grows larger and larger. How do we achieve this transparency while harnessing the efficiencies AI brings. This is where the field of Explainable AI (XAI) can help. 

 {% pdf "{{site.assets}}/e/explainable_ai_whitepaper.pdf" %}

 More at:
  * whitepaper - [https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf](https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf)
  * [https://towardsdatascience.com/what-is-explainable-ai-xai-afc56938d513](https://towardsdatascience.com/what-is-explainable-ai-xai-afc56938d513)
  * [https://en.wikipedia.org/wiki/Explainable_artificial_intelligence#](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence#)

 See also [E], [Black Box Model], [White Box Model]


# Explainability Spectrum

 ![]( {{site.assets}}/e/explainability_spectrum.png ){: width="100%"}

 See also [E], [Chain Of Thought Prompting]


# Explanatory Variable

 We will refer to inputs as features, and the phenomena they represent as explanatory variables. Other names for explanatory variables include "predictors", "regressors", "controlled variables", and "exposure variables".

 See also [E], [Feature], [Response Variable]


# Exploding Gradient Problem

 Activation that are large trends to become larger and larger! The solution for this is to use activation functions such as the sigmoid or the tanh ones. Unfortunately using such activation function leads to the vanishing gradient problem experienced during backpropagation. Another solution is to use gradient clipping in backpropagation.

 See also [E], [Activation Function], [Gradient Clipping], [Recurrent Neural Network], [Vanishing Gradient Problem]


# Exploitation

 In [Reinforcement Learning]

 allocate rest of budget to invest in slot

 See also [E], ... 


# Exploration

 In [Reinforcement Learning]

 allocate certain amount of money to find the most profitable slots

 See also [E], ... 

# Expressiveness

 See also [E], [Hyperparameter]


# Extreme Gradient Boosting

# XGBoost

 An ensemble method, XGBoost (extreme gradient boosting) is a popular and efficient open-source implementation of the gradient-boosted trees algorithm. Gradient boosting is a machine learning algorithm that attempts to accurately predict target variables by combining the estimates of a set of simpler, weaker models (several decision trees?). By applying gradient boosting to decision tree models in a highly scalable manner, XGBoost does remarkably well in machine learning competitions. It also robustly handles a variety of data types, relationships, and distributions. It provides a large number of hyperparameters—variables that can be tuned to improve model performance. This flexibility makes XGBoost a solid choice for various machine learning problems such as classifications and regressions. Example image recognition of a car: Before you recognize the car, does the thing have wheels, are they door, etc... if it has all of those features then it must be a car.

 ![]( {{site.assets}}/e/extreme_gradient_boosting.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=XXHhrlL-FWc" %}

 More at:
  * ...

 See also [E], [Bagging], [Boosting], [Classification], [Ensemble Method], [Hyperparameters], [Machine Learning], [Random Forest], [Ranking], [Regression]
