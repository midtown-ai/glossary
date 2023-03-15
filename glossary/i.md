---
title: I
permalink: /i/

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


# I, Robot Movie

 I, Robot is a 2004 American science fiction action film directed by Alex Proyas. 
 In 2035, highly intelligent robots fill public service positions throughout the dystopian world, operating under three rules to keep humans safe. Detective Del Spooner (Smith) investigates the alleged suicide of U.S. Robotics founder Alfred Lanning (Cromwell) and believes that a human-like robot called Sonny (Tudyk) murdered him.

 {% youtube "https://www.youtube.com/watch?v=s0f3JeDVeEo" %}

 More at:
  * [https://en.wikipedia.org/wiki/I,_Robot_(film)](https://en.wikipedia.org/wiki/I,_Robot_(film))

 See also [AI Movie]

# IBM Company

 See also [I], [Company]


# IBM Watson

 IBM Watson is a question-answering computer system capable of answering questions posed in natural language, developed in IBM's DeepQA project by a research team led by principal investigator David Ferrucci. Watson was named after IBM's founder and first CEO, industrialist Thomas J. Watson.

 The computer system was initially developed to answer questions on the quiz show Jeopardy! and, in 2011, the Watson computer system competed on Jeopardy! against champions Brad Rutter and Ken Jennings, winning the first place prize of $1 million.

 {% youtube "https://www.youtube.com/watch?v=P18EdAKuC1U" %}

 {% youtube "https://www.youtube.com/watch?v=b2M-SeKey4o" %}

 More at:
  * site - [https://www.ibm.com/watson](https://www.ibm.com/watson)
  * wikipedia - [https://en.wikipedia.org/wiki/IBM_Watson](https://en.wikipedia.org/wiki/IBM_Watson)

 See also [I], [IBM Company]


# Image Analysis

 See also [I], [Amazon Recognition]


# Image Classifier

 A component that does image classification.

 See also [I], [Image Classification]


# Image Classification

 Convolutional Neural Network (resNet). Supervised algorithm.

 More at:
   * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)

 See also [I], [Convolutional Neural Network], [ResNET Model]


# Image Decoder

 See also [I], [Decoder], [Image Encoder]


# Image Encoder

 ~ data compression where the encoder compress the data (not always) Ex: 2 images of lions. Instead of a comparing pixel to pixel, first encode the image to extract similarities and then compare the similarities. The translation from pixels to similarities is done by an encoder. First, let’s call encoder the process that produce the “new features” representation from the “old features” representation (by selection or by extraction) and decoder the reverse process. Dimensionality reduction can then be interpreted as data compression where the encoder compress the data (from the initial space to the encoded space, also called latent space) whereas the decoder decompress them. Of course, depending on the initial data distribution, the latent space dimension and the encoder definition, this compression/representation can be lossy, meaning that a part of the information is lost during the encoding process and cannot be recovered when decoding.

 See also [I], [Encoder], [Image Decoder]


# Image inpainting

 Masking of an area of an image and having it reconstructed by going through an autoencoder.

 See also [I], [Masked Language Learning Model]


# Image Reconstruction

 ![]( {{site.assets}}/i/image_reconstruction.png ){: width="100%"}

 Above is a pipeline for image reconstruction. The input image is fed to Flamingo/BLIP to generate a caption, which is fed to DALL-E/SD to reconstruct an image. The generated image is compared with the input image using the CLIP image encoder in the embedding space. Each input image has human-annotated captions which can be used to evaluate the generated caption.

 See also [I], [BLIP Model], [CLIP Image Encoder], [Text Reconstruction]


# Imagen

 {% youtube "https://www.youtube.com/watch?v=qhtYPhPWCsI" %}

 {% pdf "{{site.assets}}/i/imagen_paper.pdf" %}

 More at:
   * [https://imagen.research.google/](https://imagen.research.google/)
   * [https://www.louisbouchard.ai/google-brain-imagen/](https://www.louisbouchard.ai/google-brain-imagen/)

 See also [I], [Latent Diffusion Model]


# ImageNet Dataset

 ~ 1.2 million images with 1000 label each (= used for supervised learning, or not!) !ImageNet is an image dataset organized according to the !WordNet hierarchy. Each meaningful concept in !WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in !WordNet; the majority of them are nouns (80,000+). In !ImageNet, we aim to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, we hope ImageNet will offer tens of millions of cleanly labeled and sorted images for most of the concepts in the !WordNet hierarchy. The !ImageNet project was inspired by two important needs in computer vision research. The first was the need to establish a clear North Star problem in computer vision. While the field enjoyed an abundance of important tasks to work on, from stereo vision to image retrieval, from 3D reconstruction to image segmentation, object categorization was recognized to be one of the most fundamental capabilities of both human and machine vision. Hence there was a growing demand for a high quality object categorization benchmark with clearly established evaluation metrics. Second, there was a critical need for more data to enable more generalizable machine learning methods. Ever since the birth of the digital era and the availability of web-scale data exchanges, researchers in these fields have been working hard to design more and more sophisticated algorithms to index, retrieve, organize and annotate multimedia data. But good research requires good resources. To tackle this problem at scale (think of your growing personal collection of digital images, or videos, or a commercial web search engine’s database), it was critical to provide researchers with a large-scale image database for both training and testing. The convergence of these two intellectual reasons motivated us to build !ImageNet.

 More at:
   * [https://image-net.org/challenges/LSVRC/index.php](https://image-net.org/challenges/LSVRC/index.php)

 See also [I], [Supervised Learning], [Transfer Learning], [WordNet Dataset]


# Imitation Learning

# IL

 In imitation learning (IL), an agent is given access to samples of expert behavior (e.g. videos of humans playing online games or cars driving on the road) and it tries to learn a policy that mimics this behavior. This objective is in contrast to reinforcement learning (RL), where the goal is to learn a policy that maximizes a specified reward function. A major advantage of imitation learning is that it does not require careful hand-design of a reward function because it relies solely on expert behavior data, making it easier to scale to real-world tasks where one is able to gather expert behavior (like video games or driving). This approach of enabling the development of AI systems by data-driven learning, rather than specification through code or heuristic rewards, is consistent with the key principles behind Software 2.0.

 More at:
  * [https://ai.stanford.edu/blog/learning-to-imitate/](https://ai.stanford.edu/blog/learning-to-imitate/)
  * [https://www.technologyreview.com/2022/11/25/1063707/ai-minecraft-video-unlock-next-big-thing-openai-imitation-learning/](https://www.technologyreview.com/2022/11/25/1063707/ai-minecraft-video-unlock-next-big-thing-openai-imitation-learning/) (blocked?)

 See also [I], [Adversarial Imitation Learning], [Behavioral Cloning], [IQ-Learn Model], [Learning Method], [Reinforcement Learning], [Software 2.0]


# Imputation

 A way to deal with missing/incomplete data. Instead of eliminating the data point, insert the average or another value to use the other attributes of the samples.

 See also [I], [Data Point]


# Inductive Reasoning

 Coming up with `rules to explain the current observation`. Sometimes the truth can be learned ;-)

 See also [I], [Deductive Reasoning], [Truth]


# Inference

 An inference means running your machine learning model on new data). A prediction/action/complex plan that is devised/based on acquired knowledge. That is based on deductive reasoning (sherlock holmes!).

 See also [I], [Inference Point]


# Inference Point

 An endpoint to connect to behind which your model is running.

 See also [I], [Model]


# Initialization

 Initialization (of clustering algorithm)

 See also [I], [Hyperparameter]


# Input Layer

 See also [I], [Neural Network]


# Input Space

 Ex: raw pixel values. After training, the last layer of the model has captured the important patterns of the input that are needed for the image classification task. In the latent space, images that depict the same object have very close representations. Generally, the distance of the vectors in the latent space corresponds to the semantic similarity of the raw images. Below, we can see how the latent space of an animal classification model may seem. The green points correspond to the latent vector of each image extracted from the last layer of the model. We observe that vectors of the same animals are closer to the latent space. Therefore, it is easier for the model to classify the input images using these feature vectors instead of the raw pixel values:

 ![]( {{site.assets}}/i/input_space.png ){: width="100%"}

 See also [I], [Encoder], [Latent Space], [Latent Vector], [Word Embedded Space]


# Input Weight

 See also [I], [Artificial Neuron], [Backpropagation]


# Instance Segmentation

 Along with pixel level classification, we expect the computer to classify each instance of class separately. It is called instance segmentation.That is different instances of the same class are segmented individually in instance segmentation. Once an instance is given a name, it becomes an entity!

  ![]( {{site.assets}}/i/semantic_and_instance_segmentation.png ){: width="100%"}

 More at:
  * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)

 See also [I], [Convoluted Neural Network], [Entity Extraction], [Semantic Segmentation], [U-Net Architecture]


# InstructGPT Model

 A model that is a pre-trained GPT model and is fine tuned using reinforcement learning based on human feedback. A precursor of the chatGPT model. Large language models like GPT-3 are often used to follow instructions to execute user’s tasks. However, quite often, these models generate toxic or untruthful outputs that are not related to the input instructions. This is mostly due to the fact that models like GPT-3 are trained to predict the next word in a sentence rather than to execute a specific task. This is precisesly the problem OpenAI tried to address with InstructGPT, a language model that builds upon GPT-3 language capabilities but improves it its ability to follow instructions.

 {% pdf "{{site.assets}}/i/instructgpt_model_paper.pdf" %}

 The InstructGPT is build in three steps.
  1. The first step fine-tunes pretrained GPT-3 using 13k dataset. This dataset is from two sources:
    1. The team hired human labelers, who were asked to write and answer prompts — think NLP tasks. For example the human labeler was tasked to create an instruction and then multiple query & response pairs for it.
    1. The prompts by the end users in the Open.ai API, Playground. These prompts included various NLP tasks — text generation, Q&A, summarization etc.
   Supervised learning is used for the fine-tuning of the pretrained GPT-3. The dataset includes both inputs, but as well corresponding human labeled output.
  1. The second step and third step rely on reinforcement learning. Let’s first review the second step — the reward model.
    * The reward model is trained with 50k additional prompts. Prompt and multiple model outputs are generated. Model outputs are ranked by human from best to worse. The reward model is then trained to predict the human preferred output.
    * The third step is to optimize the policy using the reward model with 31k dataset. The data is purely from the Playground tool without any labeler written prompts. Therefore it differs from the first two steps.

  A prompt is generated. An output is generated by the policy. Reward is given for the output based on the reward model. The achieved reward is then used to optimize the policy using PPO algorithm.

 ![]( {{site.assets}}/i/instructgpt_model.png ){: width="100%"}

 There is a difference between the way the GPT-3 and the InstructGPT generate outputs. GPT-3 was designed to predict next token. This is important to keep in mind. Despite GPT-3 is able to predict the next word — the output could be unhelpful. Think for example toxic speech in end-user application. The misalignment refers in NLP — to the issue of outputs not matching user’s intent. `The InstructGPT is fine-tuned to human preference using reinforcement learning`. This means, that rather than just predicting next token, it tries instead to respond with an output — preferred by human labeler. The InstructGPT model is optimized differently from the GPT-3. It rewards human preference. Therefore it is better able to solve user tasks.

 More at:
  * paper - [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
  * blog post - [https://tmmtt.medium.com/the-instructgpt-e25797d8f4df](https://tmmtt.medium.com/the-instructgpt-e25797d8f4df)
  * gpt vs chatgpt vs instructgpt - [https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5](https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5)
  * openai post - [https://openai.com/blog/instruction-following/](https://openai.com/blog/instruction-following/)

 See also [I], [ChatGPT Model], [Digital Watermark], [GPT Model], [Proximal Policy Optimization], [Reinforcement Learning], [Reinforcement Learning Human Feedback], [Reward Model]


# Insufficient Data Algorithm

 Algorithms to deal with small datasets.

 See also [I], [Data Augmentation], [Graph Neural Network], [Meta Learning], [Multi-Task Leaning], [One-Shot Learning], [Transfer Learning], [Zero-Shot Learning]


# Intelligence

 (Prompt) What is intelligence?

 [ChatGPT Answer] :
  * Intelligence refers to the ability to learn, understand, and make judgments or have opinions that are reasonable.
  * It can also refer to the ability to acquire and apply knowledge and skills.
  * It is a complex trait that is influenced by both genetic and environmental factors.

 ![]( {{site.assets}}/i/intelligence.png ){: width="100%"}

 See also [I], [Artificial Intelligence]


# Intent Analysis

 See also [I], [Amazon Lex]


# InterpretML

 Developed by [Microsoft Company] as an open source project, InterpretML is “a toolkit to help understand models and enable responsible machine learning”. 

 More at:
  * [https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145](https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145)

 See also [I], ...


# Inverse Document Frequency

# IDF

 IDF measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:

 ```
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
 ```

 See also [I], [TF-IDF]


# Inverse Dynamics Model

# IDM

 OpenAI gathered 2,000 hours of video labeled with mouse and keyboard actions and trained an inverse dynamics model (IDM) to predict actions given past and future frames – this is the PreTraining part.
 
 See also [I], [Video Pre-Trained Model]


# Inverse Q-Learning

 See also [I], [Imitation Learning], [IQ-Learn Model]


# Inverse Reinforcement Learning

# IRL

 See also [I], [Behavioural Cloning], [Imitation Learning], [IQ-Learn Model], [Reinforcement Learning], [Reward Function]


# IQ-Learn Model

 {% pdf "{{site.assets}}/i/iqlearn_model_paper.pdf" %}

 More at:
  * blog - [https://ai.stanford.edu/blog/learning-to-imitate/](https://ai.stanford.edu/blog/learning-to-imitate/)
  * site - [https://div99.github.io/IQ-Learn/](https://div99.github.io/IQ-Learn/)
  * code - [https://github.com/Div99/IQ-Learn](https://github.com/Div99/IQ-Learn)

 See also [I], [Imitation Learning], [Inverse Q-Learning]


# Isaac Gym

 Physics based reinforcement learning environment

 More at :
  * [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)

 See also [I], [Nvidia Company], [Reinforcement Learning Environment]


# Isolation Forest

 The Isolation Forest works a bit differently than a Random Forest. It also creates a bunch of decision trees, but then it calculates the path length necessary to isolate an observation in the tree. The idea being that isolated observations, or anomalies, are easier to isolate because there are fewer conditions necessary to distinguish them from the normal cases. Thus, the anomalies will have shorter paths than normal observations and reside closer to the root of the tree.

 See also [I], [Ensemble Method]


# Iteration

 Each time a batch is processed is called an iteration. Note that the processing of the entire dataset, called an epoch, may require several iterations. This is particularly the case in the case of a large / very-large dataset.

 See also [I], [Batch], [Epoch]
