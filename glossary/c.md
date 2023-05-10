---
title: C
permalink: /c/

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


# Caffe

# Caffe2

 PyTorch is not an end-to-end machine learning development tool; the development of actual applications requires conversion of the PyTorch code into another framework such as Caffe2 to deploy applications to servers, workstations, and mobile devices.

 More at:
   * [https://viso.ai/deep-learning/pytorch-vs-tensorflow/](https://viso.ai/deep-learning/pytorch-vs-tensorflow/)

 See also [C], [Deep Learning Framework]


# Carnegie Mellon University

# CMU

 {% youtube "https://www.youtube.com/watch?v=krGtcHHGmpk" %}

 More at:
  * [https://ai.cs.cmu.edu/](https://ai.cs.cmu.edu/)
  * research - [https://ai.cs.cmu.edu/research](https://ai.cs.cmu.edu/research)

 See also [C], ...


# Case-Based Reasoning

# CBR

 In artificial intelligence and philosophy, case-based [reasoning] (CBR), broadly construed, is the process of solving new problems based on the solutions of similar past problems.

 In everyday life, an auto mechanic who fixes an engine by recalling another car that exhibited similar symptoms is using case-based [reasoning]. A lawyer who advocates a particular outcome in a trial based on legal precedents or a judge who creates case law is using case-based [reasoning]. So, too, an engineer copying working elements of nature (practicing biomimicry), is treating nature as a database of solutions to problems. Case-based reasoning is a prominent type of analogy solution making.

 It has been argued that case-based reasoning is not only a powerful method for computer reasoning, but also a pervasive behavior in everyday human problem solving; or, more radically, that all reasoning is based on past cases personally experienced.

 See also [C], ...


# Casual Language Modeling

 Based on the context (previous words) find out the most likely following work. One that word is found, the new word is used to estimate the next one.

 See also [C], [Autoregressive Model], [Decoder], [GPT Model], [Natural Language Generation]


# CatBoost Library

 CatBoost is a machine learning method based on [Gradient Boosting] over [Decision Trees][Decision Tree].

 Main advantages:
  * Superior quality when compared with other GBDT libraries on many datasets.
  * Best in class prediction speed.
  * Support for both numerical and categorical features.
  * Fast GPU and multi-GPU support for training out of the box.
  * Visualization tools included.
  * Fast and reproducible distributed training with Apache Spark and CLI.

 More at:
  * [https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145](https://towardsdatascience.com/9-awesome-python-packages-for-machine-learning-that-should-deserve-more-credit-dbad17263145)
  * [https://github.com/catboost/catboost](https://github.com/catboost/catboost)
  * [https://github.com/catboost/tutorials/#readme](https://github.com/catboost/tutorials/#readme) - tutorials

 See also [C], ...


# Categorical Variable

 A variable that takes discrete non-numerical value, such as a shirt size (XS, S, M, L, XL) or gender (M, F). Because computer works with numbers, to be processed categorical variables are normally turned into discrete variables.

 See also [C], [Discrete Variable], [Variable Type]


# Cell Block

 A cell in jupyter!


# Central Limit Theorem

 In probability theory, the central limit theorem (CLT) establishes that, in many situations, when independent random variables are summed up, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed.

 See also [C], [Gaussian Distribution]


# Chain Of Thought Prompting

 A solution to get the explainability of a model OR RATHER ITS OUTPUT! Generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even fine-tuned GPT-3 with a verifier.

 ![]( {{site.assets}}/c/chain_of_thought_prompting.png ){:width="100%"}

 See also [C], [Explanability], [GPT Model], [Natural Language Programming], [PaLM Model]


# Chained Model

 Each model does one thing. e.g. verifier.

 See also [C], [Model]


# Chatbot

 See [Virtual Assistant]


# ChatGPT Model

 A [GPT model] that has a state,  that is you can have a discussion/dialog with the device. This model is [fine-tuned with "supervised" interactions][Supervised Fine-Tuning] as was done with the [InstructGPT model], a precursor to ChatGPT. In recent weeks, the internet has been going crazy with the new ChatGPT model. In general, ChatGPT is part of a series of releases around [GPT 3.5][GPT Model] that are highlighting some of the capabilities of the upcoming [GPT-4 model][GPT Model]. One of the key differences of ChatGPT with previous models is its ability to follow instructions. This is powered another model called InstructGPT which OpenAI quietly unveiled at the beginning of the year.

 ![]( {{site.assets}}/c/chatgpt_to_1m_users.jpeg ){: width="100%"}

 ![]( {{site.assets}}/c/chatgpt_to_100m_users.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=AsFgn8vU-tQ" %}

 {% youtube "https://www.youtube.com/watch?v=pOmpqdlVCoo" %}

 More at:
  * adoption rate - [https://www.linkedin.com/pulse/chatgpts-100m-users-2-months-more-impressive-than-you-gilad-nass/](https://www.linkedin.com/pulse/chatgpts-100m-users-2-months-more-impressive-than-you-gilad-nass/)
  * gpt vs chatgpt vs instructgpt - [https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5](https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5)
  * webgpt chrome extension - [https://twitter.com/DataChaz/status/1610556519531089921](https://twitter.com/DataChaz/status/1610556519531089921)
  * [https://www.cnn.com/2022/12/05/tech/chatgpt-trnd/index.html](https://www.cnn.com/2022/12/05/tech/chatgpt-trnd/index.html)
  * [https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5](https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5)
  * [https://www.technologyreview.com/2023/01/26/1067299/chatgpt-workout-plans/](https://www.technologyreview.com/2023/01/26/1067299/chatgpt-workout-plans/)

 See also [C], [Chatbot], [Digital Watermark], [Feedback], [Plagiarism Checker], [Reward Model], [Reinforcement Learning], [Sparrow Model]


# Child Development Milestone

 Skills such as taking a first step, smiling for the first time, and waving “bye bye” are called developmental milestones. Children reach milestones in how they play, learn, speak, act, and move.

 More at:
  * [https://www.cdc.gov/ncbddd/actearly/milestones/index.html](https://www.cdc.gov/ncbddd/actearly/milestones/index.html)
  * [https://www.autonomousagents.stanford.edu/modeling-human-learning-and-develop](https://www.autonomousagents.stanford.edu/modeling-human-learning-and-develop)

 See also [C], ...
 

# Chinchilla Model

 An optimized model of [Goopher][Gopher Model]. Achieved the same performance with fewer parameters!

```
FLOPS   Params    Nb_of_tokens     ==> Performance
set       <search space>               set
```

 {% youtube "https://www.youtube.com/watch?v=PZXN7jm9IC0" %}

 {% pdf "https://arxiv.org/pdf/2203.15556.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

 See also [C], ...


# Chroma Database

 An in-memory [vector datatabase] ...

 ```
import chromadb

client = chromadb.Client()

collection = client.create_collection("test")

collection.add(
    embeddings=[
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
    ],
    metadatas=[
        {"uri": "img1.png", "style": "style1"},
        {"uri": "img2.png", "style": "style2"},
        {"uri": "img3.png", "style": "style1"},
        {"uri": "img4.png", "style": "style1"},
        {"uri": "img5.png", "style": "style1"},
        {"uri": "img6.png", "style": "style1"},
        {"uri": "img7.png", "style": "style1"},
        {"uri": "img8.png", "style": "style1"},
    ],
    documents=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8"],
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
)

query_result = collection.query(
        query_embeddings=[[1.1, 2.3, 3.2], [5.1, 4.3, 2.2]],
        n_results=2,
    )

print(query_result)
```
 when run, it outputs
 ```
{'ids': [['id1', 'id5'], ['id2', 'id4']], 'embeddings': None, 'documents': [['doc1', 'doc5'], ['doc2', 'doc4']], 'metadatas': [[{'uri': 'img1.png', 'style': 'style1'}, {'uri': 'img5.png', 'style': 'style1'}], [{'uri': 'img2.png', 'style': 'style2'}, {'uri': 'img4.png', 'style': 'style1'}]], 'distances': [[0.0, 0.0], [11.960000038146973, 11.960000038146973]]}
 ```

 More at:
  * home - [https://www.trychroma.com/](https://www.trychroma.com/)
  * docs - [https://docs.trychroma.com/getting-started](https://docs.trychroma.com/getting-started)
  * colab 
    * [https://colab.research.google.com/drive/1QEzFyqnoFxq7LUGyP1vzR4iLt9PpCDXv](https://colab.research.google.com/drive/1QEzFyqnoFxq7LUGyP1vzR4iLt9PpCDXv)
    * [https://github.com/hwchase17/chroma-langchain](https://github.com/hwchase17/chroma-langchain)
  * Articles
    * [https://blog.langchain.dev/langchain-chroma/](https://blog.langchain.dev/langchain-chroma/)

 See also [C], ...


# CIDEr Score

 See also [C], [MSFT COCO Caption Dataset]


# CICERO Model

 A Model built by [Meta].

 Diplomacy is what AI researchers call a “seven player, zero sum and deterministic game of imperfect information”. A seven player game is much harder to solve than a two player game such as chess or Go. You must consider the many possible strategies of not one but six other players. This makes it much harder to write an AI to play the game. Diplomacy is also a game of imperfect information, because players make moves simultaneously. Unlike games such as chess or Go, where you know everything about your opponent’s moves, players in Diplomacy make moves not knowing what their opponents are about to do. They must therefore predict their opponents’ next actions. This also adds to the challenge of writing an AI to play it. Finally, Diplomacy is a zero sum game in which if you win, I lose. And the outcome is deterministic and not dependent on chance. Nonetheless, before victory or defeat, it still pays for players to form alliances and team up on each other. Indeed, one of the real challenges in playing the game is managing the informal negotiations with other players before making simultaneous moves. The main reason Cicero’s performance is a scientific breakthrough is that it can both play the game well, and also perform these informal negotiations. This combination of natural language processing and strategic reasoning is a first for any game-playing AI.

 More at:
   * site - [https://ai.facebook.com/research/cicero/](https://ai.facebook.com/research/cicero/)
   * [https://about.fb.com/news/2022/11/cicero-ai-that-can-collaborate-and-negotiate-with-you/](https://about.fb.com/news/2022/11/cicero-ai-that-can-collaborate-and-negotiate-with-you/)
   * [https://ai.facebook.com/blog/cicero-ai-negotiates-persuades-and-cooperates-with-people/](https://ai.facebook.com/blog/cicero-ai-negotiates-persuades-and-cooperates-with-people/)
   * science article (private) - [https://www.science.org/doi/10.1126/science.ade9097?fbclid=IwAR1is0uOvw8uSQaJjTNKeevCKanq3TnVsLiS2wY0RwHX3zreCuwqPHKTcVI](https://www.science.org/doi/10.1126/science.ade9097?fbclid=IwAR1is0uOvw8uSQaJjTNKeevCKanq3TnVsLiS2wY0RwHX3zreCuwqPHKTcVI)
   * science article (public) - [https://www.science.org/content/article/ai-learns-art-diplomacy-game?cookieSet=1](https://www.science.org/content/article/ai-learns-art-diplomacy-game?cookieSet=1)
   * request for proposal - [https://ai.facebook.com/research/request-for-proposal/towards-human-AI-cooperation/](https://ai.facebook.com/research/request-for-proposal/towards-human-AI-cooperation/)
   * gizmodo - [https://www.gizmodo.com.au/2022/11/an-ai-named-cicero-can-beat-humans-in-diplomacy-a-complex-alliance-building-game-thats-a-big-deal/](https://www.gizmodo.com.au/2022/11/an-ai-named-cicero-can-beat-humans-in-diplomacy-a-complex-alliance-building-game-thats-a-big-deal/)

 See also [C], ...


# CIFAR Dataset

 Datasets created by [Alex Krizhevsky][ALex Krizhevsky: Person] for [AlexNet][AlexNet Model]

 Datasets
  * CIFAR-10
  * CIFAR-100

 The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes/categories. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. Computer algorithms for recognizing objects in photos often learn by example. CIFAR-10 is a set of images that can be used to teach a computer how to recognize objects. Since the images in CIFAR-10 are low-resolution (32x32), this dataset can allow researchers to quickly try different algorithms to see what works. CIFAR-10 is a labeled subset of the 80 million tiny images dataset. When the dataset was created, students were paid to label all of the images.

 ![]( {{site.assets}}/c/cifar10.png ){: width="100%"}

 More at 
   * [https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/)
   * [https://maet3608.github.io/nuts-ml/tutorial/cifar10_example.html](https://maet3608.github.io/nuts-ml/tutorial/cifar10_example.html)

 See also [C], [Dataset]


# Classification

 A type of supervised learning algorithm. The goal in classification is to take input values and organize them into two or more categories. The categories are normally mutually exclusive (ex is this shape a circle, a rectangle or a triangle? Beware of 3-d shape projections, i.e. perspectives!). An example classification use case is fraud detection. In fraud detection, the goal is to take information about the transaction and use it to determine if the transaction is either fraudulent or not fraudulent. When XGBoost is given a dataset of past transactions and whether or not they were fraudulent, it can learn a function that maps input transaction data to the probability that transaction was fraudulent. Models used for classification:
  * Decision tree 
  * Logistic regression
  * Support Vector Machine (~ boundary zone in an hyperplane)

 ![]( {{site.assets}}/c/classification.png ){: width="100%"}

 See also [C], [Binary Classification], [Decision Tree], [Hyperplane], [Logistic Regression], [Multiclass Classification], [Supervised Learning], [Support Vector Machine], [XGBoost]


# Claude Model

 An [LLM] built by [Anthropic]

 {% youtube "https://www.youtube.com/watch?v=KB5r9xmrQBY" %}

 {% pdf "{{site.assets}}/c/claude_model_paper.pdf" %}

 More at:
  * cost estimate - [https://orenleung.com/anthropic-claude-next-cost](https://orenleung.com/anthropic-claude-next-cost)

 See also [C], ...


# CLIP Image Encoder

 Encode an image into the embedding space.

 See also [C], [CLIP Model], [Embedding Space], [Encoder]


# CLIP Text Encoder

 Encode a text prompt into the embedding space.

 See also [C], [CLIP Model], [Embedding Space], [Encoder]


# Clustering

 Ex: Clustering is also used by internet radio services; given a collection of songs, a clustering algorithm might be able to group the songs according to their genres. Using different similarity measures, the same clustering algorithm might group the songs by their keys, or by the instruments they contain.

 See also [C], [Initialization], [Unsupervised Learning]


# CNTK

 CNTK is ...


# Codex Model

 A model built by [OpenAI]

 See also [C], ...


# Cognosys AI Company

 Build a UI for their task-driven autonomous agent

 More at:
  * home - [https://www.cognosys.ai/](https://www.cognosys.ai/)
  * blgo - [https://www.cognosys.ai/blog](https://www.cognosys.ai/blog)

 See also [C], ...


# Cohere AI Company

 Cohere (stylized as co:here) is a Canadian startup that provides [Natural Language Processing] models that help companies improve human-machine interactions. Cohere was founded in 2019 by Aidan Gomez, Ivan Zhang, and Nick Frosst.

 More at:
  * [https://cohere.ai/](https://cohere.ai/)

 See also [C], [Company]


# Collaborative Filtering

 Used for recommendation of song/movies/etc where people mark what they like. If a person A has the same tastes as another person B, then what person B likes will be recommended to person A.

 ![]( {{site.assets}}/c/collaborative_filtering.png ){: width="100%"}

 The problem with this approach is that if a new song/movie is made available, it cannot be recommended! ( = Cold start problem )

 See also  


# Colossal Clean Crawled Corpus Dataset

# C4 Dataset

 To accurately measure the effect of scaling up the amount of pre-training, one needs a dataset that is not only high quality and diverse, but also massive. Existing pre-training datasets don’t meet all three of these criteria — for example, text from Wikipedia is high quality, but uniform in style and relatively small for our purposes, while the Common Crawl web scrapes are enormous and highly diverse, but fairly low quality.

 To satisfy these requirements, we developed the Colossal Clean Crawled Corpus (C4), a cleaned version of Common Crawl that is two orders of magnitude larger than Wikipedia. Our cleaning process involved deduplication, discarding incomplete sentences, and removing offensive or noisy content. This filtering led to better results on downstream tasks, while the additional size allowed the model size to increase without overfitting during pre-training. 

 More at:
  * [https://www.tensorflow.org/datasets/catalog/c4](https://www.tensorflow.org/datasets/catalog/c4)

 See also [C], [Corpus], [Dataset], [T5 Model]


# Company

 Example of companies are:

  * [Abnormal Security](https://abnormalsecurity.com/) - Email cyberattack detection ([Forbes](https://www.forbes.com/companies/abnormal-security/?sh=4c992843536a))
  * [Adept AI] - AI model developer. Focus on [AGI] through automation of tasks on desktops ([Forbes](https://www.forbes.com/companies/adept/?list=ai50&sh=7eb9ebdf4ebb))
  * [Adobe] - Offer a midjourey alternative called [Adobe Firefly]
  * [AlphaSense](https://www.alpha-sense.com/) - Focus on market intelligence search ([Forbes](https://www.forbes.com/companies/alphasense/?list=ai50&sh=298a91255f19))
  * [Amazon] - Focus on public cloud and partnership with 3rd parties
  * [Anduril Industries](https://www.anduril.com/) - Focus on defense software with AI ([Forbes](https://www.forbes.com/companies/anduril-industries/?list=ai50&sh=3edae6e7e083))
  * [Anthropic] - Focus on LLM, building an alternative to [GPT models][GPT Model] ([Forbes](https://www.forbes.com/companies/anthropic/?list=ai50&sh=2cdb4d2fb94e))
  * [Apple] - Large company in the world, strangely not so influential in the AI industry. At least not yet!
  * [Arize AI](https://arize.com/) - Focus on data issue detection ([Forbes](https://www.forbes.com/companies/arize-ai/?list=ai50&sh=72ba67871f36))
  * [Baidu] - Chinese search engine
  * [Bayesian Health](https://www.bayesianhealth.com/) - Focus on patient risk detection ([Forbes](https://www.forbes.com/companies/bayesian-health/?list=ai50&sh=257bfdaa13ac))
  * [BHuman] - Focus on creating deepfake of yourself!
  * [Canvas](https://www.canvas.build/) - Focus on construction robots
  * [Character AI](https://beta.character.ai/) - Chatbot application
  * [Clari](https://www.clari.com/) - Focus on sales software
  * [Coactive AI](https://coactive.ai/) - Data labelling software
  * [Cohere AI] - Focus on NLP applications
  * [Cognosys AI] - Task driven autonomous agent, similar to [AutoGPT][AutoGPT Model]
  * [Databricks](https://www.databricks.com/) - Data storage and analytics
  * [DeepMind] - Focus on AI applications in science
  * [Descript](https://www.descript.com/) - Video and podcast editing
  * [Eightfold AI](https://eightfold.ai/) - Recruiting software
  * [ElevenLabs AI] - Focus on Text-to-speech rendition
  * [FarmWise Labs](https://farmwise.io/home) - Weeding tractors for farming
  * [Fermat](https://www.youtube.com/@fermat_ws/videos): Collaboration canvas with AI
  * [Futuri Media] - Content selection for media, such as tv and radio. Includes [RadioGPT]!
  * [GitHub] - Code repositories with advanced features including  AI pair programming with Codex
  * [Glean](https://glean.co/) - Internal workplace search
  * [Google] - Known for its search engine and ad-placement business model. Challenged by Microsoft
  * [Gong](https://www.gong.io/) - Sales software
  * [Got It AI](https://www.app.got-it.ai/) - ELMAR LLM for the enterprise with truth checker!
  * [Gretel AI](https://gretel.ai/) - 
  * [Harvey](ihttps://techcrunch.com/2022/11/23/harvey-which-uses-ai-to-answer-legal-questions-lands-cash-from-openai/) - Digital assistant for lawyers
  * [Hugging Face] - Open-source AI library, a model hub, with datasets, and space/UI with [Gradio][Gradio Module]
  * [Inflection AI] - A chatbot that listens and talks
  * [Insitro](https://www.insitro.com/) - Drug discovery
  * [Ironclad](https://ironcladapp.com/) - Legal contract management
  * [Jasper](https://www.jasper.ai/) - Copywriting software
  * [Meta] - Formerly known as Facebook with a strong focus on the multiverse and more recently on AI
  * [Microsoft] - One of the largest company in the world, leading changes with AI 
  * [Midjourney AI](https://www.midjourney.com/) - AI image generator
  * [MosaicML](https://www.mosaicml.com/) - AI model training tools
  * [Moveworks](https://www.moveworks.com/) - Automated IT support
  * [Neeva](https://neeva.com/) - Personalized search engine
  * [Neptune AI] -
  * [Neuralink] - A company lead by [Elon Musk] that is focusing on human [Brain] Machine Interfaces
  * [Nvidia] - The leading supplier of [GPU]
  * [OpenAI] - Focus on democratizing AI. Known for releasing [ChatGPT][ChatGPT Model]
  * [Pachama](https://pachama.com/) - Forestry satellite data analysis
  * [PathAI](https://www.pathai.com/) - Drug discovery and diagnosis
  * [PolyAI](https://poly.ai/) - Voice chatbots
  * [Quora] - A static Q&A internet site that is not offering an interface to chatbots though its interface, poe.
  * [RevComm](https://www.revcomm.co.jp/) - Voice analysis software ([Forbes](https://www.forbes.com/companies/revcomm/?sh=6c1080a7340c))
  * [Runway] - Focus on generative AI for images and now videos
  * [Scale AI] - Data labeling provider
  * [Shield AI](https://shield.ai/) - Autonomous defense software
  * [Slingshot Aerospace](https://slingshotaerospace.com/) - Space simulation software
  * [Snorkel AI](https://snorkel.ai/) - Data labeling software
  * [Stability AI] - Focus on [diffusion model] or image generation, adopted the open-source philosophy
  * [Supertranslate AI] - Focus on generating proper subtitles to videos
  * [Synthesia] - Focus on AI avatars
  * [Tome](https://beta.tome.app/) - Presentation creation software
  * [Trigo](https://www.trigoretail.com/) - Cashierless retail checkout
  * [Unlearn.AI](https://www.unlearn.ai/) - Clinical trial forecasting
  * [Vannevar Labs](https://www.vannevarlabs.com/) - Defense intelligence software
  * [Vectra AI](https://www.vectra.ai/) - Cyberattack detection
  * [VIZ.AI](https://www.viz.ai/) - Disease detection
  * [Waabi](https://waabi.ai/) - Autonomous trucking technology
  * [Weights & Biases](https://wandb.ai/site) - Developer tools for AI
  * [Writer](https://writer.com/) - Copywriting software

 Drug Discovery
  * ...

 Driverless cars
  * [Cruise] - Self-driving, now robotaxis! 
  * [Waymo] - Focus on self-driving car 
  * others: [WeRide](https://www.weride.ai/en), [Momenta](https://www.momenta.cn/en/), [Didi](https://web.didiglobal.com/), [Pony AI](https://pony.ai)
  * defunct: [Argo AI](https://en.wikipedia.org/wiki/Argo_AI)

 Vector Databases:
  * [Chroma][Chroma Database] - in memory database (good for development!)
  * [Milvus][Milvus Database] - project supported by the [LFAI&Data]
  * [Pinecone] - building the [Pinecone Database]

 Synthetic data
  * [Datagen](https://datagen.tech/) - Synthetic data for faces/images
  * [Mostly AI](https://mostly.ai/) - Generative AI for tabular data 

 Robotics
  * [Boston Dynamics] - Focus on robotics
  * [Engineered Arts] - Focus on social robots, such as [Ameca][Ameca Robot]
  * [Hanson Robotics] - Build humanoid for consumer, entertainment, service, healthcare, and research applications. 
  * [Softbank Robotics] - Focus on [social robots][Social Robot]

 Education
  * [Chegg](https://www.chegg.com) - AI assistant called [CheegMate](https://www.chegg.com/cheggmate)
  * [Duolingo](https://www.duolingo.com) - Learn a language with role play with [Duolingo Max](https://blog.duolingo.com/duolingo-max/)
  * [Khan Academy](https://www.khanacademy.org) - AI assistant called [Khanmigo](https://www.khanacademy.org/khan-labs)
  * [Pearson](https://www.pearson.com/en-us.html) - Not bot yet! Still in denial?

 Deployment:
  * [Netlify] -

 {% youtube "https://www.youtube.com/watch?v=62ptqfCcizQ" %}

 More at:
  * Forbes AI top 50:
    * 2023 - [https://www.forbes.com/lists/ai50/?sh=1f9472b1290f](https://www.forbes.com/lists/ai50/?sh=1f9472b1290f)
    * 2022 - [https://www.forbes.com/sites/helenpopkin/2022/05/06/ai-50-2022-north-americas-top-ai-companies-shaping-the-future/?sh=63dcbbdc34b5](https://www.forbes.com/sites/helenpopkin/2022/05/06/ai-50-2022-north-americas-top-ai-companies-shaping-the-future/?sh=63dcbbdc34b5) 
    * 2021 - [https://www.forbes.com/sites/alanohnsman/2021/04/26/ai-50-americas-most-promising-artificial-intelligence-companies/?sh=12d718d177cf](https://www.forbes.com/sites/alanohnsman/2021/04/26/ai-50-americas-most-promising-artificial-intelligence-companies/?sh=12d718d177cf)
  * CNBC disruptor 50:
    * 2023 - [https://www.cnbc.com/2023/05/09/these-are-the-2023-cnbc-disruptor-50-companies.html](https://www.cnbc.com/2023/05/09/these-are-the-2023-cnbc-disruptor-50-companies.html)
    * 2022 - [https://www.cnbc.com/disruptor-50-2022/](https://www.cnbc.com/disruptor-50-2022/)
    * 2021 - [https://www.cnbc.com/disruptor-50-2021/](https://www.cnbc.com/disruptor-50-2021/)

 See also [C], ...


# Complexity

 Complexity (of a model) is
  * Attention-based model: for input sequence of length N, each layer's output is NxN (all to all comparison) and therefore each layer is o(N^2) for sequence of length N

 See also [C], [Hyperparameter], [Attention-Based Model]


# Compliance Analysis

 This is ...


# Computer Vision

 * [Ibject Detection]
 * [Object Recognition]
 * [Object Tracking]
 
 See also [C], [Convoluted Neural Network], [OpenCV Library], [ResNET Model]


# Computer Vision and Pattern Recognition Conference

# CVPR Conference

 An [AI conference] related to [computer vision] and [pattern recognition]

 More at:
  * twitter - [https://twitter.com/cvpr/](https://twitter.com/cvpr/)
  * [https://cvpr2023.thecvf.com/](https://cvpr2023.thecvf.com/)
  * [https://cvpr2022.thecvf.com/](https://cvpr2022.thecvf.com/)
  * [https://cvpr2021.thecvf.com/](https://cvpr2021.thecvf.com/)

 See also [C], ...


# Concept Video

 Here is an example of concept video of a Knoledge Navigator that later became the [Siri Virtual Assistant]

 {% youtube "https://www.youtube.com/watch?v=-jiBLQyUi38" %}

 {% include vimeoPlayer.html id=25551192 %}

 More at:
  * [https://en.wikipedia.org/wiki/Knowledge_Navigator](https://en.wikipedia.org/wiki/Knowledge_Navigator)

 See also [C], ...


# Conditioning

 See also [C], [Diffusion Model], [Latent Diffusion Model]


# Conditional GAN

# CGAN

 In this GAN the generator and discriminator both are provided with additional information that could be a class label or any modal data. As the name suggests the additional information helps the discriminator in finding the conditional probability instead of the joint probability.

 ![]( {{site.assets}}/c/conditional_gan.jpeg ){: width="100%"}

 See also [C], [Generative Adversarial Network]


# Conditional Random Fields

 See also [C], [Discriminative Classifier]


# Confidence Interval

 A confidence interval is the range of values needed to match a confidence level for estimating the features of a complete population.

 ![]( {{site.assets}}/c/confidence_interval.png ){: width="100%"}

 See also [C], [Gaussian Distribution]


# Confusion Matrix

 A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted by the machine learning model. In the case where N=2 (true or false), it shows false and true positive as well as false and true negative.

 ![]( {{site.assets}}/c/confusion_matrix_binary_classifier.jpeg ){: width="100%"}

 :warning: Beware:
  * [precision] = [recall] !
  * [specificity] = precision if classes in binary classifier are inverted!
  * [precision] = true positive / total positive

 ![]( {{site.assets}}/c/confusion_matrix_multiclass_classifier.png ){: width="100%"}

 More at:
  * [https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5](https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5)

 See also [C], [Accuracy], [Classification], [Model Drift], [Prediction Error]


# Constraint Satisfaction Problem

 See also [C], [Variable Model]


# Consumer Electronic Show

# CES

 {% youtube "https://www.youtube.com/watch?v=CnwGrI6T7X0" %}

 See also [C], ...


# Continuous Action Space

 In [Reinforcement Learning], a non-finite set of [Actions][Action]. You define a minimum and a maximum, i.e. a range, for your parameters and the agent can select a value for that range automatically. 

 See also [C], ...


# Continuous Convolution

 See also [C], [Discrete Convolution]


# Continuous Variable

 A variable that can take any value, possibly within a range.

 See also [C], [Variable Type]


# Contrastive Language Image Pre-training Model

# CLIP Model

 CLIP is a [dual-encoder contrastive model] that was developed by [OpenAI] and released open-source in 01/05/2021

 CLIP is a vision-language model that aligns image and text representations into a shared embedding space.
 is trained on large-scale image and text pair datasets to obtain a unified representation of different representations for the same concept. For that, an image encoder and a text encoder separately map images and text into a high-dimensional space, and a distance-based loss is utilized to enforce representations of identical concepts in neighborhood regions.
 CLIP is another neural network that is able to determine how well a caption (or prompt) matches an image. In other words, CLIP is a neural network that efficiently learns visual concepts from [natural language supervision].

 {% youtube "https://www.youtube.com/watch?v=fGwH2YoQkDM" %}

 Model:
  * large scale learning - bigger transformer models for image and text embeddings
  * trained on 400 million (image, text) pairs using ConVIRT model trained fro scratch
  * pre-training method: predicting only which text as a whole is paired with which image and not the exact words of that text (contrastive, i.e. binary-classification task/approach for matching text to image)
  * Use vision transformer to reduce training time and required compute resources compared with ResNet model.

 ![]( {{site.assets}}/c/clip_model.png ){: width="100%"}

 {% pdf "https://arxiv.org/pdf/2103.00020.pdf" %}

 More at:
  * [https://openai.com/research/clip](https://openai.com/research/clip)
  * code - [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
  * paper - [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
  * [https://openai.com/blog/clip/](https://openai.com/blog/clip/)
  * Hierarchical Text-Conditional Image Generation with CLIP Latents (paper) - [https://arxiv.org/abs/2204.06125](https://arxiv.org/abs/2204.06125)
  * articles
   * [https://www.pinecone.io/learn/clip/](https://www.pinecone.io/learn/clip/)

 See also [C], [CLIP Image Encoder], [CLIP Text Encoder], [Embedding Space], [Vision Transformer], [VQGAN]


# Contrastive Learning

 {% youtube "https://www.youtube.com/watch?v=sftIkJ8MYL4" %}

 See also [C], [Contrastive Loss]


# Contrastive Loss Function

 See also [C], [Contrastive Learning], [Loss Function]


# Convolution

 In math, Convolution = Merging the shape of 2 functions together. Ex: Function that fires fireworks * smoke for 1 firework overtime = smoke in the air at a specific time ( = cumulative add all contribution of all firework)

 {% youtube "https://www.youtube.com/watch?v=QmcoPYUfbJ8" %}

 {% youtube "https://www.youtube.com/watch?v=acAw5WGtzuk" %}

 [Discrete convolution]

 {% youtube "https://www.youtube.com/watch?v=KuXjwB4LzSA" %}

 [Continuous convolution]

 .... 

 More at:
  * [https://en.wikipedia.org/wiki/Convolution](https://en.wikipedia.org/wiki/Convolution)

 See also [C], [Convolutional Neural Network], [Image Kernel]


# Convolution Autoencoder

 A CNN to latent space, and from latent space to a deconvolution neural network ?

 See also [C], [Convolution Neural Network], [Deconvolution Neural Network]


# Convolutional Layer

 In a CNN, each layer tries to recognize a different pattern = extract features.

 See also [C], [Convolutional Neural Network], [Fully Connected Layer], [Image Kernel], [Max Pooling Layer]


# Convolutional Neural Network

# CNN

 `Particularly useful for image analysis/processing` such as object recognition, image classification, semantic segmentation (object in image), artistic style transfer (filter on an image with the style of another image often a painting), meow generator (find cats in image?) . `The idea is that the pixel are not completely independent from the one surrounding them. CNN takes the surrounding pixel into consideration as well instead of just an independent pixel`. Use filter. Max Pooling layers (dimension reduction of outputs to downstream layers to convert a tensor into a vector). A succession of convolution-subsampling layers. Example: Does a pixel belongs to an object or not? .

 {% youtube "https://www.youtube.com/watch?v=FmpDIaiMIeA" %}

 ![]( {{site.assets}}/c/convolutional_neural_network.png ){: width="100%"}

 The hidden layers are designed to process the input in a way that optimizes for signal and image processing /recognition. ==> recognize features instead of pixel!

 ![]( {{site.assets}}/c/cnn_zone_matching.png ){: width="100%"}

 `When using kernel, we are implicitly saying that pixel outside of the kernel do not have an impact on ... This is where attention-based models may be better than CNN, where attention to other pixel in the image needs to be taken into consideration`

 More at:
  * [https://setosa.io/ev/image-kernels/](https://setosa.io/ev/image-kernels/)
  * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)
  * [https://medium.com/easyread/an-introduction-to-convolution-neural-network-cnn-for-a-beginner-88548e4b2a84](https://medium.com/easyread/an-introduction-to-convolution-neural-network-cnn-for-a-beginner-88548e4b2a84)

 See also [C], [Attention-Based Model], [Convolution], [Convolutional Layer], [Deconvolution Neural Network], [Fully Connected Layer], [Image Classification], [Image Kernel], [Instance Segmentation], [Latent Space], [Neural Network], [Object Detection], [Max Pooling Layer], [Rectified Linear Unit], [Region Based CNN], [Semantic Segmentation], [Subsampling]


# Convolutional Neural Network Feature Extractor

# CNN Feature Extractor


 When using a [CNN], ... [ResNet50 Architecture]

 {% pdf "{{site.assets}}/c/convolutional_neural_network_feature_extractor_paper.pdf" %}

 More at:
  * [https://medium.com/birdie-ai/how-to-use-cnns-as-feature-extractors-54c69c1d4bdf](https://medium.com/birdie-ai/how-to-use-cnns-as-feature-extractors-54c69c1d4bdf)

 See also [C}, ...


# Coral Hardware

 A [Tensor Processing Unit (TPU)][Tensor Processing Unit] compatible with any computer including the [Raspberry Pi Computer]

 {% youtube "https://www.youtube.com/watch?v=ydzJPeeMiMI" %}

 More at:
  * [https://www.amazon.com/Google-G950-01456-01-Coral-USB-Accelerator/dp/B07S214S5Y](https://www.amazon.com/Google-G950-01456-01-Coral-USB-Accelerator/dp/B07S214S5Y)
  * [https://coral.ai/projects/teachable-sorter#project-intro](https://coral.ai/projects/teachable-sorter#project-intro)
  * [https://teachablemachine.withgoogle.com/](https://teachablemachine.withgoogle.com/)

 See also [C], ...


# Coreference

 Understand the entities a speak refers to when he uses nouns, pronouns. ex I, You, my sister, your sister, etc Function of the speaker perspective.

 More at:
   * [https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30)

 See also [C], [NLP Benchmark]


# CoreML Format

 Format for ML models to load on devices made by [Apple]

 See also [C], ...


# CoreML Framework

 Switch UI programming language

 More at:
  * [https://developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml)

 See also [C], [CoreML Format], [CoreML Tool]


# CoreML Tool

 Tools to convert models to [CoreML Format], etc and integrate a model in [CoreML Framework]

 More at:
  * [https://coremltools.readme.io/docs](https://coremltools.readme.io/docs)

 See also [C], ...


# Corpus

 GPT-3 is pre-trained on a corpus of text from five datasets: Common Crawl, WebText2, Books1, Books2, and Wikipedia..
  * Colossal Clean Crawled Corpus (C4) : Used by T5 model
  * Common Crawl : The Common Crawl corpus (collection of texts) comprises petabytes of data including raw web page data, metadata, and text data collected over eight years of web crawling. OpenAI researchers use a curated, filtered version of this dataset.
  * Web2text : WebText2 is an expanded version of the WebText dataset, which is an internal OpenAI corpus created by scraping web pages of particularly high quality. To vet for quality, the authors scraped all outbound links from Reddit that received at least three karma (an indicator for whether other users found the link interesting, educational, or just funny). WebText2 contains 40 gigabytes of text from these 45 million links, over 8 million documents.
  * Book1 and Book2 : Books1 and Books2 are two corpora (plural of corpus) that contain the text of tens of thousands of books on various subjects.
  * Wikipedia : The Wikipedia corpus is a collection including all English-language articles from the crowdsourced online encyclopedia Wikipedia at the time of finalizing the GPT-3’s dataset in 2019. This dataset has roughly 5.8 million English articles.

 See also [C], [Dataset], [GPT], [Natural Language Processing]


# Correlation

 :warning: Correlation is not causation!

 Correlation refers to the statistical relationship between two variables. In other words, it measures the extent to which two variables are related to each other.

 A correlation can be positive or negative. A positive correlation means that the two variables move in the same direction. For example, if one variable increases, the other variable also tends to increase. A negative correlation means that the two variables move in opposite directions. For example, if one variable increases, the other variable tends to decrease.

 Correlation can be measured by a statistic called the [correlation coefficient], which ranges from -1 to +1.

 ![]( {{site.assets}}/c/correlation.png ){: width="100%"}

 More at:
  * [https://en.wikipedia.org/wiki/Correlation](https://en.wikipedia.org/wiki/Correlation)

 See also [C], ...


# Correlation Coefficient

 :warning: Correlation is not causation!

 In [Statistics], correlation or dependence is any statistical relationship, whether causal or not, between two random variables.

 | Correlation coefficient | Correlation strength | Correlation type |
 | -0.7 to -1   | Very strong | Negative |
 | -0.5 to -0.7 | Strong | Negative |
 | -0.3 to -0.5 | Moderate | Negative |
 | -0 to -0.3 | Weak | Negative |
 | 0 | None | Zero |
 | 0 to 0.3 | Weak | Positive |
 | 0.3 to 0.5 | Moderate | Positive |
 | 0.5 to 0.7 | Strong | Positive |
 | 0.7 to 1 | Very strong | Positive |

 ![]( {{site.assets}}/c/correlation_coefficient.png ){: width="100%"}

 More at:
  * [https://www.scribbr.com/statistics/correlation-coefficient/](https://www.scribbr.com/statistics/correlation-coefficient/)

 See also [C], ...

# Cost

 Cost vs Reward = minimize vs maximize

 Cost = negative reward

 See also [C], [REward]


# Cost Function

 See [Loss Function]


# CPU

 See also [C], [GPU], [Lambda], [Hyperparameter Optimization]


# CreateML Application

 An application to easily create ML models

 {% youtube "https://www.youtube.com/watch?v=49D0T8_O4OE" %}

 More at:
  * [https://developer.apple.com/videos/play/wwdc2022/110332/](https://developer.apple.com/videos/play/wwdc2022/110332/)

 See also [C], [CoreML Framework]


# Critical Assessment of Structure Prediction Challenge

# CASP Challenge

 Critical Assessment of Structure Prediction (CASP), sometimes called Critical Assessment of Protein Structure Prediction, is a community-wide, worldwide experiment for protein structure prediction taking place every two years since 1994. CASP provides research groups with an opportunity to objectively test their structure prediction methods and delivers an independent assessment of the state of the art in protein structure modeling to the research community and software users. Even though the primary goal of CASP is to help advance the methods of identifying protein three-dimensional structure from its amino acid sequence many view the experiment more as a “world championship” in this field of science. More than 100 research groups from all over the world participate in CASP on a regular basis and it is not uncommon for entire groups to suspend their other research for months while they focus on getting their servers ready for the experiment and on performing the detailed predictions.

 In December 2018, CASP13 made headlines when it was won by [AlphaFold][AlphaFold Model], an [artificial intelligence] program created by [DeepMind]. In November 2020, an improved version 2 of AlphaFold won CASP14. According to one of CASP co-founders John Moult, AlphaFold scored around 90 on a 100-point scale of prediction accuracy for moderately difficult protein targets. AlphaFold was made open source in 2021, and in CASP15 in 2022, while DeepMind did not enter, virtually all of the high-ranking teams used AlphaFold or modifications of AlphaFold.

 More at:
  * [https://en.wikipedia.org/wiki/CASP](https://en.wikipedia.org/wiki/CASP)

 See also [C], ...


# Cross-Attention

 Allow the decoder to access information from encoders to make better predictions. In text-to-image generation, through the cross-attention mechanism, the information of the text is fused to the visual feature vectors.

 See also [C], [Attention], [Latent Diffusion Model], [Self-Attention], [Transformer Model]


# Cross-Entropy

 ```
Likelihood of sequence = P(X) = product_of ( P(i=0,t,x_i/x_<i)

Cross_entropy = - log(P(X)) / t
 ```

 See also [C], [Cross-Entropy Loss Function], [Entropy], [Perplexity]


# Cross-Entropy Loss Function

 Frequently used as a loss function for neural networks. To understand it, you need to understand the following (and in that order!): Surprisal, Entropy, Cross-Entropy, Cross-Entropy Loss.
  * Surprisal:  “Degree to which you are surprised to see the result”. Now it's easy to digest my word when I say that I will be more surprised to see an outcome with low probability in comparison to an outcome with high probability. Now, if Pi is the probability of ith outcome then we could represent surprisal (s) as:

 ```
s = log ( 1 / Pi)
 ```

 ![]( {{site.assets}}/c/cross_entropy_loss_function_surprise_graph.png ){: width="100%"}

  * Entropy: Since I know surprisal for individual outcomes, I would like to know **surprisal for the event**. It would be intuitive to take a weighted average of surprisals. Now the question is what weight to choose? Hmmm…since I know the probability of each outcome, taking probability as weight makes sense because this is how likely each outcome is supposed to occur. This weighted average of surprisal is nothing but Entropy (e) and if there are n outcomes then it could be written as:

 ```
entropy = e = sum(0, n, Pi * log (1/Pi)
 ```

  * Cross-Entropy: Now, what if each outcome’s actual probability is Pi but someone is estimating probability as Qi. In this case, each event will occur with the probability of Pi but surprisal will be given by Qi in its formula (since that person will be surprised thinking that probability of the outcome is Qi). Now, weighted average surprisal, in this case, is nothing but cross-entropy(c) and it could be scribbled as:

 ```
cross-entropy = c = sum(0, n, Pi * log (1/Qi)
 ```

  Cross-entropy is always larger than entropy and it will be same as entropy only when Pi=Qi
  * Cross-Entropy Loss: In the plot below, you will notice that as estimated probability distribution moves away from actual/desired probability distribution, cross-entropy increases and vice-versa. Hence, we could say that minimizing cross-entropy will move us closer to actual/desired distribution and that is what we want. This is why we try to reduce cross-entropy so that our predicted probability distribution end up being close to the actual one. Hence, we get the formula of cross-entropy loss as:

 ```
cross-entropy loss = c = sum(0, n, Pi * log (1/Qi)

# And in the case of binary classification problem where we have only two classes, we name it as binary cross-entropy loss and above formula becomes:
binary cross-entropy loss = c = sum(0, 1, Pi * log (1/Qi) = Po * log(1/Qo) + (1-Po) * log(1/Q1) 
 ```

 ![]( {{site.assets}}/c/cross_entropy_loss_function_graph1.png ){: width="100%"}

 ![]( {{site.assets}}/c/cross_entropy_loss_function_graph2.png ){: width="100%"}
 
 This plot helps you visualize the cross-entropy between two distributions. The Red function represents a desired probability distribution, for simplicity a gaussian distribution is shown here. While the Orange function represents estimated probability distribution. The purple bar shows cross-entropy between these two distributions which is in simple words the area under the blue curve.
 More at
   * plot - [https://www.desmos.com/calculator/zytm2sf56e](https://www.desmos.com/calculator/zytm2sf56e)
   * [https://medium.com/@vijendra1125/understanding-entropy-cross-entropy-and-softmax-3b79d9b23c8a](https://medium.com/@vijendra1125/understanding-entropy-cross-entropy-and-softmax-3b79d9b23c8a)
   * [https://machinelearningmastery.com/cross-entropy-for-machine-learning/](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)

 See also [C], [Binary Cross-Entropy Loss Function], [Cross-Entropy], [Entropy], [Loss Function]


# Cross-Validation Sampling Method

 Cross-validation is a powerful preventative measure against overfitting. The idea is clever: Use your initial training data to generate multiple mini train-test splits. Use these splits to tune your model. In standard k-fold cross-validation, we partition the data into k subsets, called folds. Then, we iteratively train the algorithm on k-1 folds while using the remaining fold as the test set (called the “holdout fold”). Cross-validation allows you to tune hyperparameters with only your original training set. This allows you to keep your test set as a truly unseen dataset for selecting your final model.

 ![]( {{site.assets}}/c/kfold_cross_validation.png ){: width="100%"}

 More at:
   * 

 See also [C], [Dataset], [Development Subset], [Holdout Fold], [Resampling Method], [Testing Subset], [Training Subset]


# Cruise Company

 A robotaxi company

 More at:
  * [https://getcruise.com/](https://getcruise.com/)

 See also [C], ...


# Cubic Regression

 See also [C], [Non-Linear Regression], [Regression]

 Cuda Core 

 See also [C], [GPU]


# Cumulative Distribution Function

# CDF

 Graph or histogram reporting the probability that a function has reached this value or is below.

 See also [C], [Distribution]


# Curse of Dimensionality

 See also [C], [HPO]


# Custom Churn Prediction

 See also [C], [Regression], [Supervised Learning]


# Cycle Generative Adversarial Network

# Cycle GAN

 Image-to-image translation involves generating a new synthetic version of a given image with a specific modification, such as translating a summer landscape to winter. This opens up the possibility to do a lot of interesting tasks like photo-enhancement, image colorization, style transfer, season translation, object transfiguration, and generating photos from paintings, etc. Traditionally, training an image-to-image translation model requires a dataset comprised of paired examples. That is, a large dataset of many examples of input images X (e.g. summer landscapes) and the same image with the desired modification that can be used as an expected output image Y (e.g. winter landscapes). The requirement for a paired training dataset is a limitation. These datasets are challenging and expensive to prepare, e.g. photos of different scenes under different conditions. In many cases, the datasets simply do not exist, such as famous paintings and their respective photographs. As such, there is a desire for techniques for training an image-to-image translation system that does not require paired examples. Specifically, where any two collections of unrelated images can be used and the general characteristics extracted from each collection and used in the image translation process. For example, to be able to take a large collection of photos of summer landscapes and a large collection of photos of winter landscapes with unrelated scenes and locations as the first location and be able to translate specific photos from one group to the other. This is called the problem of unpaired image-to-image translation.

 ![]( {{site.assets}}/c/cyclegan_architecture.png ){: width="100%"}

 At first glance, the architecture of the CycleGAN appears complex. Let’s take a moment to step through all of the models involved and their inputs and outputs. Consider the problem where we are interested in translating images from summer to winter and winter to summer. We have two collections of photographs and they are unpaired, meaning they are photos of different locations at different times; we don’t have the exact same scenes in winter and summer.
  * Collection 1: Photos of summer landscapes.
  * Collection 2: Photos of winter landscapes.
 We will develop an architecture of two GANs, and each GAN has a discriminator and a generator model, meaning there are four models in total in the architecture. The first GAN will generate photos of winter given photos of summer, and the second GAN will generate photos of summer given photos of winter.
  * GAN 1: Translates photos of summer (collection 1) to winter (collection 2).
  * GAN 2: Translates photos of winter (collection 2) to summer (collection 1).
 Each GAN has a conditional generator model that will synthesize an image given an input image. And each GAN has a discriminator model to predict how likely the generated image is to have come from the target image collection. The discriminator and generator models for a GAN are trained under normal adversarial loss like a standard GAN model. We can summarize the generator and discriminator models from GAN 1 as follows:
  * Generator Model 1:
   * Input: Takes photos of summer (collection 1).
   * Output: Generates photos of winter (collection 2).
  * Discriminator Model 1:
   * Input: Takes photos of winter from collection 2 and output from Generator Model 1.
   * Output: Likelihood of image is from collection 2.
 So far, the models are sufficient for generating plausible images in the target domain but are not translations of the input image.

 ![]( {{site.assets}}/c/cyclegan_season_transfer.png ){: width="100%"}
 ![]( {{site.assets}}/c/cyclegan_style_transfer.png ){: width="100%"}
 ![]( {{site.assets}}/c/cyclegan_object_transfiguration.png ){: width="100%"}

 Beware:
  * input can be an image of frequencies that represent a voice and therefore can be used to change your voice!

 More at:
  * [https://machinelearningmastery.com/what-is-cyclegan/](https://machinelearningmastery.com/what-is-cyclegan/)

 See also [C], [Generative Adversarial Network], [Spectrogram Image]
