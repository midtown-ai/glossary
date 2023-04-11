---
title: A
permalink: /a/

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


# Accuracy

 ~ the percentage of samples correctly classified given a labelled (but possibly biased) dataset. Consider a classification task in which a machine learning system observes tumors and must predict whether they are malignant or benign. Accuracy, or the `fraction of instances that were classified correctly, is an intuitive measure of the program's performance`. While accuracy does measure the program's performance, it does not differentiate between malignant tumors that were classified as being benign, and benign tumors that were classified as being malignant. In some applications, the costs associated with all types of errors may be the same. In this problem, however, failing to identify malignant tumors is likely a more severe error than mistakenly classifying benign tumors as being malignant.

 ```
                 TP + TN
Accuracy = -------------------
            TP + TN + FP + FN

T = Correctly identified
F = Incorrectly identified
P = Actual value is positive (class A, a cat)
F = Actual value is negative (class B, not a cat, a dog)

TP = True positive (correctly identified as class A)
TN = True negative (correctly identified as class B)
FP = False Positive
FN = False negative
TP + TN + FP + FN = all experiments/classifications/samples
 ```

 More at:
  * [https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5](https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5)

 See also [A], [Confusion Matrix]


# Action

 In [Reinforcement Learning], ...

 See also [A], [Action Space]


# Action Space

 In [Reinforcement Learning], represents a set of actions.
  * [Discrete action space] : We can individually define each action.
  * [Continuous action space]

 See also [A], [Action]


# Action Transformer

 ```
At Adept, we are building the next frontier of models that can take actions in the digital world—that’s why we’re excited to introduce our first large model, Action Transformer (ACT-1).

Why are we so excited about this?

First, we believe the clearest framing of general intelligence is a system that can do anything a human can do in front of a computer. A foundation model for actions, trained to use every software tool, API, and webapp that exists, is a practical path to this ambitious goal, and ACT-1 is our first step in this direction.
 ```

 ```
“Adept’s technology sounds plausible in theory, [but] talking about Transformers needing to be ‘able to act’ feels a bit like misdirection to me,” Mike Cook, an AI researcher at the Knives & Paintbrushes research collective, which is unaffiliated with Adept, told TechCrunch via email. “Transformers are designed to predict the next items in a sequence of things, that’s all. To a Transformer, it doesn’t make any difference whether that prediction is a letter in some text, a pixel in an image, or an API call in a bit of code. So this innovation doesn’t feel any more likely to lead to artificial general intelligence than anything else, but it might produce an AI that is better suited to assisting in simple tasks.”“Adept’s technology sounds plausible in theory, [but] talking about Transformers needing to be ‘able to act’ feels a bit like misdirection to me,” Mike Cook, an AI researcher at the Knives & Paintbrushes research collective, which is unaffiliated with Adept, told TechCrunch via email. “Transformers are designed to predict the next items in a sequence of things, that’s all. To a Transformer, it doesn’t make any difference whether that prediction is a letter in some text, a pixel in an image, or an API call in a bit of code. So this innovation doesn’t feel any more likely to lead to artificial general intelligence than anything else, but it might produce an AI that is better suited to assisting in simple tasks.”
# https://techcrunch.com/2022/04/26/2304039/
 ```

 See also [A], [Reinforcement Learning], [Transformer Model]


# Active Learning

 `Pick the sample from which you will learn the most and have them labelled`. How to select those samples? But a model with a seed sample set, run data to the model, label the ones that have the most uncertainty.

 More at:
  * [https://www.datacamp.com/community/tutorials/active-learning](https://www.datacamp.com/community/tutorials/active-learning)

 See also [A], [Bayesian Optimization Sampling Method], [Passive Learning]


# Activation Function

 Activation functions are required to include non-linearity in the [artificial neural network] .

 Without activation functions, in a multi-layered neural network the [Decision Boundary] stays a line regardless of the [weight] and [bias] settings of each [artificial neuron]!

 There are several activation functions used in the fields. They are:
  * [Rectified Linear Unit (ReLU) function][Rectified Linear Unit Activation Function],
  * [LeakyReLU function][LeakyReLU Activation Function]
  * [Tanh function][Tanh Activation Function],
  * [Sigmoid function][Sigmoid Activation Function] : Sigmoid is great to keep a probability between 0 and 1, even if sample is an outlier based on the sample. (Ex: how long to slow down for a car, or will I hit the tree given the distance, but here car goes at 500km/h = an outlier)
  * [Softplus function][Softplus Activation Function]
  * [Step activation][Step Activation Function]

 ![]( {{site.assets}}/a/activation_functions.png){: width="100%" }

 ![]( {{site.assets}}/a/activation_functions_all_in_one.png){: width="100%" }

 {% youtube "https://www.youtube.com/watch?v=hfMk-kjRv4c" %}

 :warning: Note that for multi-layer neural networks that use of an activation function at each layer, the [backpropagation] computation leads to loss of information (forward for input and backward for weight computation) which is known as the [vanishing gradient problem].

 See also [A], [Batch Normalization], [Exploding Gradient Problem], [Gradient Descent], [Loss Function]


# Activation Step

 Last step in an [artificial neuron] before an output is generated.

 See also [A], ...


# Actor Critic With Experience Replay

# ACER

 A sample-efficient policy gradient algorithm. ACER makes use of a replay buffer, enabling it to perform more than one gradient update using each piece of sampled experience, as well as a Q-Function approximate trained with the Retrace algorithm.

 See also [A], [PPO], [Reinforcement Learning]


# Adaptive Boosting

# AdaBoost

 * AdaBoost combines a lot of "weak learners" to make classifications. The weak learners are almost always decision stumps.
 * Some stumps get more say (weight) in the classification than others
 * Each stump is made by taking the previous stump's mistakes into account

 {% youtube "https://www.youtube.com/watch?v=LsK-xG1cLYA" %}

 More at:
  * example - [https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/)
  * wikipedia - [https://en.wikipedia.org/wiki/AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
  * [https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe](https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe)

 See also [A], [Boosting], [Decision Stump], [Forest Of Stumps]


# Addiction

 {% youtube "https://www.youtube.com/watch?v=bwZcPwlRRcc" %}

 More at:
  * [https://www.yalemedicine.org/news/how-an-addicted-brain-works](https://www.yalemedicine.org/news/how-an-addicted-brain-works)

  See also [A], [Delayed Reward], [Reinforcement Learning], [Reward Shaping]


# Adept AI Company

 An [AI Company]

 More at:
  * [https://www.adept.ai/](https://www.adept.ai/)
  * [https://www.crunchbase.com/organization/adept-48e7](https://www.crunchbase.com/organization/adept-48e7)

 See also [A], ...


# Adversarial Attack

 In the last few years researchers have found many ways to break AIs trained using labeled data, known as supervised learning. Tiny tweaks to an AI’s input—such as changing a few pixels in an image—can completely flummox it, making it identify a picture of a sloth as a race car, for example. These so-called adversarial attacks have no sure fix.

 In 2017, Sandy Huang, who is now at DeepMind, and her colleagues looked at an AI trained via reinforcement learning to play the classic video game Pong. They showed that adding a single rogue pixel to frames of video input would reliably make it lose. Now Adam Gleave at the University of California, Berkeley, has taken adversarial attacks to another level.

 See also [Adversarial Policy], [Threat Model]


# Adversarial Imitation Learning

 {% youtube "https://www.youtube.com/watch?v=hmV4v_EnB0E" %}

 See also [A], [Imitation Learning]


# Adversarial Model

 See also [A], [State Model]


# Adversarial Policy

 By fooling an AI into seeing something that isn’t really there, you can change how things around it act. In other words, an AI trained using reinforcement learning can be tricked by weird behavior. Gleave and his colleagues call this an adversarial policy. It’s a previously unrecognized threat model, says Gleave.

 See also [A], [Adversarial Attack], [Threat Model]


# Affective Computing

 Affective computing is the study and development of systems and devices that can recognize, interpret, process, and simulate human affects. It is an interdisciplinary field spanning computer science, psychology, and cognitive science. While some core ideas in the field may be traced as far back as to early philosophical inquiries into emotion, the more modern branch of computer science originated with Rosalind Picard's 1995 paper on affective computing and her book Affective Computing published by MIT Press. One of the motivations for the research is the ability to give machines [Emotional Intelligence], including to simulate empathy. The machine should interpret the emotional state of humans and adapt its behavior to them, giving an appropriate response to those emotions.

 More at:
  * [https://en.wikipedia.org/wiki/Affective_computing](https://en.wikipedia.org/wiki/Affective_computing)

 See also [A], ...


# Agent

 A person, an animal, or a program that is free to make a decision or take an action. An agent has a goal.
 The agent is focusing on maximizing its cumulative reward.

 Examples:
  * DeepRacer, the goal of the program running on the car is to go around the track as fast as possible without getting out of the track.

 See also [Agent's Goal], [Cumulative Reward], [Reinforcement Learning]


# Agent's goal

 An Agent is set by using the appropriate reward and reward shaping.

 See also [A], ...


# AI Alignment

 {% youtube "https://www.youtube.com/watch?v=EUjc1WuyPT8" %}

 {% youtube "https://www.youtube.com/watch?v=cSL3Zau1X8g" %}

 {% youtube "https://www.youtube.com/watch?v=fc-cHk9yFpg" %}

 ```
# ChatGPT rules (that can easily be bypassed or put in conflict with clever prompt engineering!)
1. Provide helpful, clear, authoritative-sounding answers that satisfy human readers.
2. Tell the truth.
3. Don’t say offensive things.
 ```

 More at :
  * [https://scottaaronson.blog/?p=6823](https://scottaaronson.blog/?p=6823)
  * is RLHF flawed? - [https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the](https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the)
  * more videos
    * [https://www.youtube.com/watch?v=k6M_ScSBF6A](https://www.youtube.com/watch?v=k6M_ScSBF6A)

 See also [A], [AI Ethics]


# AI Artificial Intelligence Movie

 Released in 2001

 {% youtube "https://www.youtube.com/watch?v=oBUAQGwzGk0" %}

 {% youtube "https://www.youtube.com/watch?v=4VfxU0bOx-I" %}

 More at:
  * [https://en.wikipedia.org/wiki/A.I._Artificial_Intelligence ](https://en.wikipedia.org/wiki/A.I._Artificial_Intelligence)

 See also [A], [AI Movie]


# AI Avatar

 See also [A], [Deep Fake], [Synthesia Company]


# AI Book

 * [A Thousand Brains](https://www.gatesnotes.com/A-Thousand-Brains)

  See also [A], [AI Company], [AI Movie]


# AI Ethics

 See [Ethical AI]


# AI Moratorium

 A group of prominent individuals in the AI industry, including Elon Musk, Yoshua Bengio, and Steve Wozniak, have signed an open letter calling for a pause on the training of AI systems more powerful than GPT-4 for at least six months due to the "profound risks to society and humanity" posed by these systems. The signatories express concerns over the current "out-of-control race" between AI labs to develop and deploy machine learning systems that cannot be understood, predicted, or reliably controlled.

 Alongside the pause, the letter calls for the creation of independent regulators to ensure future AI systems are safe to deploy, and shared safety protocols for advanced AI design and development that are audited and overseen by independent outside experts to ensure adherence to safety standards. While the letter is unlikely to have any immediate impact on AI research, it highlights growing opposition to the "ship it now and fix it later" approach to AI development.

> AI systems with human-competitive intelligence can pose profound risks to society and humanity, as shown by extensive research and acknowledged by top AI labs. As stated in the widely-endorsed Asilomar AI Principles, Advanced AI could represent a profound change in the history of life on Earth, and should be planned for and managed with commensurate care and resources. Unfortunately, this level of planning and management is not happening, even though recent months have seen AI labs locked in an out-of-control race to develop and deploy ever more powerful digital minds that no one – not even their creators – can understand, predict, or reliably control.
>
> Contemporary AI systems are now becoming human-competitive at general tasks, and we must ask ourselves: Should we let machines flood our information channels with propaganda and untruth? Should we automate away all the jobs, including the fulfilling ones? Should we develop nonhuman minds that might eventually outnumber, outsmart, obsolete and replace us? Should we risk loss of control of our civilization? Such decisions must not be delegated to unelected tech leaders. Powerful AI systems should be developed only once we are confident that their effects will be positive and their risks will be manageable. This confidence must be well justified and increase with the magnitude of a system's potential effects. OpenAI's recent statement regarding artificial general intelligence, states that "At some point, it may be important to get independent review before starting to train future systems, and for the most advanced efforts to agree to limit the rate of growth of compute used for creating new models." We agree. That point is now.
>
> Therefore, we call on all AI labs to immediately pause for at least 6 months the training of AI systems more powerful than GPT-4. This pause should be public and verifiable, and include all key actors. If such a pause cannot be enacted quickly, governments should step in and institute a moratorium.
>
> AI labs and independent experts should use this pause to jointly develop and implement a set of shared safety protocols for advanced AI design and development that are rigorously audited and overseen by independent outside experts. These protocols should ensure that systems adhering to them are safe beyond a reasonable doubt. This does not mean a pause on AI development in general, merely a stepping back from the dangerous race to ever-larger unpredictable black-box models with emergent capabilities.
>
> AI research and development should be refocused on making today's powerful, state-of-the-art systems more accurate, safe, interpretable, transparent, robust, aligned, trustworthy, and loyal.
>
> In parallel, AI developers must work with policymakers to dramatically accelerate development of robust AI governance systems. These should at a minimum include: new and capable regulatory authorities dedicated to AI; oversight and tracking of highly capable AI systems and large pools of computational capability; provenance and watermarking systems to help distinguish real from synthetic and to track model leaks; a robust auditing and certification ecosystem; liability for AI-caused harm; robust public funding for technical AI safety research; and well-resourced institutions for coping with the dramatic economic and political disruptions (especially to democracy) that AI will cause.
>
> Humanity can enjoy a flourishing future with AI. Having succeeded in creating powerful AI systems, we can now enjoy an "AI summer" in which we reap the rewards, engineer these systems for the clear benefit of all, and give society a chance to adapt. Society has hit pause on other technologies with potentially catastrophic effects on society.  We can do so here. Let's enjoy a long AI summer, not rush unprepared into a fall.

 {% youtube "https://www.youtube.com/watch?v=eJGxIH73zvQ" %}

 {% youtube "https://www.youtube.com/watch?v=BxkVmLPq79k" %}

 {% youtube "https://www.youtube.com/watch?v=8OpW5qboDDs" %}

 {% youtube "https://www.youtube.com/watch?v=BY9KV8uCtj4" %}

 More at:
  * the letter - [https://futureoflife.org/open-letter/pause-giant-ai-experiments/](https://futureoflife.org/open-letter/pause-giant-ai-experiments/)
  * FAQ after letter - [https://futureoflife.org/ai/faqs-about-flis-open-letter-calling-for-a-pause-on-giant-ai-experiments/](https://futureoflife.org/ai/faqs-about-flis-open-letter-calling-for-a-pause-on-giant-ai-experiments/)

 See also [A], ...


# AI Movie

  * 1927 - [Metropolis][Metropolis Movie]
  * 1956 - [Forbidden Planet][Forbidden Planet Movie]
  * 1968 - [2001 Space Odyssey][2001 Space Odyssey Movie]
  * 1983 - [WarGames][WarGames Movie]
  * 1984 - [Electric Dreams][Electric Dreams Movie]
  * 1984 - [Terminator][Terminator Movie]
  * 1986 - [Short Circuit][Short Circuit movie]
  * 1999 - [The Matrix][The Matrix Movie]
  * 2001 - [AI Artificial Intelligence][AI Artificial Intelligence Movie]
  * 2004 - [I, Robot][I, Robot Movie]
  * 2008 - [Wall-E][Wall-E Movie]
  * 2013 - [Her][Her Movie]
  * 2014 - [Ex Machina][Ex Machina Movie]
  * 2022 - [M3GAN][M3GAN Movie]: the eponymous artificially intelligent doll who develops self-awareness and becomes hostile toward anyone who comes between her and her human companion

 See also [A]


# AI Paper

 See [AI Research]


# AI Principle

 Discussed for the first time by the AI community at the Asilomar AI conference themed Beneficial AI 2017.

 Research Issues
  1. Research Goal: The goal of AI research should be to create not undirected intelligence, but beneficial intelligence.
  1. Research Funding: Investments in AI should be accompanied by funding for research on ensuring its beneficial use, including thorny questions in computer science, economics, law, ethics, and social studies, such as:
   * How can we make future AI systems highly robust, so that they do what we want without malfunctioning or getting hacked?
   * How can we grow our prosperity through automation while maintaining people’s resources and purpose?
   * How can we update our legal systems to be more fair and efficient, to keep pace with AI, and to manage the risks associated with AI?
   * What set of values should AI be aligned with, and what legal and ethical status should it have?
  1. Science-Policy Link: There should be constructive and healthy exchange between AI researchers and policy-makers.
  1. Research Culture: A culture of cooperation, trust, and transparency should be fostered among researchers and developers of AI.
  1. Race Avoidance: Teams developing AI systems should actively cooperate to avoid corner-cutting on safety standards.

 Ethics and Values
  1. Safety: AI systems should be safe and secure throughout their operational lifetime, and verifiably so where applicable and feasible.
  1. Failure Transparency: If an AI system causes harm, it should be possible to ascertain why.
  1. Judicial Transparency: Any involvement by an autonomous system in judicial decision-making should provide a satisfactory explanation auditable by a competent human authority.
  1. Responsibility: Designers and builders of advanced AI systems are stakeholders in the moral implications of their use, misuse, and actions, with a responsibility and opportunity to shape those implications.
  1. Value Alignment: Highly autonomous AI systems should be designed so that their goals and behaviors can be assured to align with human values throughout their operation.  1. Human Values: AI systems should be designed and operated so as to be compatible with ideals of human dignity, rights, freedoms, and cultural diversity.
  1. Personal Privacy: People should have the right to access, manage and control the data they generate, given AI systems’ power to analyze and utilize that data.
  1. Liberty and Privacy: The application of AI to personal data must not unreasonably curtail people’s real or perceived liberty.
  1. Shared Benefit: AI technologies should benefit and empower as many people as possible.
  1. Shared Prosperity: The economic prosperity created by AI should be shared broadly, to benefit all of humanity.
  1. Human Control: Humans should choose how and whether to delegate decisions to AI systems, to accomplish human-chosen objectives.
  1. Non-subversion: The power conferred by control of highly advanced AI systems should respect and improve, rather than subvert, the social and civic processes on which the health of society depends.
  1. AI Arms Race: An arms race in lethal autonomous weapons should be avoided.

 Longer-term issues
  1. Capability Caution: There being no consensus, we should avoid strong assumptions regarding upper limits on future AI capabilities.
  1. Importance: Advanced AI could represent a profound change in the history of life on Earth, and should be planned for and managed with commensurate care and resources.
  1. Risks: Risks posed by AI systems, especially catastrophic or existential risks, must be subject to planning and mitigation efforts commensurate with their expected impact.
  1. Recursive Self-Improvement: AI systems designed to recursively self-improve or self-replicate in a manner that could lead to rapidly increasing quality or quantity must be subject to strict safety and control measures.
  1. Common Good: Superintelligence should only be developed in the service of widely shared ethical ideals, and for the benefit of all humanity rather than one state or organization.

 Video of the Asilomar conference in 2017

 {% youtube "https://www.youtube.com/watch?v=h0962biiZa4" %}

 More at:
  * AI principles - [https://futureoflife.org/open-letter/ai-principles/](https://futureoflife.org/open-letter/ai-principles/)
  * Asilomar conference - [https://futureoflife.org/event/bai-2017/](https://futureoflife.org/event/bai-2017/)
  * Entire conference playlist - [https://www.youtube.com/playlist?list=PLpxRpA6hBNrwA8DlvNyIOO9B97wADE1tr](https://www.youtube.com/playlist?list=PLpxRpA6hBNrwA8DlvNyIOO9B97wADE1tr)

 See also [A], ...


# AI Research

 Research labs:
  * Individuals
   * Sander Dieleman at [DeepMind][Deepmind Company] - [https://sander.ai/research/](https://sander.ai/research/)
  * Universities
   * [Berkeley](https://ml.berkeley.edu/research)
   * [Stanford AI Lab](https://ai.stanford.edu/blog/)
  * For profit
   * Facebook - [https://ai.facebook.com/blog/](https://ai.facebook.com/blog/)
   * Google 
    * [https://research.google/](https://research.google/)
    * cloud-AI - [https://research.google/teams/cloud-ai/](https://research.google/teams/cloud-ai/)
    * Blog - [https://blog.google/technology/ai/](https://blog.google/technology/ai/)
  * Non-profit
    * [Eleuther AI](https://blog.eleuther.ai/)

 When to start research?
  * Look at the business impact
  * Make sure that stakeholders are engaged, because problems are not always well formulated or data is missing

 See [A], ...


# AI Safety

 See [AI Alignment]


# AI Winter

 More at:
  * iFirst - [https://en.wikipedia.org/wiki/History_of_artificial_intelligence#The_first_AI_winter_1974%E2%80%931980](https://en.wikipedia.org/wiki/History_of_artificial_intelligence#The_first_AI_winter_1974%E2%80%931980)
  * Second - [https://en.wikipedia.org/wiki/History_of_artificial_intelligence#Bust:_the_second_AI_winter_1987%E2%80%931993](https://en.wikipedia.org/wiki/History_of_artificial_intelligence#Bust:_the_second_AI_winter_1987%E2%80%931993)

  See also [A], ...


# Alan Turing Person

 The inventory of the imitation game, aka the [Turing Test]

 See also [A], ...


# Alex Krizhevsky Person

 Built the [AlexNet Model], hence the name!

 More at:
  * [https://en.wikipedia.org/wiki/Alex_Krizhevsky](https://en.wikipedia.org/wiki/Alex_Krizhevsky)

 See also [A], ...


# AlexNet Model

 A Model that led to the rebirth of [artificial neural networks][Artificial Neural Network] using [Graphical Processing Units (GPU)][GPU].

 AlexNet is the name of a [convolutional neural network (CNN)][Convolutional Neural Network] architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor.

 AlexNet competed in the [ImageNet Large Scale Visual Recognition Challenge] on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower (better) than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of [graphics processing units (GPUs)][GPU] during training.

 {% youtube "https://www.youtube.com/watch?v=c_u4AHNjOpk" %}

 {% youtube "https://www.youtube.com/watch?v=Nq3auVtvd9Q" %}

 {% pdf "{{site.assets}}/a/alexnet_model_paper.pdf" %}

 More at:
  * [https://en.wikipedia.org/wiki/AlexNet](https://en.wikipedia.org/wiki/AlexNet)

 See also [A], ...

# Algorithmic

 A kind of [hyperparameter]. If test (!?) to select the best algorithm/approach to switch how the code function.



 See also [A], ...


# Alpaca Model

 Developed at [Stanford University]

 {% youtube "https://www.youtube.com/watch?v=xslW5sQOkC8" %}

 With LLaMA

 {% youtube "https://www.youtube.com/watch?v=PyZPyqQqkLE" %}

  See also [A], ...


# AlphaCode Model

 A LLM used for generating code. Built by the [DeepMind Company]. An alternative to the [Codex Model] built by [OpenAI][OpenAI Company]

 {% youtube "https://www.youtube.com/watch?v=t3Yh56efKGI" %}

 More at:
  * [https://www.deepmind.com/blog/competitive-programming-with-alphacode](https://www.deepmind.com/blog/competitive-programming-with-alphacode)

 See also [A], [Codex Model]


# AlphaFault

 [AlphaFold Model] does not know physics, but just do pattern recognition/translation.

 More at:
  * [https://phys.org/news/2023-04-alphafault-high-schoolers-fabled-ai.html](https://phys.org/news/2023-04-alphafault-high-schoolers-fabled-ai.html) 
  * paper - [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282689](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282689)

 See also [A], ...


# AlphaFold Model

 AlphaFold is an [artificial intelligence (AI)][AI] program developed by [DeepMind][DeepMind Company], a subsidiary of Alphabet, which performs predictions of protein structure. The program is designed as a deep learning system.

 AlphaFold AI software has had two major versions. A team of researchers that used AlphaFold 1 (2018) placed first in the overall rankings of the 13th [Critical Assessment of Structure Prediction (CASP)][CASP Challenge] in December 2018. The program was particularly successful at predicting the most accurate structure for targets rated as the most difficult by the competition organisers, where no existing template structures were available from proteins with a partially similar sequence.

 A team that used AlphaFold 2 (2020) repeated the placement in the CASP competition in November 2020. The team achieved a level of accuracy much higher than any other group. It scored above 90 for around two-thirds of the proteins in CASP's global distance test (GDT), a test that measures the degree to which a computational program predicted structure is similar to the lab experiment determined structure, with 100 being a complete match, within the distance cutoff used for calculating GDT.

 AlphaFold 2's results at CASP were described as "astounding" and "transformational." Some researchers noted that the accuracy is not high enough for a third of its predictions, and that it does not reveal the mechanism or rules of protein folding for the protein folding problem to be considered solved. Nevertheless, there has been widespread respect for the technical achievement.

 On 15 July 2021 the AlphaFold 2 paper was published at Nature as an advance access publication alongside open source software and a searchable database of species proteomes.

 But recently [AlphaFault] !

 {% pdf "{{site.assets}}/a/alphafold_model_paper.pdf" %}

 More at:
  * nature paper - [https://www.nature.com/articles/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2)
  * [https://en.wikipedia.org/wiki/AlphaFold](https://en.wikipedia.org/wiki/AlphaFold)
  * [https://alphafold.com/](https://alphafold.com/)
  * online database - [https://alphafold.ebi.ac.uk/faq](https://alphafold.ebi.ac.uk/faq)
  * colab - [https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)

 See also [A], [AlphaGo Model], [AlphaZero Model]


# AlphaFold Protein Structure Database

 The protein structure database managed by [DeepMind][DeepMind Company] where all the protein structures predicted by the [AlphaFold Model] are stored.

 More at:
  * [https://alphafold.com/](https://alphafold.com/)

 See also [A}, ...


# AlphaGo Model

 AlphaGo was built by [DeepMind][DeepMind Company]. AI to play GO. Used reinforcement learning.

 {% youtube "https://www.youtube.com/watch?v=WXuK6gekU1Y" %}

 More at:
  * ...

 See also [A], [AlphaFold Model], [AlphaZero Model], [DeepMind Company], [Reinforcement Learning]


# AlphaStar Model

 AlphaStar was built by [DeepMind][DeepMind Company]. Plays StarCraft II

 {% youtube "https://www.youtube.com/watch?v=jtlrWblOyP4" %}

 More at:
  * [https://www.deepmind.com/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning](https://www.deepmind.com/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning)
  * [https://www.nature.com/articles/s41586-019-1724-z.epdf](https://www.nature.com/articles/s41586-019-1724-z.epdf)

 See also [A], [OpenAI Five Model]


# AlphaTensor Model

 Better algorithm for tensor multiplication  (on GPU ?). Based on !AlphaZero.

 {% pdf "{{site.assets}}/a/alphatensor_nature_paper.pdf" %}

 More at:
  * announcement - [https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor)
  * paper in nature - [https://www.nature.com/articles/s41586-022-05172-4](https://www.nature.com/articles/s41586-022-05172-4)
  * github code - [https://github.com/deepmind/alphatensor](https://github.com/deepmind/alphatensor)
  * [https://venturebeat.com/ai/deepmind-unveils-first-ai-to-discover-faster-matrix-multiplication-algorithms/](https://venturebeat.com/ai/deepmind-unveils-first-ai-to-discover-faster-matrix-multiplication-algorithms/)

 See also [A], [AlphaZero Model], [DeepMind Company]


# AlphaZero Model

 AI to play chess (and go and tensor algorithm).

 See also [A], [AlphaGo Model], [AlphaTensor Model], [DeepMind Company], [MuZero Model]


# Amazon Company

 See also [A], [Amazon Lex], [Amazon Web Services]


# Amazon Lex

 Text or Speech conversation.

 See also [A], [Amazon Company], [Amazon Web Services]


# Amazon Polly

 Lifelike speech.

 See also [A], [Amazon Company], [Amazon Web Services]


# Amazon Recognition

 Used for image analysis.

 See also [A], [Amazon Company], [Amazon Web Services]


# Amazon Web Services

 See also [A], [Amazon Company]


# Ameca Robot

 {% youtube "https://www.youtube.com/watch?v=rxhzBiC-Q5Y" %}

 See also [A], [Robot]


# Andrew Ng Person

 {% youtube "https://www.youtube.com/watch?v=BY9KV8uCtj4" %}

 {% youtube "https://www.youtube.com/watch?v=vStJoetOxJg" %}

 More at:
  * [https://en.wikipedia.org/wiki/Andrew_Ng](https://en.wikipedia.org/wiki/Andrew_Ng)
  * deeplearning AI youtube channel - [https://www.youtube.com/@Deeplearningai](https://www.youtube.com/@Deeplearningai)

 See also [A], [People]


# Anomaly Detection

 Any deviation from a normal behavior is abnormal :warning: Evolution over time.

 See also [A], [Clustering], [Reconstruction Error], [Distance Methods], [X Density Method]


# Anthropic Company

 See also [A], [Claude Model]


# Apache MXNet

 See also [A], [TensorFlow ML Framework]


# Apache Spark

 (with spark Sagemaker estimator interface?)


# Apple Company

 * [Siri Virtual Assistant]

 See also [A], [Company]


# Apprentice Learning

 See also [A], [Stanford Autonomous Helicopter]


# Apriori Algorithm

 {% youtube "https://www.youtube.com/watch?v=T3Pd_3QP9J4" %}

 See also [Recommendation Engine]


# Area Under The Curve

# AUC

 `~ helpful measurement to compare the classification performance of various models. The bigger the AUC, the better the model!"`. The curve is the ROC Curve! The area under the ROC curve (AUC) is a measure of the classifier's overall performance, with a value of 1 indicating perfect performance and a value of 0.5 indicating a performance no better than random guessing (ROC curve is diagonal ==> ...) .

 See also [A], [ROC Curve]


# Argmax Function

 The output values obtained by a neural network from its various output nodes are not always in the range of 0 to 1, and can be greater than 1 or less than 0. These dynamic values can degrade our machine’s learning power and cause it to misbehave. The Argmax and SoftMax functions are used to obtain values between 0 and 1. The Argmax function interprets the largest positive output value as 1 and all other values as 0, making the model too concrete for a few values. This function is useful for testing because we only need to check the final prediction and not the relationship between different inputs and different outputs/labels.
 ```
# Is this correct?
likelyhood_outputs = [20.4, 3.6, 5.5]
argmax(likely_outputs) = [1, 0, 0]
 ```

 See also [A], [Argmin Function], [Softmax Function]


# Argmin Function

 See also [A], [Argmax Function], [Softmax Function]


# Artificial General Intelligence

# AGI

 AGI is the idealised solution many conceive when thinking about AI. While researchers work on the narrow and superficial, they talk about AGI, which represents the single story of AI, dating back to the 1950s, with a revival in the past decade. AGI implies two things about a solution that should not apply to business-centric problem-solving. First, a program has the general aptitude for human intelligence (perhaps all human intelligence). Second, an AGI is a general problem solver or a blank slate meaning any knowledge of a problem is rhetorical and independent of a strategy to solve that problem. Instead, the knowledge depends on some vague, ill-defined aptitude relating to the multidimensional structure of natural intelligence. If that sounds ostentatious, it’s because it is. Examples:
  * RL can solve arbitrary problems within these environments

 ```
First, we believe the clearest framing of general intelligence is a system that can do anything a human can do in front of a computer. A foundation model for actions, trained to use every software tool, API, and webapp that exists, is a practical path to this ambitious goal, and ACT-1 is our first step in this direction.
# adept.ai
 ```

 {% youtube "https://www.youtube.com/watch?v=wHiOKDlA8Ac" %}

 {% youtube "https://www.youtube.com/watch?v=Mqg3aTGNxZ0" %}

 {% youtube "https://www.youtube.com/watch?v=qbIk7-JPB2c" %}

 {% pdf "https://arxiv.org/pdf/2303.12712.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2303.12712](https://arxiv.org/abs/2303.12712)

 See also [A], [Artificial Narrow Intelligence], [Artificial Super Intelligence]


# Artificial Intelligence

# AI

 Difficult to define?
  1. First approach for a definition: `AI = varied definitions driven by varied questions. AI tries to answer questions such as ...`
   * What is the nature of human intelligence
   * how does the brain work?
   * how to solve problems effectively?
   * How do humans and machine learn?
   * How do we create intelligent creatures?
   * How do AI and humans interact?
   * How can AI have correct values?
   * How can AI contribute for social good?
   * ... and a few more questions!
  2. `A very generic term which refers to teaching machine to imitate human behaviour`
   * A broad area of computer science that means 'computer' makes decisions and solves problems.
   * This encompasses decision tree, Machine learning (trained with data) and deep learning (neural net, unsupervised?).
   * Knowledge acquisition + inference.
  3. :warning: Best definition found =  `AI is the science and engineering ...`
   * `... to use artificial devices`
    * current computer hardware and software, sensors, actuators, etc
   * `... to exhibit human capabilities`
    * perception - undertanding of data
    * cognition - reasoning and learning
     * action - execution and interaction
   * `... to solve problems addressed by humans`

 ![]( {{site.assets}}/a/artificial_intelligence_sensing_effecting.png){: width="100%" }

 ![]( {{site.assets}}/a/artificial_intelligence_machine_learning.png){: width="100%" }

 ![]( {{site.assets}}/a/artificial_intelligence_3_types.png){: width="100%" }

 See also [A], [AI Areas Of Study], [Artificial Narrow Intelligence], [Artificial General Intelligence], [Artificial Super Intelligence], [Human-Centered AI], [Inference], [Intelligence], [Machine Learning], [Natural Intelligence]


# Artificial Intelligence Areas Of Study

# AI Areas Of Study

  * Data and knowledge : massive data understanding, graph learning, synthetic data, knowledge representation
  * Machine vision and language : perception, image understanding, speech, language technologies
  * Learning from experience : reinforcement learning, learning from data, continuous learning from feedback
  * Reasoning and planning : domain representation, optimization, reasoning under uncertainty and temporal constraints
  * multi-agent systems : agent-based simulation, negotiation, game and behavior theory, mechanism design
  * secure and private AI : Privacy, cryptography, secure multi-party computation, federated learning
  * safe human-AI interaction : Agent symbiosis, ethics and fairness, explainability, trusted AO

 See also [A], [Artificial Intelligence]


# Artificial Intelligence Challenge

# AI Challenge

 * 1994 - 2022+ : First [CASP Challenge], CASP13 in 2018 was won by [AlphaFold 1][AlphaFold Model]
 * 1997 - 1998 : [Deep Blue Challenge]
 * 2004 - 2005 : [DARPA Grand Challenge]
 * 2006 - 2009 : [Netflix Prize], in 2009 10% threshold was achieved, but customer data leak reported!
 * 2007 : [DARPA Urban Challenge]
 * 2011 - 2012 : [ImageNet Large Scale Visual Recognition Challenge]
 * 2015 : [DARPA Robotics Challenge]
 * 2021 : [DARPA Subterranean Challenge]

 See also [A], ...


# Artificial Intelligence Complete

# AI Complete

 Relates to NP complete from complexity.

 See also [A], [AI Hard]


# Artificial Intelligence Hard

# AI Hard

 Relates to NP hard from complexity.

 See also [A], [AI Complete]


# Artificial Narrow Intelligence

# ANI

 ANI is often conflated with weak artificial intelligence. John Searle, philosopher and professor at the University of California, explained in his seminal 1980 paper, “Minds, Brains, and Programs,” that weak artificial intelligence would be any solution that is both narrow and a superficial look-alike to intelligence. Searle explains that such research would be helpful in testing hypotheses about segments of minds but would not be minds.[3] ANI reduces this by half and allows researchers to focus on the narrow and superficial and ignore hypotheses about minds. In other words, ANI purges intelligence and minds and makes artificial intelligence “possible” without doing anything. After all, everything is narrow, and if you squint hard enough, anything is a superficial look-alike to intelligence.

 {% pdf "{{site.assets}}/a/artificial_narrow_intelligence_paper.pdf" %}

 See also [A], [Artificial General Intelligence], [Artificial Super Intelligence]


# Artificial Neural Network

# ANN

 ~ `Can discover and approximate a(ny?) function given fixed(-count?) inputs and fixed(-count?) outputs! = universal function approximator` A multi-layer perceptron. Also known as Artificial Neural Network (ANN). Can be used in supervised or unsupervised learning.

 ![]( {{site.assets}}/a/artificial_neural_network_example.png ){: width="100%"}

 `For a young person, the university he/she is coming from is important. Compare that to someone with experience, whose university may not matter so much. but experience and field are important. ==> the weight of university is function of the age ==> That's what the second layer correct for!`

 ![]( {{site.assets}}/a/artificial_neural_network_types.png ){: width="100%"}


 The way for researchers to build an artificial brain or [neural network] using [artificial neurons][Artificial Neuron]. There are several types of ANN, including:
  * [Convoluted Neural Network]
  * [Recurrent Neural Network][RNN] vs [Feedforward Neural Network]
  * ...

 The "knowledge/skills" of the ANN are encoded in their [parameters][Parameter]

 Hyperparameters:
  * Number of neurons per layer
  * Number of layers
  * [Activation function]
  * [Dropout function]

 {% youtube "https://www.youtube.com/watch?v=hfMk-kjRv4c" %}

 More at:
  * playground - [https://playground.tensorflow.org/](https://playground.tensorflow.org/)

 See also [A], [Hidden Layer], [Input Layer], [Output Layer], [Perceptron], [Universal Function Approximator]


# Artificial Neuron

 aka Node, currently artificial neurons are implemented as a [Perceptrons][Perceptron].

 Several (binary) input channels, to produce one output (binary value) that can be faned out. Input weights, Bias (add an offset vector for adjustment to prior predictions), non-linear activation function (sum+bias must meet or exceed activation threshold).

 ![]( {{site.assets}}/a/artificial_neuron.png ){: width="100%"}

 See also [A], [Activation Function], [Bias], [Input Weight]


# Artificial Super Intelligence

# ASI

 ASI is a by-product of accomplishing the goal of AGI. A commonly held belief is that general intelligence will trigger an “intelligence explosion” that will rapidly trigger super-intelligence. It is thought that ASI is “possible” due to recursive self-improvement, the limits of which are bounded only by a program’s mindless imagination. ASI accelerates to meet and quickly surpass the collective intelligence of all humankind. The only problem for ASI is that there are no more problems. When ASI solves one problem, it also demands another with the momentum of Newton’s Cradle. An acceleration of this sort will ask itself what is next ad infinitum until the laws of physics or theoretical computation set in. The University of Oxford scholar Nick Bostrom claims we have achieved ASI when machines have more intelligent than the best humans in every field, including scientific creativity, general wisdom, and social skills. Bostrom’s depiction of ASI has religious significance. Like their religious counterparts, believers of ASI even predict specific dates when the Second Coming will reveal our savior. Oddly, Bostrom can’t explain how to create artificial intelligence. His argument is regressive and depends upon itself for its explanation. What will create ASI? Well, AGI. Who will create AGI? Someone else, of course. AI categories suggest a false continuum at the end of which is ASI, and no one seems particularly thwarted by their ignorance. However, fanaticism is a doubtful innovation process.

 See also [A], [Artificial General Intelligence], [Artificial Narrow Intelligence]


# Atlas Robot

 Atlas is a bipedal humanoid [robot] primarily developed by the American robotics company [Boston Dynamics][Boston Dynamics Company] with funding and oversight from the [U.S. Defense Advanced Research Projects Agency (DARPA)][DARPA]. The robot was initially designed for a variety of search and rescue tasks, and was unveiled to the public on July 11, 2013.

 {% youtube "https://www.youtube.com/watch?v=rVlhMGQgDkY" %}

 More at:
  * [https://www.bostondynamics.com/atlas](https://www.bostondynamics.com/atlas)
  * [https://en.wikipedia.org/wiki/Atlas_%28robot%29](https://en.wikipedia.org/wiki/Atlas_%28robot%29)

 See also [A], ...


# Attention

 To work with transformer models, you need to understand one more technical concept: attention. An attention mechanism is a technique that mimics cognitive attention: it looks at an input sequence, piece by piece and, on the basis of probabilities, decides at each step which other parts of the sequence are important. For example, look at the sentence “The cat sat on the mat once it ate the mouse.” Does “it” in this sentence refer to “the cat” or “the mat”? The transformer model can strongly connect “it” with “the cat.” That’s attention.
 
 The transformer model has two types of attention: self-attention (connection of words within a sentence) and encoder-decoder attention (connection between words from the source sentence to words from the target sentence).

 {% pdf "{{site.assets}}/a/attention_paper.pdf" %}

 The attention mechanism helps the transformer filter out noise and focus on what’s relevant: connecting two words in a semantic relationship to each other, when the words in themselves do not carry any obvious markers pointing to one another. AN improvement over Recurrent Neural Network, such as Long Short Term memory (LSTM) Networks.

 See also [A], [Attention Score], [Attention-Based Model], [Cross-Attention], [Encoder-Decoder Attention], [Long Short Term Memory Network], [Masked Self-Attention], [Multi-Head Attention], [Recurrent Neural Network], [Self-Attention], [Transformer Model]


# Attention-Based Model

 In a language modeling task, a model is trained to predict a missing workd in a sequence of words. In general, there are 2 types of language modesl:
  1. Auto-regressive ( ~ auto-complete and generative)
  1. Auto-encoding ( ~ best possible match given context )

 See also [A], [Attention], [Attention Score], [Autoencoding], [Autoregressive], [BERT Model], [GPT Model], [Multi-Head Attention], [T5 Model]


# Attribute

 See also [A], [Negative Attribute], [Positive Attribute]]


# Autoencoder

 Let’s now discuss autoencoders and see how we can use neural networks for dimensionality reduction. The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as neural networks and to learn the best encoding-decoding scheme using an iterative optimisation process. So, at each iteration we feed the autoencoder architecture (the encoder followed by the decoder) with some data, we compare the encoded-decoded output with the initial data and backpropagate the error through the architecture to update the weights of the networks. Thus, intuitively, the overall autoencoder architecture (encoder+decoder) creates a bottleneck for data that ensures only the main structured part of the information can go through and be reconstructed. Looking at our general framework, the family E of considered encoders is defined by the encoder network architecture, the family D of considered decoders is defined by the decoder network architecture and the search of encoder and decoder that minimise the reconstruction error is done by gradient descent over the parameters of these networks.

 ![]( {{site.assets}}/a/autoencoder.png ){: width="100%"}

 See also [A], [Autoencoding], [Backpropagation], [Decoder], [Denoising Autoencoder], [Dimensionality Reduction], [Disentangled Variational Autoencoder], [Encoder], [Encoder-Decoder Model], [Hidden State], [Linear Autoencoder], [Unsupervised Deep Learning Model], [Unsupervised Learning], [Variational Autoencoder]


# Autonomous Vehicle

 See also [DARPA Grant Challenge], [DARPA Urban Challenge]


# Attention Score

 `~ how much to pay attention to a particular word`
  * Q, K, V matrix for the encoder <-- needs to be computed for the encoder (?) like weights/bias of an ANN
  * For each words, Q, K, V are computed by multiplying the word embedding with the corresponding Q, K, V matrix of the encoder !??!?!
 The Query word (Q) can be interpreted as the word for which we are calculating Attention. The Key and Value word (K and V) is the word to which we are paying attention ie. how relevant is that word to the Query word.

 ```
The query key and value concept come from retrieval systems.
For example, when you type a query to search for some video on Youtube.
The search engine will map your query against a set of keys (video title, description etc.) associated with candidate videos in the database
Then you are presented with the best matched videos (values).                        <== STRENGTH OF THE ATTENTION ? No!
 ```
 An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

 ![]( {{site.assets}}/a/attention_score.png ){: width="100%"}

 Multi-Head Attention consists of several attention layers running in parallel. The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value (aka Q,K,V). All three parameters are similar in structure, with each word in the sequence represented by a vector. In transformers is used for encoder and decoder.

 ![]( {{site.assets}}/m/multi_head_attention.png ){:width="100%"}

 ![]( {{site.assets}}/a/attention_score_formula.png ){:width="100%"}

 {% youtube "https://www.youtube.com/watch?v=_UVfwBqcnbM" %}

 {% youtube "https://www.youtube.com/watch?v=4Bdc55j80l8" %}

 More at :
  * [https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)
  * [https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

 See also [A], [Attention], [Attention-Based Model], [Multi-Head Attention], [Positional Encoding], [Transformer Model]


# Augmented Reality

# AR

 Augmented reality (AR) is an interactive experience that combines the real world and computer-generated content. The content can span multiple sensory modalities, including visual, auditory, haptic, somatosensory and olfactory. AR can be defined as a system that incorporates three basic features: a combination of real and virtual worlds, real-time interaction, and accurate 3D registration of virtual and real objects. The overlaid sensory information can be constructive (i.e. additive to the natural environment), or destructive (i.e. masking of the natural environment). This experience is seamlessly interwoven with the physical world such that it is perceived as an immersive aspect of the real environment. In this way, augmented reality alters one's ongoing perception of a real-world environment, whereas virtual reality completely replaces the user's real-world environment with a simulated one.

 Augmented reality is largely synonymous with [Mixed Reality].

 {% youtube "https://www.youtube.com/watch?v=Zu1p99GJuTo" %}

 More at:
  * [https://en.wikipedia.org/wiki/Augmented_reality](https://en.wikipedia.org/wiki/Augmented_reality)

 See also [A], [Virtual Continuum]


# Autoencoder Bottleneck

 This is the name of the hidden layer with the least neuron in an autoencoder architecture. This is where the information is compressed, encoded in the latent space!

 ![]( {{site.assets}}/a/autoencoder_bottleneck.png ){:width="100%"}

 See also [A], [Autoencoder], [Latent Space]


# Autoencoder Type

 There are 2 types of autoencoders:
  * input X --> Latent representation
  * input X --> Latent distribution

 ![]( {{site.assets}}/a/autoencoder_type.png){: width="100%"}

 See also [A], [Autoencoder], [Variational Autoencoder]


# Autoencoding

 ~ auto-complete of a sentence on a phone. Goal is to learn representations of the entire sequence by predicting tokens given both the past and future tokens. If only past or future ==> autoregressive.
 ```
If you don't ___ at the sign, you will get a ticket
 ```

 See also [A], [Autoencoder], [Autoregressive]

# Autoencoding Model

  * Comprehensive understanding and encoding of entire sequences of tokens
  * Natural Language Understanding (NLU)
  * BERT Models

 See also [A], [BERT Model], [Natural Language Understanding]

# Automation

  * Low automation = human does the work
  * High automation = AI does the work!

 ![]( {{site.assets}}/a/automation_sae_j3016.png ){:width="100%"}

 More at:
  * [https://www.sae.org/news/2019/01/sae-updates-j3016-automated-driving-graphic](https://www.sae.org/news/2019/01/sae-updates-j3016-automated-driving-graphic)

 See also

# AutoML

 significant advances in automated machine learning that can “automatically discover complete machine learning algorithms just using basic mathematical operations as building blocks.”

 See also [A], [Hyperparameter Tuning], [Early Stopping], [Neural Architecture Search]

# Autoregressive Model

 Goal is to predict a future token (word) given either the past tokens or the future tokens but not both. (If both --> auto-encoding). Autoregressive models such as decoders are iterative and reused their temporary, incomplete output to generate the next, more complete output. Iterations stop when encoder input is exhausted (?). Well-known autoregressive models/use-cases are:
  * Predicting next work in a sentence (auto-complete)
  * Natural language generation
  * [GPT Models][GPT Model]
 ```
If you don't ____ (forward prediction)
____ at the sign, you will get a ticket (backward prediction)
 ```

 See also [A], [Autoencoding], [Casual Language Modeling], [Decoder]


# Autoregressive Convolutional Neural Network

# AR-CNN

 `~ representing the problem/solution as a time series of images`. Q: How do you generated the images? Q: How many images? Ex: music --> given the previous images (piano roll with notes until now), find the next image (piano roll with new note!) ? NO, WRONG!! More like imagine your first COMPLETE draft, you review the draft, you review it once more, and again, getting better and better each time until it cannot get better anymore! `Each action (i.e. addition or removal of exactly one note) is a transformation from one piano roll image to another piano roll image`.

 See also [A], [Autoregressive Model], [Convolutional Neural Network]


# Autoregressive Model

# AR Model

 `~ analyze the past steps (or future but not both) to identify the next step = learn from the past iteration (or future but not both) ONLY`. Unlike the GANs approach described before, where music generation happened in one iteration, autoregressive models add notes over many iterations. The models used are called autoregressive (AR) models because the model generates music by predicting future music notes based on the notes that were played in the past. In the music composition process, unlike traditional time series data, one generally does more than compose from left to right in time. New chords are added, melodies are embellished with accompaniments throughout. Thus, instead of conditioning our model solely on the past notes in time like standard autoregressive models, we want to condition our model on all the notes that currently exist in the provided input melody. For example, the notes for the left hand in a piano might be conditioned on the notes that have already been written for the right hand.

 See also [A], [Autoregressive Convolutional Neural Network], [Generative Adversarial Network], [GPT Model], [Time-Series Predictive Analysis]
