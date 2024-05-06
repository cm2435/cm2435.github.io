# Towards Robust Reasoning in Language Models via Mechanistic Interpretability: A research proposal

(Dated: March 23, 2024)


# Towards Robust Reasoning in Language Models via Mechanistic Interpretability: A Research Proposal

Date: March 3rd, 2024 | Estimated Reading Time: 30 min | Author: Charlie Masters

## Table of Contents

<details>
<summary>I. Language Models: Alignment, Reasoning and Interpretability</summary>
</details>

<details>
<summary>II. Model Interpretability and Behavioural Interventions</summary>
</details>

<details>
<summary>III. Proposal for New Research</summary>

  <details>
  <summary>A. Investigate the generality of mechanistic interpretability methods across LLM architectures</summary>
  </details>

  <details>
  <summary>B. Discover Novel Mechanistic Interpretability Algorithms</summary>
  </details>

  <details>
  <summary>C. Investigate interventions on LLM architectures using Mechanistic Interpretability</summary>
  </details>

</details>

<details>
- IV. Conclusion
</details>

<details>
- V. References
</details>

## Brief preface
This was a short paper I wrote as part of a PhD application to the Kings College Department of Safe and Trusted AI. It is a good primer into a range of topics including mechanistic interpretability, language model allignment, safety and some other areas of current research.

## Abstract

Large Language Models (LLMs) in recent years have become an industry backbone for a range of tasks. Despite the effectiveness of recently developed alignment and safety methods, our ability to construct safe and reliable models is limited by the low level of interpretability of LLMs and the lack of effective methods for controlling dangerous language model behaviour. In this proposal, we review the current state of LLM alignment and interpretability, discussing how new methods like mechanistic interpretability can be used to probe LLM understanding and behaviour. We also will discuss the role of these methods in designing new techniques to perform interventions on model behaviour to create safe and trustworthy systems. Finally, we propose possible avenues for research using mechanistic interpretability and representation-based methods to further research common interpretability structures across different model types, new novel interpretability methods and the role of interpretability analysis in designing rigorous AI systems.

## I. Language models: allignment, reasoning and interpretability

Over the past few years, Large Language Models (LLMs) have become increasingly capable across a wide range of tasks [[1](https://arxiv.org/abs/2204.02311)], [[2](https://arxiv.org/abs/2204.02311)]. As an emergent property of parameter scale [[3](https://arxiv.org/abs/2204.02311)] we have seen LLMs exhibit behaviour that resembles reasoning across a diverse range of topics and concepts.

Additionally, recent methods allowing for LLMs to align themselves with human preferences [[4](https://arxiv.org/abs/1706.03741)] have proved highly performant in taking the significant world knowledge and reasoning behaviours learned during pre-training, and orienting their behaviour with preferable behaviours like Honesty, Helpfulness and Harmlessness [[5](https://arxiv.org/abs/2204.05862)]. From this, we have seen a notable improvement in LLM performance in a range of tasks across NLP [[6](https://arxiv.org/abs/2009.03300)], general reasoning [[7](https://arxiv.org/abs/2307.13692)], mathematics [[8](https://arxiv.org/abs/2110.14168)] and many more.

In light of this widespread adoption in industry, alignment research has become increasingly important with the critical need for systems that are reliable, safe and robust. There has been significant policy research on the possible effects of models producing incorrect or hazardous outputs [[9](https://arxiv.org/abs/2009.03300)], [[10](https://arxiv.org/abs/2306.05949)]. To strive for the goal of safe AI, since the original Reinforcement Learning from Human Feedback paper [[11](https://arxiv.org/abs/1706.03741)] a series of other algorithms have been proposed. These works aim to improve alignment results in several ways, including using non-reinforcement learning-based algorithms which are less sensitive to hyperparameter choice [[12](https://arxiv.org/abs/2305.18290)], better-aligning model outputs with the preference dataset [[13](https://arxiv.org/abs/2402.19085)] or lowering the difficulty of sourcing preference data [[14](https://arxiv.org/abs/2402.01306)].

Despite promising initial results in the development of reliable agents that can reason effectively and safely, there has been notable recent work calling into question the true reasoning capacity of transformer-based LLMs [[15](https://doi.org/10.1145/3442188.3445922)]. By modelling compositional reasoning tasks as Directed Acyclic Graphs of logical reasoning steps, it has been shown [[16](https://arxiv.org/abs/2305.18654)] that transformers learn to emulate reasoning skills by performing linear subgraph matching with previously seen training DAGs rather than developing systematic problem-solving skills. Similarly, by defining reasoning tasks as formal grammars [[17](https://arxiv.org/abs/2207.02098)], prior work has concluded transformers, recursive neural networks (RNNs) and other common architectures unable to generalise on a range of context-free and context-sensitive tasks.

## II. Model interpretability and behavioural interventions

To better understand how these models store world knowledge and reason about tasks, there has been significant and diverse research to shed light on how LLMs represent information. It has been hypothesised [[18](https://arxiv.org/abs/2209.10652)] that there is a linear structure of the representations of many human interpretable topics in language models. This is well supported by previous works finding emergent structure in learned representations across a range of tasks and architectures. [[19](https://arxiv.org/abs/2023)], [[20](https://aclanthology.org/N13-1090)], [[21](https://arxiv.org/abs/2106.12423)]. 

In recent years, mechanistic interpretability has made progress in reverse engineering neural networks by using a range of techniques.

Traditionally, interpretability research was conducted with methods such as saliency maps [[22](https://arxiv.org/abs/2008.05122)] [[23](https://arxiv.org/abs/1312.6034)] which identify key input features that contribute the most to the network's output. Unfortunately, these methods are often noisy and without a mechanistic explanation of model behaviour, doing model interventions is difficult. By contrast, mechanistic interpretability seeks to directly reverse engineer neural networks, similar to how one might reverse engineer a compiled binary computer program.

To try to understand how the model components line up with human-understandable concepts, correspondence between the components of a model and human-understandable concepts, [[24](https://arxiv.org/abs/2211.00593)] tries to learn the 'circuit view' of GPT-2, using causal interventions to try to explain how it performs language modelling. However, this proves difficult with the circuit exhibiting complex behaviour that is hard to codify. One cause of this is polysemanticity [[25](https://arxiv.org/abs/2210.01892)], where single neurons in a network can represent a mixture of unrelated features. Although early works tried to examine neural network behaviour by training sparse models where neurons didn't exhibit 'superposition' of concepts [[18](https://arxiv.org/abs/2209.10652)], later works [[26](https://arxiv.org/abs/2309.08600)] found counter-examples showing that this would be unlikely to be achievable in general cases. Instead of training models with 'monosemantic' neurons, this paper generates interpretable learned features by using a sparse autoencoder. The results of this are that some of the extracted features are 'causal units' which don't correspond to single neurons, making circuit-based analysis challenging.

Orthogonal to mechanistic interpretability methods, other works aim to take a 'top-down' approach to understanding the latent structure of language models. Representation Engineering [[27](https://arxiv.org/abs/2310.01405)] has found popularity in research as a method to probe and perform interventions on the representations of language models when encountering specific topics. Taking inspiration from prior work in neurology, a 'Hopfieldian' view of LLM reasoning as the product of the models' representational spaces has proved effective in not only probing latent structures in opensource language models but also 'steering' model behaviours towards user-defined behaviours [[28](https://arxiv.org/abs/2312.03813)].

In practice, both mechanistic and activation-baed interpretability methods have empowered seminal breakthroughs in how language models store information, and as a result how they behave. One work [[29](https://openreview.net/forum?id=A0HKeKl4Nl)] uses formal grammars to mechanistically probe precisely what is happening to the model's underlying capabilities. This analysis concludes that (i) Supervised fine-tuning of a language model rarely alters the underlying model capabilities; (ii) Instead of updating all of the models' learned world knowledge, fine-tuning normally results in language models learning a 'wrapper' for the task creating the illusion of modified latent knowledge. 

Building on these insights, researchers have made good progress in both highlighting and mitigating potentially dangerous model behaviour. Recent research into safety [[30](https://arxiv.org/abs/2401.05566)] has shown that: (i) Safety alignment algorithms update model weights in a sparse manner [[31](https://arxiv.org/abs/2402.05162)], leading to only small regions of neurons which are critical to alignment and thus are susceptible to just fine-tuning the model to remove safety features (ii) malignant backdoors can be inserted into language models by poisoning pre-training data which cannot be removed by standard safety training techniques. Since this, other works [[32](https://arxiv.org/abs/2403.05030)] have been able to design methods to suppress harmful behaviour in models that weren't surfaced by adversarial testing or in preference data. In addition, a better insight into how models store latent information and how fine-tuning methods impact model behaviour would complement current research [[33](https://arxiv.org/abs/2305.20050)] [[34](https://arxiv.org/abs/2312.09390)] into being models capable at advanced reasoning. In the iterative cycle of identifying and patching possible attack vectors for LLMs, mechanistic interpretability has proved a useful tool for safety research.

## III. Proposal for new research

Mechanistic interpretability when combined with other methods has great capacity as a framework for both understanding and improving language models to make them more robust and safe. Prior work discussed has shown that although LLMs are by their nature difficult to interpret and hard to rigorously enforce to exhibit preferable behaviours. Despite this, interpretability has provided a diverse toolkit for better understanding LLMs and as a result designing novel algorithms for LLM alignment.

In light of this, there are a range of possible avenues of research. We will explore the main open questions in each section, and then cover a brief proposal of action on how to tackle these open problems.

### A. Investigate the generality of mechanistic interpretability methods across LLM architectures

As LLMs grow in popularity in the industry, the diversity of architectures powering state-of-the-art models is growing. To reduce the issues imposed by the linear memory and time complexities of the transformer block with respect to sequence length, both RWKV [[35](https://arxiv.org/abs/2305.13048)] and state-space models [[36](https://arxiv.org/abs/2312.00752)] have seen great success as the backbone of performant, compute efficient LLMs. Although there has been some work [[37](https://arxiv.org/abs/2404.05971)] showing that language model interpretability and steering techniques likely can be used on recursive architectures, this is largely an unstudied domain of interpretability and still holds interesting unanswered questions. If interpretability techniques like dictionary learning generalise across language model architecture, this may suggest that there is a common structure to storing features across different types of neural models. Additionally, there is an interesting avenue of analysis for seeing how these rules generalise across model scales and modality: discovering new patterns for how multimodal language models map features from different input spaces would lay important groundwork for solving contemporary problems like alignment in multimodal models [[38](https://arxiv.org/abs/2403.17830)] and network feature collapse [[39](https://arxiv.org/abs/2402.05162)].

To investigate these generality rules, we propose building on the approach of [[26](https://arxiv.org/abs/2309.08600)] and training a sparse-autoencoder on a set of models with a given data budget such as a subsample of the stack or pile [[40](https://arxiv.org/abs/2101.00027)], varying a key feature of either model size or model architecture. We then evaluate how key indicators like autoencoder reconstruction loss and feature density vary with experiment hyperparameters to try to find and discover scaling or relation laws. The end goal of this would discovering a seminal relationship like 'transformer models with a computing budget greater than X undergoes a phase transition in feature density, relating to sudden emergent reasoning ability'. Separately, finding constant results in feature interpretability across architectures like mixture-of-experts, transformers and state space models would provide more definitive answers to key questions posed by prior works [[26](https://arxiv.org/abs/2309.08600)] as to whether learned features are a fundamental property or just a useful post-hoc description of the reference language model.

### B. Discover Novel Mechanistic Interpretability Algorithms

Existing mechanistic interpretability methods despite being promising have exhibited specific flaws. Sparse AutoEncoders (SAE's) have proven a powerful tool for learning linear interpretable feature bases in models [[26](https://arxiv.org/abs/2309.08600)][[41](https://github.com/jbloomAus/SAELens)]. Despite this, recent analysis shows SAEs trained to interpret GPT-2 medium [[42](https://github.com/jbloomAus/SAELens)] have reconstruction errors that are systemic rather than random. As a result, decreasing this error may not increase SAE faithfulness, indicating that they may miss out on important parts of the learned representations of LLMs. It is reported in prior works [[43](https://aclanthology.org/D19-1006)] that in language models the representations have an anisotropic distribution. The L-1 regularisation used in sparse-autoencoders makes the primary logit mode zero-centred, creating for some models a structural error which explains the above finding. We propose exploring models which can learn more faithful reconstructions, potentially taking inspiration from [[28](https://arxiv.org/abs/2312.03813)] and using a mean-centred approximation in our dictionary learning model.

Separately, circuit-based methods learned via causal interventions [[24](https://arxiv.org/abs/2211.00593)] have reported unexplained behaviours in even smaller LLMs such as Redundancy, Compensation and the unexpected utilization of known Structures. As part of our work in discovering novel explainability methods, we plan on exploring if models can explain their features and circuits. There has been some success in this in [[44](https://arxiv.org/abs/2401.06102)] which builds on the idea of mapping the representation of features to a model's vocabulary space and doing interventions on the computation. The goal of this is to develop new methods that can explore previously unexplained properties of derived circuits at scale and allow for a direct comparison of the efficacy of neuron, activation and feature-level explainability approaches.

### C. Investigate interventions on LLM architectures using Mechanistic Interpretability

Mechanistic interpretability provides a strong framework for understanding the behaviour of language models and how they store information. Pragmatically, there has been significant work leveraging the internal representation of the hidden states of LLMs. Stemming from this, researchers have created models with state-of-the-art truthfulness [[27](https://arxiv.org/abs/2310.01405)], frameworks for steering LLM generations towards key topics [[28](https://arxiv.org/abs/2312.03813)] and designed novel adversarial attack methods [[45](https://arxiv.org/abs/2311.09433)]. Despite these successes, most contemporary methods for activation steering suffer from drawbacks such as needing manually crafted 'steering vectors' to perform activation-based interventions [[27](https://arxiv.org/abs/2310.01405)] or struggle to scale to larger architectures [[46](https://www.lesswrong.com/posts/iaJFJ5Qm29ixtrWsn/sparse-coding-for-mechanistic-interpretability-and#Introduction)].

To address these issues, an important unexplored component of these techniques is knowledge of how models store key concepts and world knowledge. Further studying of the role of sparse-coding methods to learn good activation guides for inference time interventions on LLMs could provide great utility to alignment research. Finally, as previously discussed, recent work has tried to probe the compositional reasoning capability of common LLM architectures by posing reasoning as formal structures such as DAGs or grammars. As interpretability methods improve with time, they can likely be used to glean information about how transformers and other foundational blocks of LLMs combine topics to create complex chains of reasoning. Further research into this topic could therefore enable the development of models with new superior reasoning capabilities.

## IV. Conclusion

In this proposal, we have done a brief review of language models and how recent methods around alignment improved their abilities across a range of tasks. To analyse how these methods update the store of features and world knowledge, mechanistic interpretability has been a source of a range of useful techniques to reverse engineer the inner behaviour of LLMs, highlighting key areas of improvement of safety techniques and providing the theoretical basis for performing interventions to create safe and trustworthy foundation models. We have also highlighted some contemporary methods for areas of further research to generate new novel methods to do mechanistic interpretability, find general principles and scaling-laws for interpretability techniques across language model architectures and discover new methods to increase the capability of LLMs in reasoning, safety and allignment.

## V. References

[1] TECHREPORT: A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. [Language models are unsupervised multitask learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Technical report, OpenAI, 2019.

[2] : OpenAI and Josh Achiam et Al. [GPT-4 technical report](https://arxiv.org/abs/2303.08774).

[3] : Aakanksha Chowdhery et Al. [PaLM: Scaling language modeling with pathways](https://arxiv.org/abs/2204.02311), 2022.

[4] : Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan, Aidan O'Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. [AI alignment: A comprehensive survey](https://arxiv.org/abs/2310.19852), 2024.

[5] : Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. [Training a helpful and harmless assistant with reinforcement learning from human feedback](https://arxiv.org/abs/2204.05862), 2022.

[6] : Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. [Measuring massive multitask language understanding](https://arxiv.org/abs/2009.03300), 2021.

[7] : Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J. Nay, Kshitij Gupta, and Aran Komatsuzaki. [ARB: Advanced reasoning benchmark for large language models](https://arxiv.org/abs/2307.13692), 2023.

[8] : Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. [Training verifiers to solve math word problems](https://arxiv.org/abs/2110.14168), 2021.

[9] : Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and William Yang Wang. [On the risk of misinformation pollution with large language models](https://arxiv.org/abs/2305.13661), 2023.

[10] ARTICLE: et Al. Solaiman. [Evaluating the social impact of generative ai systems in systems and society](https://arxiv.org/abs/2306.05949). arXiv preprint arXiv:2306.05949, 6 2023.

[11] ARTICLE: Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741). 2017.

[12] ARTICLE: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. [Direct preference optimization: Your language model is secretly a reward model](https://arxiv.org/abs/2305.18290). 2023.

[13] : Yiju Guo, Ganqu Cui, Lifan Yuan, Ning Ding, Jiexin Wang, Huimin Chen, Bowen Sun, Ruobing Xie, Jie Zhou, Yankai Lin, Zhiyuan Liu, and Maosong Sun. [Controllable preference optimization: Toward controllable multi-objective alignment](https://arxiv.org/abs/2402.19085), 2024.

[14] : Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. [KTO: Model alignment as prospect theoretic optimization](https://arxiv.org/abs/2402.01306), 2024.

[15] INPROCEEDINGS: Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. [On the dangers of stochastic parrots: Can language models be too big?](https://doi.org/10.1145/3442188.3445922) . In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT '21, page 610–623, New York, NY, USA, 2021. Association for Computing Machinery.

[16] ARTICLE: Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaid Harchaoui, and Yejin Choi. [Faith and fate: Limits of transformers on compositionality](https://arxiv.org/abs/2305.18654). 2023.

[17] : Gr´egoire Del´etang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, and Pedro A. Ortega. [Neural networks and the chomsky hierarchy](https://arxiv.org/abs/2207.02098), 2023.

[18] ARTICLE: Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. [Toy models of superposition](https://arxiv.org/abs/2209.10652). 2022.

[19] ARTICLE: Maxime Oquab, Timoth´ee Darcet, and Moutakanni. [Dinov2: Learning robust visual features without supervision](https://arxiv.org/abs/2023). arXiv preprint arXiv:2023, 2023.

[20] INPROCEEDINGS: Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. [Linguistic regularities in continuous space word representations](https://aclanthology.org/N13-1090). In Lucy Vanderwende, Hal Daum´e III, and Katrin Kirchhoff, editors, Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 746–751, Atlanta, Georgia, June 2013. Association for Computational Linguistics.

[21] : Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. [Alias-free generative adversarial networks](https://arxiv.org/abs/2106.12423), 2021.

[22] : Ian Tenney, James Wexler, Jasmijn Bastings, Tolga Bolukbasi, Andy Coenen, Sebastian Gehrmann, Ellen Jiang, Mahima Pushkarna, Carey Radebaugh, Emily Reif, and Ann Yuan. [The language interpretability tool: Extensible, interactive visualizations and analysis for nlp models](https://arxiv.org/abs/2008.05122), 2020.

[23] : Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. [Deep inside convolutional networks: Visualising image classification models and saliency maps](https://arxiv.org/abs/1312.6034), 2014.

[24] : Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. [Interpretability in the wild: a circuit for indirect object identification in gpt-2 small](https://arxiv.org/abs/2211.00593), 2022.

[25] : Adam Scherlis, Kshitij Sachan, Adam S. Jermyn, Joe Benton, and Buck Shlegeris. [Polysemanticity and capacity in neural networks](https://arxiv.org/abs/2210.01892), 2023.

[26] ARTICLE: Joshua Batson Brian Chen Adam Jermyn Tom Conerly et Al. Trenton Bricken, Adly Templeton. [Towards monosemanticity: Decomposing language models with dictionary learning](https://arxiv.org/abs/2309.08600). 2023.

[27] ARTICLE: Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. [Representation engineering: A top-down approach to ai transparency](https://arxiv.org/abs/2310.01405). 2023.

[28] : Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. [Improving activation steering in language models with mean-centring](https://arxiv.org/abs/2312.03813), 2023.

[29] INPROCEEDINGS: Samyak Jain, Robert Kirk, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka, Tim Rockt¨aschel, Edward Grefenstette, and David Krueger. [What happens when you fine-tuning your model? mechanistic analysis of procedurally generated tasks](https://openreview.net/forum?id=A0HKeKl4Nl). In The Twelfth International Conference on Learning Representations, 2024.

[30] ARTICLE: Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse, Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Brauner, Holden Karnofsky, Paul Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, S¨oren Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, and Ethan Perez. [Sleeper agents: Training deceptive llms that persist through safety training](https://arxiv.org/abs/2401.05566). 2024.

[31] : Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, and Peter Henderson. [Assessing the brittleness of safety alignment via pruning and low-rank modifications](https://arxiv.org/abs/2402.05162), 2024.

[32] : Stephen Casper, Lennart Schulze, Oam Patel, and Dylan Hadfield-Menell. [Defending against unforeseen failure modes with latent adversarial training](https://arxiv.org/abs/2403.05030), 2024.

[33] : Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. [Let's verify step by step](https://arxiv.org/abs/2305.20050), 2023.

[34] : Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeff Wu. [Weak-to-strong generalization: Eliciting strong capabilities with weak supervision](https://arxiv.org/abs/2312.09390), 2023.

[35] : Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. [Rwkv: Reinventing rnns for the transformer era](https://arxiv.org/abs/2305.13048), 2023.

[36] : Albert Gu and Tri Dao. [Mamba: Linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752), 2023.

[37] : Gon¸calo Paulo, Thomas Marshall, and Nora Belrose. [Does transformer interpretability transfer to rnns?](https://arxiv.org/abs/2404.05971), 2024.

[38] : Zhelun Shi, Zhipin Wang, Hongxing Fan, Zaibin Zhang, Lijun Li, Yongting Zhang, Zhenfei Yin, Lu Sheng, Yu Qiao, and Jing Shao. [Assessment of multimodal large language models in alignment with human values](https://arxiv.org/abs/2403.17830),