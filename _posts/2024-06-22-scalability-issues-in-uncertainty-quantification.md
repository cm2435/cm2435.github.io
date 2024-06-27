# Model and Computational Complexity

Large Language Models are characterized by their ever-growing model complexity and large-scale architectures, typically containing billions of parameters. This significant scale poses some difficulty in quantifying the uncertainty of language models without prohibitive cost, latency, or hardware requirements.

Methods such as deep ensembles [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474), sampling-based methods [Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142), and more recently conformal prediction [Angelopoulos & Bates, 2022](https://arxiv.org/abs/2107.07511) have been used to great success for uncertainty quantification. However, these models are significantly more computationally expensive than inferencing their frequentist counterparts.

These techniques treat the model parameters \(\theta\) fitted to data \(D\) as random variables that can be used to compute an approximation \(q(\theta)\) to the true posterior distribution \(p(\theta \mid D)\), which is intractable to compute directly as a result of the extreme dimensionality of language space. This approximation \(q(\theta)\) is typically significantly more computationally expensive than computing a point estimate of the output.

This can be illustrated by deep ensemble-based methods [Malinin & Gales, 2021](https://arxiv.org/abs/2002.07650), where the predictive posterior is found by taking the expectation over an ensemble of models \(\{P(y \mid x; \theta^{(m)})\}_{m=1}^M\):

$$
P(y \mid x, D) \approx \frac{1}{M} \sum_{m=1}^M P(y \mid x, \theta^{(m)}), \quad \theta^{(m)} \sim q(\theta) \approx p(\theta \mid D)
$$

where each model \(\theta^{(m)}\) maps between sequences of inputs \(\{x_1, \dots, x_T\} = x \in \mathcal{X}\) and targets \(\{y_1, \dots , y_L\} = y \in \mathcal{Y}\). To find the uncertainty in \(y\), we then compute the entropy of this predictive posterior:

$$
PE(x) = H(Y \mid x, D) = - \int p(y \mid x) \ln p(y \mid x, D) \, dy
$$

where \(y\) is the realization of a sequence output given the preceding sequence \(x\). Quantifying the uncertainty in this Bayesian characterization, therefore, involves evaluating a computationally expensive integral \(PE(x)\).

Deep ensembles pose a significant computational challenge for uncertainty quantification in large language models (LLMs) due to the memory constraints of even powerful hardware accelerators. This can be illustrated by considering the naive memory requirements to ensemble even relatively small models, such as Llama-3-8b [AI@Meta, 2024](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md):

The memory required to store the model weights is \(2 \times n_{\text{parameters}} = 8 \times 10^{12} \approx 16.1\)GB. 

The required memory to store the Key-Value cache for a maximum supported context length of \(\approx\)8k tokens is \(2 \times 2 \times n_{\text{layers}} = 80 \times n_{\text{heads}} = 64 \times d_{\text{head}} = 128 \times l_{\text{context}} = 8192 \approx 40\)GB. 

For a total sum of \(memory_{\text{llama3}} \approx 56.1\)GB. For an ensemble of \(n = 5\) models, as in [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474), this requires over \(280\)GB of memory, necessitating four datacenter GPUs [NVIDIA Corporation, 2021](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) for inference, proving too computationally expensive for most use cases.

Instead, [Balabanov & Linander, 2024](https://arxiv.org/abs/2402.12264) propose an alternative approach utilizing a single base model backbone and training multiple separate Low-Rank Adaptations (LoRAs) [Hu et al., 2021](https://arxiv.org/abs/2106.09685) to approximate the posterior distribution \(p(\theta \mid D_{\text{finetune}})\). This setup offers a much more attractive computational complexity, as the weights of LoRAs represent only approximately 1% of the total model parameters. Additionally, recent work by [Chen et al., 2023](https://arxiv.org/abs/2310.18547) has further investigated efficient algorithms for multi-LoRA inference of a common prompt on a base model in near-constant time, making the computational costs of this setup nearly equal to ML inference in a frequentist setting.

A common alternate formalization is not approximating \(H(y \mid x, D)\) by ensembling over \(M\) models but by measuring the entropy across \(S\) samples drawn from a common model:

$$
\hat{H}^{(i)}_{\textit{S-MC}}[P(y \mid x, D)] \approx -\frac{1}{S}\sum_{s=1}^{S}\frac{1}{L^{(s)}}\ln P(y^{(s)} \mid x, D), \quad y^{(s)} \sim P(y \mid x, D)
$$

where \(y^{(s)}\) is a realization of the random variable \(y\). By measuring the semantic diversity in answers from a single model, the overhead in model hosting is greatly reduced.

[Kuhn et al., 2023](https://arxiv.org/abs/2302.09664) achieves this by sampling \((s(1),....s(M))\) sequences and using Monte-Carlo integration over all of the semantically equivalent answer sets \(C \in M\) to approximate the semantic entropy \(SE\):

$$
SE(x) \approx -|C|^{-1} \sum_{i=1}^{|C|} \log p(C_i \mid x)
$$

[Lin et al., 2023](https://arxiv.org/abs/2305.19187) also follow this pattern, using a prediction answer set \((X, \ldots, X_K)\) and any of a range of methods to quantify the set similarity and, by proxy, the uncertainty or confidence in the answer. [Duan et al., 2023](https://arxiv.org/abs/2307.01379) follows a similar rationale but works to better quantify the model uncertainty by accounting for 'linguistic redundancy', where not all tokens have equal contributions to uncertainty, achieving better-calibrated prediction intervals on a range of experiments. However, this improved model accuracy entails increased computational overhead. Both the incorporation of auxiliary models for assessing the relevance of individual tokens and sentences and the token-level relevance scoring worsen scalability.

[Hou et al., 2023](https://arxiv.org/abs/2311.08718) generates a set of \(C^{(k)}\) 'clarifying questions' for a question \(X\) with the end goal of quantifying both the total model uncertainty and the contributions of the epistemic and aleatoric terms.

Despite being generally less computationally costly than deep ensembles for large model sizes, sampling methods still require drawing \(K\) samples to accurately quantify the epistemic uncertainty associated with a model answer. This linear scaling of computational complexity with sample number can still be prohibitive for large or closed-source models.

Despite their widespread adoption, traditional Bayesian formulations of uncertainty quantification including deep-ensemble and sampling-based methods have been criticized for their computational complexity [Thaler et al., 2023](https://arxiv.org/abs/2212.07959), sometimes uncalibrated distributions [Rahaman & Thiery, 2021](https://arxiv.org/abs/2007.08792), and their requirement for prior distributional knowledge. In recent years, Conformal Prediction [Angelopoulos & Bates, 2022](https://arxiv.org/abs/2107.07511) (CP) has seen an increase in popularity for the uncertainty quantification of machine learning models as a result of its strong calibration and coverage guarantees. For example, conformal classifiers in place of a logit \(f({x_0..x_k})\) for a \(k\) class multi-classification problem construct a classification set \(C(x)\):

$$
C(x) = \{ \pi_1(x), \ldots, \pi_k(x) \}, \text{ where } k = \sup \left\{ k' : \sum_{j=1}^{k'} \hat{f}(\pi_j(x)) < \hat{q} \right\} + 1
$$

where \(\pi(x)\) is a permutation that sorts the classes from most to least likely according to \(f({x})\). The set includes classes greedily until their cumulative softmax probability exceeds a threshold \(\hat{q}\).

This adaptive construction allows the sets to be smaller for easier examples and larger for harder ones, while still guaranteeing \(1-\alpha\) coverage, i.e., \(\mathbb{P}(Y_{\text{test}} \in C(X_{\text{test}})) \geq 1-\alpha\). Making use of this attractive property, [Kumar et al., 2023](https://arxiv.org/abs/2305.18404) adapts this procedure to multiple-choice question answering where they conformal prediction sets from the logits of LLMs to output calibrated answer sets.

In a similar fashion, [Deutschmann et al., 2023](https://arxiv.org/abs/2309.03797) extend the beam-search decoding algorithm to produce calibrated output sets \(\{Y_0 \ldots Y_i\}\) in a long-form question answering task. Extending the CP procedure to black-box language models, [Su et al., 2024](https://arxiv.org/abs/2403.01216) use sample frequency and entropy to measure the nonconformity of model generations when log-probabilities are inaccessible.

Despite the significant research in designing accurate and efficient uncertainty quantification techniques for language models, the computational complexity of expressing uncertainty is still a notable barrier to industry adoption. The sheer scale of modern LLMs makes both ensembling and sample-based methods of uncertainty quantification unattractive.

Deep ensembles, even with recent innovations [Balabanov & Linander, 2024](https://arxiv.org/abs/2402.12264), still require complex hardware level implementations [Chen et al., 2023](https://arxiv.org/abs/2310.18547) to perform inference at scale.

Techniques centered around sample conformity like conformal prediction and semantic entropy have a comparatively lower computational cost, needing to host a single copy of the LLM, but require sampling a batch of sequences \(\{Y_i\}\) for every inference, increasing the requirements by a linear factor.

For 'computationally friendly' uncertainty quantification, there is ultimately a need for metrics of uncertainty expressed intrinsically by the model. To this end, there has been a significant body of work [Tian et al., 2023](https://arxiv.org/abs/2305.14975), [Zhang et al., 2024](https://api.semanticscholar.org/CorpusID:268876453) in calibrating LLMs to accurately quantify their confidence in a given answer. This has the benefit of requiring a single sample from a single model, making it an attractive proposition. This is an open area of research, with recent works [Xiong et al., 2024](https://arxiv.org/abs/2306.13063) highlighting shortcomings around poor calibration for weaker models and no guarantees of coverage.

Several promising areas of research for quantifying the uncertainty of language models can be found by exploring the wider task of machine learning uncertainty prediction. Prior networks [Malinin & Gales, 2018](https://arxiv.org/abs/1802.10501) use a small auxiliary network to compute the uncertainty in a candidate model efficiently.

[Lahlou et al., 2023](https://arxiv.org/abs/2102.08501) similarly train a small 'error predictor' network which learns the point-wise generalization error associated with a prediction, and by extension an upper bound on the epistemic uncertainty of the reference model. [Bui & Liu, 2023](https://arxiv.org/abs/2302.06495) instead directly looks at the model state, using the density function and softmax layers of the model to quantify the model's uncertainty in a manner that is rigorous under concept drift between training and inference distributions.

## References

- [Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. Attention is All You Need. Advances in Neural Information Processing Systems, 30.
- [Madsen et al., 2022](https://dl.acm.org/doi/10.1145/3502446). Madsen, A., Reddy, S., & Chandar, S. Post-hoc interpretability for neural NLP: A survey. ACM Computing Surveys, 55(8), 1-42.
- [Detommaso et al., 2024](https://api.semanticscholar.org/CorpusID:269004786). Detommaso, G., Bertran, M., Fogliato, R., & Roth, A. Multicalibration for Confidence Scoring in LLMs.
- [Sharma, 2017](http://dx.doi.org/10.1146/annurev-astro-082214-122339). Sharma, S. Markov Chain Monte Carlo Methods for Bayesian Data Analysis in Astronomy. Annual Review of Astronomy and Astrophysics, 55(1), 213-259.
- [Wang & Yeung, 2020](https://doi.org/10.1145/3409383). Wang, H., & Yeung, D.-Y. A Survey on Bayesian Deep Learning. ACM Computing Surveys, 55, 1-35.
- [Lin et al., 2022](https://aclanthology.org/2022.findings-acl.328). Lin, Z., Liu, J. Z., & Shang, J. Towards Collaborative Neural-Symbolic Graph Semantic Parsing via Uncertainty. Findings of the Association for Computational Linguistics: ACL 2022, 4160-4173.
- [Kuhn et al., 2023](https://arxiv.org/abs/2302.09664). Kuhn, L., Gal, Y., & Farquhar, S. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.
- [Kendall & Gal, 2017](https://arxiv.org/abs/1703.04977). Kendall, A., & Gal, Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
- [Balabanov & Linander, 2024](https://arxiv.org/abs/2402.12264). Balabanov, O., & Linander, H. Uncertainty quantification in fine-tuned LLMs using LoRA ensembles.
- [Lahlou et al., 2023](https://arxiv.org/abs/2102.08501). Lahlou, S., Jain, M., Nekoei, H., Butoi, V. I., Bertin, P., Rector-Brooks, J., Korablyov, M., & Bengio, Y. DEUP: Direct Epistemic Uncertainty Prediction.
- [Lin et al., 2023](https://arxiv.org/abs/2305.19187). Lin, Z., Trivedi, S., & Sun, J. Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models.
- [Malinin & Gales, 2021](https://arxiv.org/abs/2002.07650). Malinin, A., & Gales, M. Uncertainty Estimation in Autoregressive Structured Prediction.
- [Hu et al., 2021](https://arxiv.org/abs/2106.09685). Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. LoRA: Low-Rank Adaptation of Large Language Models.
- [Chen et al., 2023](https://arxiv.org/abs/2310.18547). Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. Punica: Multi-Tenant LoRA Serving.
- [Hüllermeier & Waegeman, 2021](https://doi.org/10.1007/s10994-021-05946-3). Hüllermeier, E., & Waegeman, W. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning, 110, 457-506.
- [Masegosa, 2020](https://arxiv.org/abs/1912.08335). Masegosa, A. R. Learning under Model Misspecification: Applications to Variational and Ensemble methods.
- [Barber & Bishop, 1998](http://research.microsoft.com/~cmbishop). Barber, D., & Bishop, C. M. Ensemble Learning in Bayesian Neural Networks. Neural Networks and Machine Learning, Springer, 215-233.
- [Anthony et al., 2023](https://blog.eleuther.ai/transformer-math-101/). Anthony, Q., Biderman, S., & Schoelkopf, H. Transformer Math 101. EleutherAI Blog.
- [Angelopoulos & Bates, 2022](https://arxiv.org/abs/2107.07511). Angelopoulos, A. N., & Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.
- [Deutschmann et al., 2023](https://arxiv.org/abs/2309.03797). Deutschmann, N., Alberts, M., & Martínez, M. R. Conformal Autoregressive Generation: Beam Search with Coverage Guarantees.
- [Kumar et al., 2023](https://arxiv.org/abs/2305.18404). Kumar, B., Lu, C., Gupta, G., Palepu, A., Bellamy, D., Raskar, R., & Beam, A. Conformal Prediction with Large Language Models for Multi-Choice Question Answering.
- [Su et al., 2024](https://arxiv.org/abs/2403.01216). Su, J., Luo, J., Wang, H., & Cheng, L. API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access.
- [Kadavath et al., 2022](https://arxiv.org/abs/2207.05221). Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones, A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman, S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J., Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Olsson, C., Ringer, S., Amodei, D., Brown, T., Clark, J., Joseph, N., Mann, B., McCandlish, S., Olah, C., & Kaplan, J. Language Models (Mostly) Know What They Know.
- [Tian et al., 2023](https://arxiv.org/abs/2305.14975). Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.
- [Xiong et al., 2024](https://arxiv.org/abs/2306.13063). Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.
- [Zhang et al., 2024](https://api.semanticscholar.org/CorpusID:268876453). Zhang, M., Huang, M., Shi, R., Guo, L., Peng, C., Yan, P., Zhou, Y., & Qiu, X. Calibrating the Confidence of Large Language Models by Eliciting Fidelity.
- [Malinin & Gales, 2018](https://arxiv.org/abs/1802.10501). Malinin, A., & Gales, M. Predictive Uncertainty Estimation via Prior Networks.
- [Thaler et al., 2023](https://arxiv.org/abs/2212.07959). Thaler, S., Doehner, G., & Zavadlav, J. Scalable Bayesian Uncertainty Quantification for Neural Network Potentials: Promise and Pitfalls.
- [Rahaman & Thiery, 2021](https://arxiv.org/abs/2007.08792). Rahaman, R., & Thiery, A. H. Uncertainty Quantification and Deep Ensembles.
- [Blei et al., 2017](https://doi.org/10.1080/01621459.2017.1285773). Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. Variational Inference: A Review for Statisticians. Journal of the American Statistical Association, 112(518), 859-877.
- [Chen et al., 2021](https://arxiv.org/abs/2107.03374). Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., & Zaremba, W. Evaluating Large Language Models Trained on Code.
- [Sahlin et al., 2021](https://doi.org/10.1002/ieam.4367). Sahlin, U., Helle, I., & Perepolkin, D. "This Is What We Don't Know": Treating Epistemic Uncertainty in Bayesian Networks for Risk Assessment. Integrated Environmental Assessment and Management, 17(1), 221-232.
- [NVIDIA Corporation, 2021](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). NVIDIA A100 Tensor Core GPU Datasheet.