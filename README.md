# Athena: Enhancing Code Understanding for Impact Analysis by Combining Transformers and Program Dependence Graphs

## Introduction
This is the official codebase for the paper "Enhancing Code Understanding for Impact Analysis by
Combining Transformers and Program Dependence Graphs". In this work, We leverage neural code models including CodeBERT, UniXCoder, and GraphCodeBERT, prominent Transformer-based code models, for initial method embedding extraction. These pre-trained neural code models are fine-tuned on code
search to learn richer representations that are aware of underlying code intent and potentially
transferring the additional knowledge learnt from code search to IA. To integrate the global
dependence information into local code semantics, the initial method embeddings are further
enhanced using an embedding propagation strategy inspired by graph convolutional networks
(GCN) [Kipf and Welling 2016] based on the constructed dependence graphs.


## Baseline
Typical IA techniques require a seed/starting entity to perform the analysis. Although the latest conceptual IA technique [Wang et al. 2018](https://pdfs.semanticscholar.org/9a5a/097f704dff272f32568ec7dc9608b5859972.pdf) starts with the change request in natural language form, we starts with the code entity to perform IA following most of existing IA techniques [Kagdi et al. 2013](https://www.cs.wm.edu/~denys/pubs/EMSE-MSR&IR-IA-Preprint.pdf);  [Cai et al. 2016](https://dl.acm.org/doi/10.1145/2894751); [Gyori et al. 2017](https://dl.acm.org/doi/10.1145/3092703.3092719); [Liu et al. 2018](https://onlinelibrary.wiley.com/doi/10.1002/smr.1960). Given that [Wang et al. 2018]
has not made their implementation publicly available, we directly use LSI and doc2vec (used in their paper) independently as conceptual IA baselines and evaluate these two models on our untangled benchmark Alexandria. Also, LSI is the most commonly used model for obtaining code semantics for existing conceptual IA techniques.


## Dependency
- CUDA 11.0
- python 3.7
- pytorch 1.7.1
- torchvision 0.8.2


## Installation

```bash
git clone https://anonymous.4open.science/r/Athena-60D4/
cd Athena
pip install -r requirements.txt
```

## Evaluation Benchmark
To evaluate Athena for the task of impact analysis, we created a large-scale benchmark, called Alexandria, that leverages an existing dataset of fine-grained, manually untangled commit information from bug-fixes. The benchmark consists of 910 commits across 25 open-source Java projects, which we use to construct 4,405 IA tasks. The benchmark is available at https://anonymous.4open.science/r/Athena-60D4/dataset/alexandria.csv.

## Reproduce Results

```bash
python main.py \
    --project_path=./projects \
    --pretrained_model_name=microsoft/graphcodebert-base \
    --finetuned_model_path=./finetuned_models/graphcodebert.bin \
    --lang=java \
    --output_dir=./athena_reproduction_package/results/graphcodebert \
```

## Fine-tuned models
The code search task is used as the proxy for the impact analysis. Specifically, we fine-tune the pre-trained models for code search based on the dataset of CodeSearchNet Java split. The fine-tuned models will be available soon.

