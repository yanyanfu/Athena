# Athena: Enhancing Code Understanding for Impact Analysis by Combining Transformers and Program Dependence Graphs

## Introduction
This is the official codebase for the paper "Enhancing Code Understanding for Impact Analysis by
Combining Transformers and Program Dependence Graphs". In this work, We leverage neural code models including CodeBERT, UniXCoder, and GraphCodeBERT, prominent Transformer-based code models, for initial method embedding extraction. These pre-trained neural code models are fine-tuned on code
search to learn richer representations that are aware of underlying code intent and potentially
transferring the additional knowledge learnt from code search to IA. To integrate the global
dependence information into local code semantics, the initial method embeddings are further
enhanced using an embedding propagation strategy inspired by graph convolutional networks
(GCN) [Kipf and Welling 2016] based on the constructed dependence graphs.


## Dependency
- CUDA 11.0
- python 3.7
- pytorch 1.7.1
- torchvision 0.8.2


## Installation

```bash
git clone https://github.com/yanyanfu/Athena.git
cd Athena
pip install -r requirements.txt
```

## Evaluation Benchmark
To evaluate Athena for the task of impact analysis, we created a large-scale benchmark, called Alexandria, that leverages an existing dataset of fine-grained, manually untangled commit information from bug-fixes. The benchmark consists of 910 commits across 25 open-source Java projects, which we use to construct 4,405 IA tasks. The benchmark is available at dataset/alexandria.csv.

## Reproduce Results

```bash
python main_multi.py \
    --project_path=./projects \
    --pretrained_model_name=microsoft/graphcodebert-base \
    --finetuned_model_path=./finetuned_models/graphcodebert.bin \
    --lang=java \
    --output_dir=./athena_reproduction_package/results/graphcodebert \
```

## Fine-tuned models
The code search task is used as the proxy for the impact analysis. Specifically, we fine-tune the pre-trained models for code search based on the dataset of CodeSearchNet Java split. The fine-tuned models are available at https://drive.google.com/drive/folders/1b7xkAA5XWSY2io6smAk-c7PeTdxt5I5a?usp=drive_link.

