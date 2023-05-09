# Athena: Combining Call Graphs and Neural Code Semantics to Improve Automated Impact Analysis

## Introduction
This is the official codebase for the paper "Combining Call Graphs and Neural Code Semantics to Improve Automated Impact Analysis". In this work, we use three large pre-trained models, namely CodeBERT, GraphCodeBERT and UniXcoder, to obtain the neural semantics of the code and adpot an embedding propogation strategy based on the call graph information to update the code semantics for accurate impact analysis.

## Dependency
- CUDA 11.0
- python 3.7
- pytorch 1.7.1
- torchvision 0.8.2


## Installation

```bash
git clone https://anonymous.4open.science/r/Athena-6557/
cd athena
pip install -r requirements.txt
```

## Evaluation Benchmark
To evaluate Athena for the task of impact analysis, we created a large-scale benchmark, called Alexandria, that leverages an existing dataset of fine-grained, manually untangled commit information from bug-fixes. The benchmark consists of 910 commits across 25 open-source Java projects, which we use to construct 4,405 IA tasks. The benchmark is available at https://anonymous.4open.science/r/Athena-6557/dataset/alexandria.csv.

## Reproduce Results

```bash
python main.py \
    --project_path=./projects \
    --pretrained_model_name=microsoft/graphcodebert-base \
    --finetuned_model_path=./finetuned_models/graphcodebert.bin \
    --lang=java \
    --output_dir=./results \
    --version=athena
```

Our results can be found at https://anonymous.4open.science/r/Athena-6557/results/.


## Fine-tuned models
The code search task is used as the proxy for the impact analysis. Specifically, we fine-tune the pre-trained models for code search based on the dataset of CodeSearchNet java split. The fine-tuned models will be available soon.

