# CIFAR-10 CNN Classification Pipeline

This project implements an end-to-end CIFAR-10 image classification
pipeline using PyTorch, Git, DVC and DagsHub.

## Dataset

CIFAR-10 contains 60,000 RGB images belonging to 10 classes.

## Pipeline

1. Prepare raw data
2. Preprocess data
3. Train CNN
4. Evaluate model

## Tools

- Python
- PyTorch
- Git
- DVC
- DagsHub

cifar10-cnn-pipeline/
│
├── src/
│   ├── prepare.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── metrics.json
├── README.md
├── .gitignore
└── requirements.txt