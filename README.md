# University of Delaware CISC 684 Final Project
This is the repository for a final project for the Introduction to Machine Learning course
at the University of Delaware. The purpose of this project is to compare the accuracy and effectiveness of a
Convolutional Neural Network and a Multi-Class Bayes Classifier.

## Dataset
I based this project heavily off an existing project and report called [TrashNet](https://github.com/garythung/trashnet), it attempts to solve the same problem I am, but uses slightly different methods. For one, they use an SVM instead of a Bayesian Classifier.

Thankfully, the dataset has been posted on [HuggingFace](https://huggingface.co/datasets/garythung/trashnet), and doesn't need to be added to the repository as HuggingFace allows for direct imports of their datasets. It contains roughly ~5000 images of different classes of recyclables and a trash class. There is a somewhat severe class imbalance in the dataset that made training harder.
- Paper: 23.5%
- Metal: 16.2%
- Glass: 19.8%
- Plastic: 19.1%
- Cardboard: 15.9%
- Trash: 5.4%

## Models

### Convolutional Neural Network

### Bayes Classifier

# Running the Models

## Prerequisites

[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed on your local machine,
runs project in a virtual environment.

## Commands

Create a virtual environment
```
conda env create -f environment.yml 
```
This will also install all necessary dependicies. 

Activate
```
conda activate bayescnn
```
I was running the Jupyter notebooks through VS Code IDE, but the can also be ran at the command line. This project uses the DirectML library due to the computational restrictions of my machine. Unfortunately I have an AMD GPU so I couldn't parallelize training without using DirectML.
