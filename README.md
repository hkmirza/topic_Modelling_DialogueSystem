
# Topic Modelling for Dialogue Systems

This repository contains the implementation of the **Topic Modelling** experiment.  
The experiment leverages **Bag-of-Topics (BoT)**, noisy labelling, **BiLSTM networks**, and contextual information to perform dialogue segmentation and topic modelling across conversational datasets.

---

## Overview
The proposed approach combines:
- **Bag-of-Topics (BoT):** Derived from topic detection outputs.  
- **Noisy Labelling:** Weak supervision for initial utterance labels.  
- **BiLSTM Word Embeddings:** Capturing contextual dependencies.  
- **Segmentation Classifier:** Refining boundaries and generating dialogue topic segmentation.  
- **Semantic Graph Visualisation:** Representing relationships among topics and utterances.

The experiment evaluates the approach on **Switchboard**, **PersonaChat**, and **MultiWOZ** corpora, using metrics such as: (you can also find dataset in other git repository)
- Mean Absolute Error (MAE)  
- WindowDiff (WD)  
- F1-score  

---

## Installation Guide

### Install Python
If you don't have Python installed, download and install it from:
- [Python Official Website](https://www.python.org/downloads/)
Verify installation:
```sh
python --version
```
## Installation
### 1. Clone the repository:
   git clone https://github.com/yourusername/topic-modelling.git
cd topic-modelling
### 2. Install dependencies:
   pip install -r requirements.txt
   Or install manually:

pip install numpy pandas tensorflow==2.12 scikit-learn matplotlib


### Optional (for embeddings & plots):

pip install tqdm seaborn

  ## Data Format
The input dataset must be a CSV file
