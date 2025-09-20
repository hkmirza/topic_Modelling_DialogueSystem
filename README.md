
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

## Requirements

To run this experiment, ensure you have **Python 3.8+** installed.  
All dependencies are listed in the `requirements.txt` file.

Install the dependencies with:

```bash
pip install -r requirements.txt

## Clone Repository

git clone https://github.com/yourusername/topic-modelling.git
cd topic-modelling


## Requirements requirements:

pip install -r requirements.txt
