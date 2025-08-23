# Group A: DLI Assignment - Prototype Overview

## Project Overview

This repository contains the work for **Group A's DLI Assignment** at [Your Institution Name]. The project is focused on **intrusion detection systems (IDS)** using **graph neural networks (GNN)**, implemented and evaluated on the **CICIDS2017 dataset**. Our prototype uses **AutoML techniques** like **AutoDP (Data Preprocessing)** and **AutoFE (Feature Engineering)** to automate model training, feature selection, and hyperparameter tuning, with the goal of achieving a robust and scalable IDS solution.

## Prototype Architecture

Our prototype follows a **graph-based model** that transforms network traffic data into a graph, where nodes represent **flows** or **hosts**, and edges are based on **similarities** (e.g., k-NN or feature correlations). We use **Graph Convolution Networks (GCN)** or **GraphConv layers** to extract relational features from the graph and a **classification head** to predict labels (attack or benign).

### Key Steps in the Pipeline:
1. **Data Preprocessing**:  
   - Data cleaning (handling missing values, encoding categorical features, etc.)
   - Feature scaling and normalization
   - Train/Validation/Test splits

2. **Graph Construction**:  
   - Nodes are constructed from **network flow data** (features such as packet length, flow duration).
   - Edges are formed based on the similarity between flows or hosts.

3. **Model Training (Graph Neural Network)**:  
   - **GraphConv Layers** (GCN-based) are used for **feature learning** on the graph.
   - Dropout layers are applied for **regularization**.
   - Global pooling (mean/max) is applied to get a fixed-size graph representation.

4. **Final Classifier**:  
   - **Dense Layer** followed by a **Softmax/Logits** output layer for classification.
   - Model evaluation is performed using common metrics: **Accuracy, Precision, Recall, F1-Score**.

5. **Evaluation Metrics**:
   - **Accuracy**: Overall percentage of correct predictions.
   - **Precision**: Percentage of correctly predicted positive instances out of all predicted positives.
   - **Recall**: Percentage of correctly predicted positive instances out of all actual positives.
   - **F1-Score**: Harmonic mean of Precision and Recall.

## Results

- **Precision**: ~**88.69%**
- **Recall**: ~**86.88%**
- **F1-Score**: ~**87.77%**
- **Accuracy**: ~**95.18%**
- **Model size**: ~**0.145 million parameters**
- **Inference time**: ~**0.02 ms/sample** (average for a batch of 5,661 samples)

### Comparison with Paper (IEEE TNSM 2024)

We compared our **GNN-based model** to a baseline **MLP model** reported in the paper:
- **Paper’s MLP (CICIDS2017)**:
  - **Accuracy**: **85.97%**
  - **Precision**: **92.07%**
  - **Recall**: **26.56%**
  - **F1**: **44.83%**
- **Our GNN-based model** (with **ADASYN balancing** and **threshold tuning**):
  - **Accuracy**: **95.18%**
  - **Precision**: **88.69%**
  - **Recall**: **86.88%**
  - **F1**: **87.77%**

---

## Key Features of Our Prototype

- **Automated Data Preprocessing (AutoDP)**:  
  We use techniques like **missing value imputation**, **feature scaling**, and **label encoding** to automatically prepare the dataset for model training.

- **Feature Engineering (AutoFE)**:  
  We automate feature selection using methods like **Recursive Feature Elimination (RFE)** and **Pearson Correlation** to reduce redundant features and retain only the most informative ones for the model.

- **Graph Neural Networks**:  
  Instead of traditional models like MLP, we use **graph neural networks (GNN)**, specifically **GraphConv layers**, to learn better representations of the network traffic data by considering the relationships between different network flows.

- **Threshold Tuning**:  
  The decision threshold for classification was tuned using **validation data**, optimizing **accuracy** and **F1-score**.

---

## Future Work & Next Steps

In the next sprint, we plan to **improve model performance** by integrating **knowledge distillation**. The idea is to use a strong **LightGBM** model as a teacher and train the **MLP** student model to learn from both the labels and soft predictions from the teacher. This process should help our **MLP** capture feature interactions that **LightGBM** excels at, while maintaining its small model size and fast inference times.

---

## Setup and Installation

To get started with this project, clone this repository and follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/gptchat12370-ai/Group-A-DLI-Assignment.git
   cd Group-A-DLI-Assignment
