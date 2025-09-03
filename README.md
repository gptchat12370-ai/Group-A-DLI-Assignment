# Group-A — Deep Learning IDS for IoHT (Attack-Graph)

## Overview
This repository contains the work of **Group-A** for the Deep Learning & Intrusion Detection assignment.  
Our prototype focuses on building an **intrusion detection system (IDS)** over **attack graphs** in Internet-of-Healthcare-Things (IoHT).  
The group evaluated multiple deep learning approaches, with **Shehab’s NN (BetterNN)** chosen as the **final prototype** due to its superior accuracy and AUC.

---

## Project Goals
- Take a real cybersecurity problem (attack-graph based intrusion detection).
- Use a public **synthetic dataset (Dataset-1)** for model training.
- Train different models (NN, CNN, Inception-ResNet-1D, GaussianNB, MLP).
- Achieve a success metric: **Accuracy ≥ 95% and AUC ≥ 0.98**.
- Bonus: Provide a **Gradio GUI** for interactive predictions.

---

## Dataset
- **Source**: Dataset-1 (synthetic) provided by the paper repository.
- **Shape**: Each sample is represented as a graph with:
  - **26 nodes**,  
  - **135 features per node** (57 categorical + 78 real-time measurements).  
- **Labels**: Binary labels on **7 action nodes** (privilege nodes).  
- **Splits**: Train (3000), Validation (1000), Test (1000).  
- **Ethics**: Dataset is synthetic → no sensitive real-world data.  

---

## Models Implemented
- **Shehab (Prototype)**: GNN.
- **Hamza**: CNN.
- **Omar**: InceptionResnetV2.
- **Rasheed**: MLP.
- **Seklani**: NB.

Each model was trained and evaluated with consistent splits and metrics.

---

## Prototype (Shehab NN)
**Architecture**
- Input: [Batch, 26 nodes, 135 features].
- Layers: Linear → BatchNorm → GELU → Dropout (x2) + residual skip.
- Output: Binary logits per node → aligned to action_mask → pooled.

**Training Setup**
- Optimizer: AdamW with OneCycleLR (max_lr=2e-3).
- Loss: BCEWithLogitsLoss with class imbalance weighting.
- Early stopping on Validation AUC.
- Threshold calibration on validation set.

---

## Results

### Validation Calibrated Threshold → Test Metrics

| Model         | Accuracy | Precision | Recall | F1   | AUC   |
|---------------|----:|---------:|----------:|-------:|-----:|------:|
| **Shehab BetterNN** | **97.54%** | 86.97% | 88.71% | 87.84% | **99.02%** |
| Paper NN     | 88.35% | 71.35% | 85.82% | 75.62% | 93.76% |

**Confusion (BetterNN):** TP=621, FP=93, TN=6207, FN=79.

- Achieved **target success metric** (Accuracy ≥95%, AUC ≥0.98).

---

## GUI (Bonus)
We implemented a **Gradio interface** allowing interactive testing.

- Select model (NN, CNN, GNB, Incep1D, MLP).
- Upload CSV row → get probability + prediction.  
- Uses saved weights (`shehab_nn.pt`, etc.).
- GUI supports `share=True` for public demo.
- Link: https://bdaa9f88338df6d755.gradio.live/

---

## Tools & Libraries
- **Frameworks**: PyTorch 2.6, PyTorch Geometric, scikit-learn, Keras (CPU only for GUI).
- **Other**: Gradio, Matplotlib, Pandas, Numpy.
- **Platform**: Google Colab (runtime <5 minutes).

---

## Collaboration & Git Evidence
- All members made meaningful Git commits:
  - **Shehab**: GNN Lead.
  - **Hamza**: CNN Lead.
  - **Omar**: Incep1D Lead.
  - **Rasheed**: MLP Lead.
  - **Seklani**: NB Lead.

---

## References
- Original Dataset & Paper Repo: [Attack Graph Dataset-1] link: https://github.com/zhenlus/GNN-IDS/blob/main/mulval_attack_graph/AttackGraph.dot .
- PyTorch & PyG documentation.  
- scikit-learn documentation.  
- TensorFlow/Keras documentation.
- Libraries: https://github.com/zhenlus/GNN-IDS/tree/main/src
- Paper: https://dl.acm.org/doi/10.1145/3664476.3664515
- Paper Code: https://github.com/zhenlus/GNN-IDS/blob/main/src/gnn_ids_dataset1.ipynb
- Dataset: https://github.com/zhenlus/GNN-IDS/tree/main/datasets/synt
---

## License
- Code: MIT License (update if needed).  
- Dataset: Follow original repository license.  
- Intended **for academic use only**.

---

## Verdict
✔️ **Achieved Accuracy = 97.54%, AUC = 0.990 → Success metric met.**  
✔️ Bonus GUI implemented and tested.  
✔️ Collaborative workflow with meaningful Git commits.  
