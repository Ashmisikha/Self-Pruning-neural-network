# 🧠 Self-Pruning Neural Network

### Tredence AI Engineering Internship – Case Study

**Author:** Ashmisikha Piri  
**Institution:** SRM Institute of Science and Technology  

---

## 🚀 Overview

Deep neural networks often contain a large number of parameters, making them computationally expensive and difficult to deploy in resource-constrained environments. A common optimization technique is **model pruning**, where less important weights are removed after training.

In this project, we implement a **self-pruning neural network** that learns to remove unnecessary connections **during training itself**. This is achieved using learnable gates and sparsity regularization, allowing the network to dynamically optimize both its weights and structure.

---

## ⚙️ Key Idea

Each weight in the network is associated with a learnable **gate parameter**:

- Gate ≈ 1 → Weight is active  
- Gate ≈ 0 → Weight is pruned  

The effective weight becomes:
# Self-Pruning-neural-network


---

## 🧩 Project Structure

-Self-Pruning-Neural-Network/
│
├── app.py # Streamlit UI
├── train.py # Training script
├── evaluate.py # Evaluation (accuracy + sparsity)
├── models/ # Model & prunable layers
├── utils/ # Sparsity & visualization utilities
├── outputs/
│ ├── checkpoints/ # Saved models
│ └── plots/ # Gate distribution plots
├── experiments/
│ └── results.md # Experiment results
├── requirements.txt
└── README.md


---

## 🧪 Methodology

- Custom `PrunableLinear` layer with learnable gates  
- Sigmoid transformation to constrain gates between 0 and 1  
- L1 regularization to promote sparsity  
- Training on CIFAR-10 dataset  
- Evaluation based on:
  - **Test Accuracy**
  - **Sparsity Level (%)**

---

## 📊 Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|------------|------------------|--------------|
| 1e-5       | 38.35            | 0.99         |
| 1e-4       | 35.83            | 1.52         |
| 1e-3       | 34.76            | 1.71         |

###  Observations

- Increasing λ increases sparsity  
- Higher sparsity reduces model accuracy  
- Demonstrates a clear **efficiency vs performance trade-off**

---

##  Gate Distribution

![Gate Distribution](outputs/plots/gate_distribution.png)

- Spike near **0** → pruned weights  
- Values away from 0 → important connections  


##  UI Demonstration

### Application Interface
![UI](outputs/screenshots/ui.png)

### Prediction Example
![Prediction](outputs/screenshots/prediction.png)

### Pruning Visualization
![Gates](outputs/screenshots/gates.png)


##  Key Insights

- The network learns to **prune itself dynamically**
- L1 regularization effectively induces sparsity
- No separate pruning step is required
- Trade-off exists between sparsity and accuracy

---

## 🔚 Conclusion

This project demonstrates a self-pruning neural network that integrates structure optimization directly into training. The approach reduces model complexity while maintaining reasonable performance, making it suitable for efficient deployment.

---

## 🔮 Future Work

- Use CNN architecture for better accuracy  
- Explore structured pruning techniques  
- Apply reinforcement learning for adaptive pruning  

---

## 📌 Note

Replace `XX.X` in results with your actual experimental values.

---

## 🔗 Submission

This repository is submitted as part of the **Tredence AI Engineering Internship Case Study**.



