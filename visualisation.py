import matplotlib.pyplot as plt
import torch

def plot_gate_distribution(model, save_path):
    gates_all = []

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            gates_all.extend(gates)

    plt.hist(gates_all, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()
