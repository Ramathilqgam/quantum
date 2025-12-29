import numpy as np
import pandas as pd

# -----------------------------
# 1. Simulate Backend Metrics
# -----------------------------
# Example: 5 backends, 100 data points per backend
num_backends = 5
num_samples = 100

# Columns: CPU%, Queue Length, Latency(ms), Throu
# hput, Error Rate
columns = ["cpu", "queue_len", "latency_ms", "throughput", "error_rate"]

# Generate random data
np.random.seed(42)
data = {
    f"b{i+1}": pd.DataFrame({
        "cpu": np.random.randint(10, 90, num_samples),
        "queue_len": np.random.randint(0, 20, num_samples),
        "latency_ms": np.random.randint(50, 300, num_samples),
        "throughput": np.random.randint(10, 100, num_samples),
        "error_rate": np.random.rand(num_samples) * 0.05
    })
    for i in range(num_backends)
}

# -----------------------------
# 2. Fuzzy Logic Controller
# -----------------------------
def fuzzy_score(cpu, queue_len, latency, throughput, error_rate):
    """
    Simple fuzzy scoring:
    - Penalize high CPU, queue, latency, error
    - Reward high throughput
    """
    score = 0
    score -= cpu * 0.4
    score -= queue_len * 0.3
    score -= latency * 0.2
    score += throughput * 0.5
    score -= error_rate * 50
    return score

# -----------------------------
# 3. Neural Network Component
# -----------------------------
# Simple feed-forward NN with one hidden layer
class SimpleNN:
    def __init__(self, input_dim):
        np.random.seed(42)
        self.weights1 = np.random.rand(input_dim, 6)
        self.bias1 = np.random.rand(6)
        self.weights2 = np.random.rand(6, 1)
        self.bias2 = np.random.rand(1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        h = self.relu(np.dot(x, self.weights1) + self.bias1)
        out = np.tanh(np.dot(h, self.weights2) + self.bias2)
        return out[0]

nn_model = SimpleNN(input_dim=5)

# -----------------------------
# 4. Compute Hybrid Score
# -----------------------------
def compute_hybrid_score(row):
    inputs = np.array([row["cpu"], row["queue_len"], row["latency_ms"],
                       row["throughput"], row["error_rate"]])
    # Normalize inputs roughly
    inputs_norm = inputs / np.array([100, 20, 300, 100, 0.05])
    
    # Fuzzy score
    f_score = fuzzy_score(*inputs_norm)
    
    # NN score
    nn_score = nn_model.forward(inputs_norm)
    
    # Combine
    return f_score + nn_score

# -----------------------------
# 5. Calculate Weights for Each Backend
# -----------------------------
def compute_backend_weights(data_dict):
    avg_scores = {}
    for backend, df in data_dict.items():
        df["hybrid_score"] = df.apply(compute_hybrid_score, axis=1)
        avg_scores[backend] = df["hybrid_score"].mean()
    
    # Normalize to sum=1
    total = sum(avg_scores.values())
    weights = {b: round(score/total, 3) for b, score in avg_scores.items()}
    return weights

# -----------------------------
# 6. Run Simulation
# -----------------------------
weights = compute_backend_weights(data)
print("Backend Weights (Adaptive Load Balancing):")
for backend, w in weights.items():
    print(f"{backend}: {w}")

# -----------------------------
# 7. Optional: Inspect Sample Data
# -----------------------------
print("\nSample Metrics for b1:")
print(data["b1"].head())
