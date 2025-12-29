from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# -----------------------------
# 1. Initialize Backends Storage
# -----------------------------
backends = {}  # Will store metrics for each backend

# -----------------------------
# 2. Fuzzy Logic Function
# -----------------------------
def fuzzy_score(cpu, queue_len, latency, throughput, error_rate):
    # Penalize high CPU, queue, latency, errors; reward throughput
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
    inputs = np.array([
        row["cpu"], row["queue_len"], row["latency_ms"], 
        row["throughput"], row["error_rate"]
    ])
    # Normalize roughly
    inputs_norm = inputs / np.array([100, 20, 300, 100, 0.05])
    
    f_score = fuzzy_score(*inputs_norm)
    nn_score = nn_model.forward(inputs_norm)
    
    return f_score + nn_score

# -----------------------------
# 5. Compute Backend Weights
# -----------------------------
def compute_backend_weights():
    avg_scores = {}
    for backend, df in backends.items():
        if df.empty:
            avg_scores[backend] = 0
            continue
        df["hybrid_score"] = df.apply(compute_hybrid_score, axis=1)
        avg_scores[backend] = df["hybrid_score"].mean()
    
    # Normalize to sum=1
    total = sum(avg_scores.values()) or 1
    weights = {b: round(score/total, 3) for b, score in avg_scores.items()}
    return weights

# -----------------------------
# 6. Flask Endpoints
# -----------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Hybrid Fuzzy-NN Load Balancer API Running",
        "available_endpoints": ["/metrics", "/route", "/stats"]
    })

@app.route("/metrics", methods=["POST"])
def add_metrics():
    """
    Example JSON payload:
    {
        "backend_id": "b1",
        "cpu": 45,
        "queue_len": 8,
        "latency_ms": 120,
        "throughput": 50,
        "error_rate": 0.01
    }
    """
    data = request.get_json()
    backend_id = data.get("backend_id")
    
    if backend_id not in backends:
        backends[backend_id] = pd.DataFrame(columns=["cpu","queue_len","latency_ms","throughput","error_rate"])
    
    # Append new metrics
    backends[backend_id] = pd.concat([
        backends[backend_id],
        pd.DataFrame([{
            "cpu": data["cpu"],
            "queue_len": data["queue_len"],
            "latency_ms": data["latency_ms"],
            "throughput": data["throughput"],
            "error_rate": data["error_rate"]
        }])
    ], ignore_index=True)
    
    return jsonify({"message": f"Metrics added for {backend_id}"}), 200

@app.route("/route", methods=["GET"])
def route_weights():
    weights = compute_backend_weights()
    return jsonify({"backend_weights": weights})

@app.route("/stats", methods=["GET"])
def backend_stats():
    stats = {b: df.tail(5).to_dict(orient="records") for b, df in backends.items()}
    return jsonify(stats)

# -----------------------------
# 7. Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
