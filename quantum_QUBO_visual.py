# quantum_QUBO_visual.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
import random

# Step 1: Generate simulated server load data
num_servers = 6
time_steps = 30
loads = []

for s in range(num_servers):
    base_load = random.randint(20, 60)
    trend = np.linspace(0, 10, time_steps)
    noise = np.random.normal(0, 2, time_steps)
    loads.append(base_load + trend + 5 * np.sin(np.linspace(0, 3, time_steps)) + noise)

load_df = pd.DataFrame(loads).T
load_df.columns = [f"Server_{i}" for i in range(num_servers)]

# Show sample load table
print("\n📊 Historical Server Load Data (last 30 mins):")
print(tabulate(load_df.tail(10), headers='keys', tablefmt='fancy_grid', showindex=False))

# Plot initial load chart
plt.figure(figsize=(10,5))
plt.plot(load_df)
plt.title("Server Load (CPU %) Over Time")
plt.xlabel("Time (min)")
plt.ylabel("CPU Utilization (%)")
plt.legend(load_df.columns)
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 2: Prepare data for LSTM
scaler = MinMaxScaler()
scaled = scaler.fit_transform(load_df)

X, y = [], []
for t in range(len(scaled)-3):
    X.append(scaled[t:t+3])
    y.append(scaled[t+3])
X, y = np.array(X), np.array(y)

model = Sequential([
    LSTM(32, activation='tanh', input_shape=(3, num_servers)),
    Dense(num_servers)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, verbose=0)

# Step 3: Predict future server loads
pred_scaled = model.predict(X[-1].reshape(1,3,num_servers))
pred_loads = scaler.inverse_transform(pred_scaled)[0]

pred_df = pd.DataFrame({
    'Server': [f"Server_{i}" for i in range(num_servers)],
    'Predicted_Load_%': np.round(pred_loads, 2)
})

print("\n🔮 Predicted Server Loads (Next Time Interval):")
print(tabulate(pred_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Step 4: Apply fuzzy logic categories
def fuzzy_category(load):
    if load < 35: return "Low"
    elif load < 60: return "Medium"
    else: return "High"

def fuzzy_weight(load):
    if load < 35: return 0.2
    elif load < 60: return 0.5
    else: return 1.2

pred_df["Fuzzy_Level"] = pred_df["Predicted_Load_%"].apply(fuzzy_category)
pred_df["Fuzzy_Weight"] = pred_df["Predicted_Load_%"].apply(fuzzy_weight)

print("\n🤖 Fuzzy Logic Classification:")
print(tabulate(pred_df, headers='keys', tablefmt='fancy_grid', showindex=False))

# Visualize fuzzy load levels
plt.figure(figsize=(8,5))
sns.barplot(data=pred_df, x='Server', y='Predicted_Load_%', hue='Fuzzy_Level', palette='coolwarm')
plt.title("Predicted Server Loads with Fuzzy Categories")
plt.ylabel("CPU Load (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Build QUBO matrix (for assignment optimization)
n = num_servers
Q = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            Q[i][j] = pred_df["Fuzzy_Weight"][i] * pred_df["Predicted_Load_%"][i]
        else:
            Q[i][j] = 0.1 * abs(pred_df["Predicted_Load_%"][i] - pred_df["Predicted_Load_%"][j])

plt.figure(figsize=(6,5))
sns.heatmap(Q, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("QUBO Matrix (Load Balancing Energy Landscape)")
plt.xlabel("Server")
plt.ylabel("Server")
plt.tight_layout()
plt.show()

# Step 6: Simulated annealing for approximate QUBO solution
def simulated_annealing(Q, iterations=5000, temp=100):
    n = Q.shape[0]
    x = np.random.randint(0, 2, n)
    best_x, best_energy = x.copy(), x @ Q @ x.T
    for _ in range(iterations):
        i = np.random.randint(0, n)
        x[i] = 1 - x[i]
        energy = x @ Q @ x.T
        if energy < best_energy or np.exp((best_energy - energy)/temp) > random.random():
            best_x, best_energy = x.copy(), energy
        temp *= 0.995
    return best_x, best_energy

solution, energy = simulated_annealing(Q)
assignments = {f"Server_{i}": int(solution[i]) for i in range(n)}

result_df = pd.DataFrame({
    'Server': [f"Server_{i}" for i in range(n)],
    'Assigned_Request': [assignments[f"Server_{i}"] for i in range(n)],
    'Predicted_Load_%': np.round(pred_loads, 2),
    'Fuzzy_Level': pred_df["Fuzzy_Level"],
    'Fuzzy_Weight': pred_df["Fuzzy_Weight"]
})

print("\n⚙️ Final Optimized Assignment via QUBO:")
print(tabulate(result_df, headers='keys', tablefmt='fancy_grid', showindex=False))
print(f"\n🧠 Minimum Energy State (cost): {energy:.3f}")

# Step 7: Visualize final assignment
plt.figure(figsize=(8,5))
sns.barplot(data=result_df, x='Server', y='Assigned_Request', palette='Blues_d')
plt.title("Final Request Assignment per Server (QUBO Output)")
plt.ylabel("Assigned Requests (binary 0/1)")
plt.tight_layout()
plt.show()
