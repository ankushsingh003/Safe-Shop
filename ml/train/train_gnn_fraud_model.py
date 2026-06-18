"""
Fraud Ring Detection — Graph Neural Network (GNN)
Layer 1 of Upgrade Roadmap: GNN for Fraud Rings

This model detects fraudulent "rings" by looking at connections between transactions
sharing the same IP Address or Customer ID.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# CONFIG
MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# --------------------------------------------------------------------------
# 1. GRAPH DATA GENERATOR / LOADER
# --------------------------------------------------------------------------
def generate_graph_data(n_orders=5000):
    """
    Loads real Kaggle dataset if available to build the transaction graph.
    Otherwise, generates synthetic transactions and forms fraud rings by sharing IP/Device.
    """
    csv_path = "ml/data/Fraudulent_E-Commerce_Transactions.csv"
    if os.path.exists(csv_path):
        print(f"      [INFO] Loading real Kaggle dataset for GNN from {csv_path}...")
        df = pd.read_csv(csv_path)
        if len(df) > n_orders:
            df = df.sample(n=n_orders, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Ensure standard column mapping
        df = df.rename(columns={
            "Transaction ID": "order_id",
            "Customer ID": "user_id",
            "Transaction Amount": "amount",
            "IP Address": "ip",
            "Device Used": "device",
            "Is Fraudulent": "is_fraud"
        })
    else:
        print(f"      [INFO] Kaggle CSV not found at {csv_path}. Generating synthetic Graph data...")
        np.random.seed(RANDOM_SEED)
        
        # Base features
        order_amounts = np.random.lognormal(4.5, 0.8, n_orders)
        
        # Shared attributes (IPs and Devices)
        n_ips = int(n_orders * 0.7)  # 70% unique IPs
        n_devices = int(n_orders * 0.8) # 80% unique devices
        
        ips = [f"ip_{i}" for i in np.random.randint(0, n_ips, n_orders)]
        devices = [f"dev_{i}" for i in np.random.randint(0, n_devices, n_orders)]
        
        df = pd.DataFrame({
            'order_id': range(n_orders),
            'amount': order_amounts,
            'ip': ips,
            'device': devices,
            'is_fraud': 0
        })

        # Create Fraud Rings (Orders sharing the same IP in rapid succession)
        fraud_ips = np.random.choice(ips, 10, replace=False)
        df.loc[df['ip'].isin(fraud_ips), 'is_fraud'] = 1
        
        # Add some random fraud too
        random_fraud = np.random.choice(range(n_orders), int(n_orders * 0.01))
        df.loc[random_fraud, 'is_fraud'] = 1

    # --------------------------------------------------------------------------
    # 2. CONSTRUCT EDGES (Connections based on shared IP or Customer/User ID)
    # --------------------------------------------------------------------------
    edge_index = []
    
    # Connect transactions sharing the same IP
    if 'ip' in df.columns:
        ip_groups = df.groupby('ip').indices
        for ip, indices in ip_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        edge_index.append([indices[i], indices[j]])
                        edge_index.append([indices[j], indices[i]])

    # Connect transactions sharing the same Customer/User ID
    if 'user_id' in df.columns:
        user_groups = df.groupby('user_id').indices
        for uid, indices in user_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        edge_index.append([indices[i], indices[j]])
                        edge_index.append([indices[j], indices[i]])

    if len(edge_index) == 0:
        # Fallback empty edge index if no connections
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Scale node features
    scaler = StandardScaler()
    amount_scaled = scaler.fit_transform(df[['amount']])
    
    x = torch.tensor(amount_scaled, dtype=torch.float)
    y = torch.tensor(df['is_fraud'].values, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y), df

# --------------------------------------------------------------------------
# 3. GNN MODEL ARCHITECTURE (GraphSAGE)
# --------------------------------------------------------------------------
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --------------------------------------------------------------------------
# 4. TRAINING LOOP
# --------------------------------------------------------------------------
def train():
    print("Generating Graph Data...")
    data, df = generate_graph_data(n_orders=2000)
    print(f"Graph created: {data.num_nodes} nodes, {data.num_edges} edges")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = FraudGNN(in_channels=1, hidden_channels=16, out_channels=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train/Test Split (Node-level)
    indices = torch.randperm(data.num_nodes)
    train_idx = indices[:int(0.8 * data.num_nodes)]
    test_idx  = indices[int(0.8 * data.num_nodes):]

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = float(pred[test_idx].eq(data.y[test_idx]).sum().item())
    acc = correct / len(test_idx)
    print(f'\nFinal Test Accuracy: {acc:.4f}')

    # Save model in ml/models
    model_path = os.path.join(MODEL_DIR, "gnn_fraud_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"GNN Model saved to {model_path}")

if __name__ == "__main__":
    train()
