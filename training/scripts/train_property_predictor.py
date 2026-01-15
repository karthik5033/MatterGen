import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from training.models.gnn import MaterialGNN

class MaterialDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.features = torch.tensor(self.df[['formation_energy', 'stability']].values, dtype=torch.float32)
        self.targets = torch.tensor(self.df['band_gap'].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def train():
    # Hyperparameters
    input_dim = 2
    hidden_dim = 64
    output_dim = 1
    lr = 0.001
    epochs = 10
    
    # Data
    dataset = MaterialDataset('training/datasets/synthetic_materials.csv')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model
    model = MaterialGNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            # In a real GNN, we would pass edge_indices as well
            outputs = model(batch_x, None) 
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}")

if __name__ == "__main__":
    # Ensure data exists
    if not os.path.exists('training/datasets/synthetic_materials.csv'):
        from training.datasets.synthetic_generator import generate_synthetic_materials
        generate_synthetic_materials()
    
    import os
    train()
