import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
synthetic_data = {
    'employee_id': range(1, 11),
    'salary': np.random.uniform(50000, 150000, 10),
    'title': np.random.choice(['Engineer', 'Manager', 'Analyst', 'Director'], 10),
    'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], 10),
    'tenure': np.random.uniform(1, 15, 10),
    'attrition': np.random.choice([0, 1], 10, p=[0.7, 0.3])  # 30% attrition rate
}

# Convert to DataFrame
df = pd.DataFrame(synthetic_data)
print("Synthetic Data:")
print(df)

# Preprocessing
class HRDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        # Label encode categorical columns
        categorical_cols = ['title', 'department']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col])
        
        # Scale numerical columns
        numerical_cols = ['salary', 'tenure']
        data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        return data
    
    def transform(self, data):
        for col, encoder in self.label_encoders.items():
            data[col] = encoder.transform(data[col])
        numerical_cols = ['salary', 'tenure']
        data[numerical_cols] = self.scaler.transform(data[numerical_cols])
        return data

# Custom Dataset
class HRDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Neural Network Model
class HRAttritionModel(nn.Module):
    def __init__(self, input_size):
        super(HRAttritionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Prepare data
preprocessor = HRDataPreprocessor()
processed_data = preprocessor.fit_transform(df.copy())

# Prepare features and target
feature_cols = ['salary', 'title', 'department', 'tenure']
X = processed_data[feature_cols].values
y = processed_data['attrition'].values

# Create dataset and dataloader
dataset = HRDataset(X, y)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Initialize model, loss function, and optimizer
model = HRAttritionModel(len(feature_cols))
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/hr_attrition_experiment')

# Training loop with gradient tracking
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_features, batch_targets in dataloader:
        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets.view(-1, 1))
        
        # Backward pass
        optimizer.zero_grad()
            # Record gradients before backward pass
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        loss.backward()
        
        # Log gradients and parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients_after_backward/{name}', param.grad, epoch)
                writer.add_histogram(f'parameters_after_backward/{name}', param.data, epoch)
        
        optimizer.step()
        total_loss += loss.item()
    
    # Log average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Add model graph to TensorBoard
dummy_input = torch.randn(1, len(feature_cols))
writer.add_graph(model, dummy_input)
writer.close()

# Function to make predictions
def predict_attrition(model, features):
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features)
        prediction = model(features_tensor)
        return prediction.item()

# Test the model with a sample employee
sample_employee = {
    'salary': 80000,
    'title': 'Engineer',
    'department': 'IT',
    'tenure': 5
}

# Preprocess sample
sample_df = pd.DataFrame([sample_employee])
processed_sample = preprocessor.transform(sample_df)
sample_features = processed_sample[feature_cols].values

# Make prediction
attrition_prob = predict_attrition(model, sample_features)
print(f"\nPredicted attrition probability for sample employee: {attrition_prob:.2f}")

# Print model architecture
print("\nModel Architecture:")
print(model)

# Save the model
torch.save(model.state_dict(), 'hr_attrition_model.pth')
print("\nModel saved to 'hr_attrition_model.pth'")


import torch
from torch.utils.tensorboard import SummaryWriter

# Load the saved model
model = HRAttritionModel(input_size=4)  # Initialize with same architecture
model.load_state_dict(torch.load('hr_attrition_model.pth'))
model.eval()

# Create a TensorBoard writer
writer = SummaryWriter('model_visualization')

# Create dummy input with correct shape
dummy_input = torch.randn(1, 4)  # Adjust size based on your input features

# Add graph to TensorBoard
writer.add_graph(model, dummy_input)
writer.close()
