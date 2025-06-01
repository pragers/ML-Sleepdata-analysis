import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Neural Network Model for Multi-Output Regression
class SleepQualityRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SleepQualityRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)  # Two output variables: NextDayHRV, NextDayHeartRate
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x= self.dropout(x)
        x = self.leaky_relu(self.layer2(x))
        x = self.leaky_relu(self.layer3(x))
        x = self.leaky_relu(self.layer4(x))
        return self.output_layer(x)

def Lossfunction(outputs,targets):
    outputs = outputs
    targets = targets
    return nn.MSELoss()(outputs, targets)
# Train the Model
def train_model(X_tensor, y_tensor, model, num_epochs=100,batch_size=32,updateFreq= 10):
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)  # Decay every 50 epochs

    # Create DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # Mini-batch training
        for inputs, targets in dataloader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = Lossfunction(outputs, targets)
            epoch_loss += loss.item()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        if epoch % updateFreq == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}: Loss = {avg_loss}")
        # Update the learning rate
        scheduler.step()

    return model


# Make Predictions
def make_predictions(model, X_tensor, y_scaler):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predicted_scaled = model(X_tensor)

    # Inverse transform the scaled outputs back to original values
    predicted = y_scaler.inverse_transform(predicted_scaled.numpy())
    return predicted