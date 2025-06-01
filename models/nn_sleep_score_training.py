import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import torch_directml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from preprocessing.data_preprocessing import preprocess_data
import preprocessing.data_preprocessing
from models.sleep_score_model_class import SleepScoreModel

def variance_loss(y_pred):
    return -torch.var(y_pred)

def unsupervised_consistency_loss(model, X_batch, noise_std=0.001):
    X_batch = X_batch  # <--- Force move to GPU
    noise = torch.normal(mean=0.0, std=noise_std, size=X_batch.shape, device=X_batch.device)
    x_noisy = X_batch + noise

    model.eval()
    with torch.no_grad():
        y_pred_clean = model(X_batch)
    model.train()

    y_pred_noisy = model(x_noisy)
    lossfunc = nn.MSELoss()
    return lossfunc(y_pred_clean, y_pred_noisy)


def weighted_loss(y_pred, y_true, weight):
    # Loss based on MSE (mean squared error)
    loss_fn = nn.MSELoss()
    return weight * loss_fn(y_pred, y_true)

def train_score_model_dynamic(X_tensor, y_score_tensor, model, num_epochs=500, lr=0.001,
                              initial_weight=1.0, decay_rate=0.99, updateRate=100, batch_size=32):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)

    dataset = TensorDataset(X_tensor, y_score_tensor)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Blend factor goes from 1 → 0 as epoch increases
        pseudo_label_weight = initial_weight * (decay_rate ** epoch)
        blend_factor = max(pseudo_label_weight, 0.5)  # Keep minimum influence from pseudo-labels

        for batch_x, batch_y in dataloader:
            batch_x = batch_x
            batch_y = batch_y
            optimizer.zero_grad()

            # Supervised loss with pseudo labels
            y_pred = model(batch_x)
            supervised_loss = weighted_loss(y_pred, batch_y, weight=1.0)

            # Unsupervised consistency loss
            consistency_loss = unsupervised_consistency_loss(model, batch_x)

            # Hybrid loss: blend_factor for supervised, (1 - blend_factor) for unsupervised
            loss = abs(blend_factor * supervised_loss + (1 - blend_factor) * (consistency_loss + 0.01 * variance_loss(y_pred)))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % updateRate == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Blend Factor: {blend_factor:.4f}")

    return model

data = pd.read_csv('B:\MachineLearning\TrainingData.csv')
data['SleepGoal'] = 8.0
X_tensor, y_tensor, x_scaler, y_scaler, retained_indices, imputed_data = preprocess_data(data)

# Average the scores across weight sets
model = SleepScoreModel(input_size=X_tensor.shape[1])
model.load_state_dict(torch.load('trained_models/sleep_model.pth'))
trained_model = train_score_model_dynamic(X_tensor,y_tensor,model,num_epochs=2000,initial_weight=1,decay_rate=0.998,updateRate = 100)
trained_model.eval()
with torch.no_grad():
    y_pred = trained_model(X_tensor)
predicted_scores = y_scaler.inverse_transform(y_pred.numpy())
predicted_scores = np.clip(predicted_scores,0,100)
print(f"Predictions: {predicted_scores[:10]}")
torch.save(model.state_dict(), 'trained_models/sleep_model.pth')
data_with_scores = data.copy()
data_with_scores.loc[imputed_data.index, 'SleepScore'] = predicted_scores
data_with_scores.to_csv('SleepDataWithScores.csv', index=False)