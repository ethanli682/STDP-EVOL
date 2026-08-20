import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class FeatureFitnessDataset(Dataset):
    """Custom dataset for feature-fitness pairs"""
    def __init__(self, features, fitness_scores):
        self.features = torch.FloatTensor(features)
        self.fitness_scores = torch.FloatTensor(fitness_scores).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.fitness_scores[idx]

class FeatureFitnessNet(nn.Module):
    """Neural network for predicting fitness from features"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(FeatureFitnessNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (regression)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FeatureFitnessPredictor:
    """Main class for training and using the model"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2, device='cpu'):
        self.device = device
        self.model = FeatureFitnessNet(input_size, hidden_sizes, dropout_rate).to(self.device)
        self.fitness_scaler = StandardScaler()
        self.is_trained = False
        self.normalize_targets = True  # Default to True, updated in train_model
        
    def train_model(self, features, fitness_scores, epochs=100, batch_size=32, 
                   learning_rate=0.001, validation_split=0.2, normalize_targets=True):
        """
        Train the neural network.
        
        Args:
            normalize_targets (bool): If True, applies StandardScaler to fitness scores.
                                      If False, assumes fitness_scores are already scaled externally.
        """
        self.normalize_targets = normalize_targets
        
        # 1. Handle Normalization
        if self.normalize_targets:
            # Normalize fitness scores (features assumed to be in [-1,1])
            fitness_processed = self.fitness_scaler.fit_transform(fitness_scores.reshape(-1, 1)).flatten()
        else:
            # Use raw scores (external normalization assumed)
            fitness_processed = fitness_scores
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, fitness_processed, test_size=validation_split, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = FeatureFitnessDataset(X_train, y_train)
        val_dataset = FeatureFitnessDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        self.model.train()
        for epoch in range(epochs):
            epoch_train_loss = 0
            for batch_features, batch_fitness in train_loader:
                batch_features = batch_features.to(self.device)
                batch_fitness = batch_fitness.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_fitness)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_fitness in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_fitness = batch_fitness.to(self.device)
                    predictions = self.model(batch_features)
                    loss = criterion(predictions, batch_fitness)
                    epoch_val_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            self.model.train()
        
        self.is_trained = True
        return train_losses, val_losses
    
    def predict_fitness(self, features):
        """Predict fitness scores for given features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            if len(features_tensor.shape) == 1:
                features_tensor = features_tensor.unsqueeze(0)
            
            raw_predictions = self.model(features_tensor).cpu().numpy()
            
            # Conditionally denormalize
            if self.normalize_targets:
                predictions = self.fitness_scaler.inverse_transform(raw_predictions)
            else:
                predictions = raw_predictions
            
        return predictions.flatten()
    
    def suggest_features_for_fitness(self, target_fitness, num_features, 
                                   num_suggestions=10, max_iterations=1000, lr=0.01):
        """Use gradient descent to find features that produce target fitness"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making suggestions")
        
        # Conditionally normalize the target
        if self.normalize_targets:
            target_val = self.fitness_scaler.transform([[target_fitness]])[0, 0]
        else:
            target_val = target_fitness
            
        target_tensor = torch.FloatTensor([target_val]).to(self.device)
        
        best_features = []
        best_losses = []
        
        for _ in range(num_suggestions):
            # Initialize random features in [-1, 1] range
            features = torch.randn(1, num_features, device=self.device, requires_grad=True)
            features.data = torch.clamp(features.data, -1, 1)
            
            optimizer = optim.Adam([features], lr=lr)
            
            best_loss = float('inf')
            best_feature_set = None
            
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                # Predict fitness
                predicted_fitness = self.model(features)
                loss = nn.MSELoss()(predicted_fitness, target_tensor)
                
                loss.backward()
                optimizer.step()
                
                # Clamp features to [-1, 1] range
                with torch.no_grad():
                    features.data = torch.clamp(features.data, -1, 1)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_feature_set = features.data.clone()
                
                if loss.item() < 1e-6:
                    break
            
            best_features.append(best_feature_set.cpu().numpy().flatten())
            best_losses.append(best_loss)
        
        sorted_indices = np.argsort(best_losses)
        suggested_features = [best_features[i] for i in sorted_indices]
        
        return suggested_features
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fitness_scaler': self.fitness_scaler,
            'is_trained': self.is_trained,
            'normalize_targets': self.normalize_targets  # Save the flag
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fitness_scaler = checkpoint['fitness_scaler']
        self.is_trained = checkpoint['is_trained']
        # Load flag, default to True for backward compatibility with old saves
        self.normalize_targets = checkpoint.get('normalize_targets', True)

        
# Example usage and demo
def generate_demo_data(n_samples=1000, n_features=10):
    """Generate synthetic data for demonstration"""
    np.random.seed(42)
    
    # Generate features in [-1, 1] range
    features = np.random.uniform(-1, 1, (n_samples, n_features))
    
    # Create a complex fitness function (you can modify this)
    fitness_scores = (
        np.sum(features**2, axis=1) +  # Quadratic term
        0.5 * np.sum(features[:, :3], axis=1) +  # Linear term for first 3 features
        0.3 * np.prod(features[:, 3:6], axis=1) +  # Interaction term
        0.1 * np.random.normal(0, 1, n_samples)  # Noise
    )
    
    return features, fitness_scores

def plot_training_history(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

# Demo usage
if __name__ == "__main__":
    # Generate demo data
    print("Generating demo data...")
    features, fitness_scores = generate_demo_data(n_samples=2000, n_features=8)
    
    # Create and train model
    print("Creating and training model...")
    predictor = FeatureFitnessPredictor(input_size=8, hidden_sizes=[128, 64, 32])
    train_losses, val_losses = predictor.train_model(
        features, fitness_scores, epochs=100, batch_size=32
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Test predictions
    print("\nTesting predictions...")
    test_features = np.random.uniform(-1, 1, (5, 8))
    predicted_fitness = predictor.predict_fitness(test_features)
    print(f"Predicted fitness scores: {predicted_fitness}")
    
    # Suggest features for target fitness
    print("\nSuggesting features for target fitness...")
    target_fitness = 2.0
    suggested_features = predictor.suggest_features_for_fitness(
        target_fitness, num_features=8, num_suggestions=3
    )
    
    print(f"Target fitness: {target_fitness}")
    for i, features in enumerate(suggested_features):
        predicted = predictor.predict_fitness([features])[0]
        print(f"Suggestion {i+1}: Features = {features.round(3)}, Predicted fitness = {predicted:.3f}")
    
    # Save model
    predictor.save_model("feature_fitness_model.pth")
    print("\nModel saved successfully!")