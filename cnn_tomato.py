from sklearn.preprocessing import StandardScaler
import numpy as np
from dataset import ImageAnnotationHandler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product  # For creating hyperparameter combinations
import os

# Directory where the model will be saved
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_cnn_model.pth')

def prepare_sift_for_cnn(sift_features, max_keypoints=100):
    fixed_size_features = []
    for descriptors in sift_features:
        if descriptors.shape[0] > max_keypoints:
            descriptors = descriptors[:max_keypoints, :]
        elif descriptors.shape[0] < max_keypoints:
            padding = np.zeros((max_keypoints - descriptors.shape[0], descriptors.shape[1]))
            descriptors = np.vstack((descriptors, padding))
        fixed_size_features.append(descriptors.flatten())
    
    return np.array(fixed_size_features)

def calculate_accuracy(predictions, labels):
    # Get the predicted class with the highest score
    predicted_classes = predictions.argmax(dim=1)
    # Calculate the number of correct predictions
    correct_predictions = (predicted_classes == labels).sum().item()
    # Calculate accuracy as the percentage of correct predictions
    accuracy = correct_predictions / labels.size(0)
    return accuracy

if __name__ == '__main__':
    # Load the data
    use_sift = False
    train_handler = ImageAnnotationHandler('tomato_data/train')
    valid_handler = ImageAnnotationHandler('tomato_data/valid')
    test_handler = ImageAnnotationHandler('tomato_data/valid')

    if use_sift:
        train_features, train_labels = train_handler.sift_features()
        valid_features, valid_labels = valid_handler.sift_features()
        test_features, test_labels = test_handler.sift_features()
        
        # Prepare SIFT features for CNN
        max_keypoints = 100
        sift_features_for_cnn_train = prepare_sift_for_cnn(train_features, max_keypoints)
        sift_features_for_cnn_valid = prepare_sift_for_cnn(valid_features, max_keypoints)
        sift_features_for_cnn_test = prepare_sift_for_cnn(test_features, max_keypoints)

        # Standardize the features
        scaler = StandardScaler()
        sift_features_for_cnn_train = scaler.fit_transform(sift_features_for_cnn_train)
        sift_features_for_cnn_valid = scaler.transform(sift_features_for_cnn_valid)
        sift_features_for_cnn_test = scaler.transform(sift_features_for_cnn_test)

        X_train = torch.tensor(sift_features_for_cnn_train, dtype=torch.float32)
        X_valid = torch.tensor(sift_features_for_cnn_valid, dtype=torch.float32)
        X_test = torch.tensor(sift_features_for_cnn_test, dtype=torch.float32)

    else:
        # Get pixel features instead of SIFT features
        target_size = (224, 224)  # Image size
        train_features, train_labels = train_handler.pixel_features(target_size=target_size)
        valid_features, valid_labels = valid_handler.pixel_features(target_size=target_size)
        test_features, test_labels = test_handler.pixel_features(target_size=target_size)

        X_train = torch.tensor(train_features, dtype=torch.float32).permute(0, 3, 1, 2)  # Change shape to [N, C, H, W]
        X_valid = torch.tensor(valid_features, dtype=torch.float32).permute(0, 3, 1, 2)
        X_test = torch.tensor(test_features, dtype=torch.float32).permute(0, 3, 1, 2)

    # Convert datasets to PyTorch tensors
    y_train = torch.tensor(train_labels, dtype=torch.long)
    
    y_valid = torch.tensor(valid_labels, dtype=torch.long)

    y_test = torch.tensor(test_labels, dtype=torch.long)

    # Create TensorDataset and DataLoader for datasets
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    if use_sift:
            # Define the CNN model
        class SimpleCNN(nn.Module):
            def __init__(self, input_shape, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(128 * input_shape[0], 256)
                self.fc2 = nn.Linear(256, num_classes)

            def forward(self, x):
                x = x.view(x.size(0), 128, -1)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        input_shape = (max_keypoints, 128)  # Each image is represented by `max_keypoints` x 128 SIFT descriptors
    else:
        # Define the CNN model for RGB images
        class SimpleCNN(nn.Module):
            def __init__(self, input_shape, num_classes):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 256)  # Adjust according to pooling
                self.fc2 = nn.Linear(256, num_classes)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(x, kernel_size=2, stride=2)
                x = torch.relu(self.conv2(x))
                x = torch.max_pool2d(x, kernel_size=2, stride=2)
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        input_shape = (3, target_size[0], target_size[1])  # Input shape for RGB images

    # Hyperparameter tuning setup
    hyperparameters = {
        # 'batch_size': [16, 32, 64],
        'batch_size': [ 32],
        # 'learning_rate': [0.001, 0.0005],
        'learning_rate': [0.001],
        # 'num_epochs': [10, 20],
        'num_epochs': [10],
    }

    num_classes = train_handler.num_categories()  # Number of classes

    # Grid search over hyperparameters
    best_valid_accuracy = 0
    best_params = None
    best_model_state = None

    for batch_size, learning_rate, num_epochs in product(hyperparameters['batch_size'],
                                                         hyperparameters['learning_rate'],
                                                         hyperparameters['num_epochs']):
        print(f'Training with batch size: {batch_size}, learning rate: {learning_rate}, epochs: {num_epochs}')

        # Create DataLoader for the current batch size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate the model, loss function, and optimizer
        model = SimpleCNN(input_shape, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            for batch_X, batch_y in train_dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                train_loss += loss.item()
                train_accuracy += calculate_accuracy(outputs, batch_y)

            # Average training loss and accuracy
            train_loss /= len(train_dataloader)
            train_accuracy /= len(train_dataloader)

            # Validation phase
            model.eval()
            valid_loss = 0.0
            valid_accuracy = 0.0
            with torch.no_grad():
                for batch_X, batch_y in valid_dataloader:
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Accumulate loss and accuracy
                    valid_loss += loss.item()
                    valid_accuracy += calculate_accuracy(outputs, batch_y)

            # Average validation loss and accuracy
            valid_loss /= len(valid_dataloader)
            valid_accuracy /= len(valid_dataloader)

            # Print epoch results
            print(f'Epoch [{epoch+1}/{num_epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')

        # Check if current model is the best
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_params = (batch_size, learning_rate, num_epochs)
            best_model_state = model.state_dict()

    print(f'Best Validation Accuracy: {best_valid_accuracy:.4f} with params: {best_params}')

    # Save the best model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(best_model_state, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

    # Evaluate on the test set with the best model
    best_model = SimpleCNN(input_shape, num_classes)
    best_model.load_state_dict(torch.load(MODEL_PATH))
    best_model.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=best_params[0], shuffle=False)
    test_accuracy = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            outputs = best_model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs, batch_y)

    test_loss /= len(test_dataloader)
    test_accuracy /= len(test_dataloader)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
