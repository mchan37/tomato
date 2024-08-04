from sklearn.preprocessing import StandardScaler
import numpy as np
from dataset import ImageAnnotationHandler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

if __name__ == '__main__':
    train_handler = ImageAnnotationHandler('tomato_data/train')
    valid_handler = ImageAnnotationHandler('tomato_data/valid')

    train_features, train_labels = train_handler.sift_features()
    valid_features, valid_labels = valid_handler.sift_features()
    
    # Prepare SIFT features for CNN
    max_keypoints = 100
    sift_features_for_cnn = prepare_sift_for_cnn(train_features, max_keypoints)

    # Standardize the features
    scaler = StandardScaler()
    sift_features_for_cnn = scaler.fit_transform(sift_features_for_cnn)

    # X_train is sift_features_for_cnn array
    # y_train is labels
    X_train = sift_features_for_cnn
    y_train = train_labels

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    # Instantiate the model, loss function, and optimizer
    input_shape = (max_keypoints, 128)  # Each image is represented by `max_keypoints` x 128 SIFT descriptors
    num_classes = train_handler.num_categories()  # 3 output units
    model = SimpleCNN(input_shape, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete")
