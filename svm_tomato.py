import numpy as np
from dataset import ImageAnnotationHandler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Load the data
    train_handler = ImageAnnotationHandler('tomato_data/train')
    valid_handler = ImageAnnotationHandler('tomato_data/valid')
    test_handler = ImageAnnotationHandler('tomato_data/valid')

    # Get pixel features
    target_size = (64, 64)  # Image size
    train_features, train_labels = train_handler.pixel_features(target_size=target_size, flatten=True)
    valid_features, valid_labels = valid_handler.pixel_features(target_size=target_size, flatten=True)
    test_features, test_labels = test_handler.pixel_features(target_size=target_size, flatten=True)

    # Combine train and validation data for hyperparameter tuning
    X_train_val = np.vstack((train_features, valid_features))
    y_train_val = np.concatenate((train_labels, valid_labels))

    # Split the combined data into train and validation sets for grid search
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    test_features = scaler.transform(test_features)

    # Hyperparameter tuning setup
    param_grid = {
        'C': [0.1],
        'gamma': [1],
        'kernel': ['rbf'],
        # 'C': [0.1, 1, 10, 100],
        # 'gamma': [1, 0.1, 0.01, 0.001],
        # 'kernel': ['rbf', 'linear'],
    }

    # Initialize the SVM classifier
    svm_model = SVC()

    # Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_params = grid_search.best_params_
    print(f'Best hyperparameters: {best_params}')

    # Evaluate on validation data
    best_svm_model = grid_search.best_estimator_
    y_valid_pred = best_svm_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f'Validation Accuracy: {valid_accuracy:.4f}')

    # Evaluate on test data
    y_test_pred = best_svm_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
