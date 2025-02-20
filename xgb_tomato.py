import numpy as np
import xgboost as xgb
from dataset import ImageAnnotationHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os

# Directory where the model will be saved
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_xgb_model.json')

if __name__ == '__main__':
    # Load the data
    train_handler = ImageAnnotationHandler('tomato_data/train')
    #valid_handler = ImageAnnotationHandler('tomato_data/valid')
    test_handler = ImageAnnotationHandler('tomato_data/valid')

    # Get pixel features instead of SIFT features
    target_size = (64, 64)  # Image size
    train_features, train_labels = train_handler.pixel_features(target_size=target_size, flatten=True)
    #valid_features, valid_labels = valid_handler.pixel_features(target_size=target_size, flatten=True)
    test_features, test_labels = test_handler.pixel_features(target_size=target_size, flatten=True)

    X_train_val = train_features
    y_train_val = train_labels

    # Split the data into train and validation sets for grid search
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Hyperparameter tuning setup
    param_grid = {
        # 'n_estimators': [50, 100, 200],
        # 'max_depth': [3, 6, 9],
        # 'learning_rate': [0.1, 0.01, 0.001],
        # 'subsample': [0.8, 1.0],
        # 'colsample_bytree': [0.8, 1.0],
        'n_estimators': [50],
        'max_depth': [3],
        'learning_rate': [0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }

    # Initialize the XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=train_handler.num_categories())

    # Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_params = grid_search.best_params_
    print(f'Best hyperparameters: {best_params}')

    # Evaluate on validation data
    best_xgb_model = grid_search.best_estimator_
    y_valid_pred = best_xgb_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f'Validation Accuracy: {valid_accuracy:.4f}')

    # Save the best model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    best_xgb_model.save_model(MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

    # Load the model for evaluation
    loaded_xgb_model = xgb.XGBClassifier()
    loaded_xgb_model.load_model(MODEL_PATH)

    # Evaluate on test data
    y_test_pred = loaded_xgb_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')