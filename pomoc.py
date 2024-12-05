# import os
# os.system('pip install tensorflow_decision_forests --upgrade')

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import os

from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def convert_to_dataframe(X, y):
    # Flatten the time series data if necessary
    if len(X.shape) > 2:
        num_samples = X.shape[0]
        num_timesteps = X.shape[1]
        num_features = X.shape[2]
        X = X.reshape((num_samples, num_timesteps * num_features))
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y, columns=['Target'])
    df = pd.concat([df_X, df_y], axis=1)
    return df

# Example for training data
train_df = convert_to_dataframe(self.x_train[i], self.y_train[i])

# Similarly for validation and test data if needed



# Define the model
rf_model = tfdf.keras.RandomForestModel(
    num_trees=100,
    max_depth=17,
    max_features=8,
    min_examples=2,  # Corresponds to min_samples_split in sklearn
    # Other hyperparameters can be set here
)

# Compile the model (no need to specify loss for regression)
rf_model.compile(metrics=["mae"])

# Fit the model
rf_model.fit(train_df, label="target")

# Evaluate on training data
train_metrics = rf_model.evaluate(train_df, return_dict=True)

# If you have a validation set
val_df = convert_to_dataframe(self.x_train[i][-validation_window_size:], self.y_train[i][-validation_window_size:])
val_metrics = rf_model.evaluate(val_df, return_dict=True)

print(f"Training MAE: {train_metrics['mae']}")
print(f"Validation MAE: {val_metrics['mae']}")

# Instead of tuner.search(), use the Random Forest model

# Prepare data
train_df = convert_to_dataframe(self.x_train[i][:-validation_window_size], self.y_train[i][:-validation_window_size])
val_df = convert_to_dataframe(self.x_train[i][-validation_window_size:], self.y_train[i][-validation_window_size:])

# Define the model
rf_model = tfdf.keras.RandomForestModel(
    num_trees=hp.Int("num_trees", min_value=50, max_value=200, step=50),
    max_depth=hp.Int("max_depth", min_value=5, max_value=20, step=5),
    max_features=hp.Choice("max_features", values=[None, "auto", "sqrt", "log2", 8]),
    min_examples=hp.Int("min_examples", min_value=2, max_value=10, step=2),
)

# Compile the model
rf_model.compile(metrics=["mae"])

# Fit the model
rf_model.fit(train_df, label="target")

# Evaluate the model
val_metrics = rf_model.evaluate(val_df, return_dict=True)
print(f"Validation MAE: {val_metrics['mae']}")


# Define the tuner builder function
def rf_model_builder(hp):
    rf_model = tfdf.keras.RandomForestModel(
        num_trees=hp.Int("num_trees", min_value=50, max_value=200, step=50),
        max_depth=hp.Int("max_depth", min_value=5, max_value=20, step=5),
        max_features=hp.Choice("max_features", values=[None, "auto", "sqrt", "log2", 8]),
        min_examples=hp.Int("min_examples", min_value=2, max_value=10, step=2),
        # Add other hyperparameters as needed
    )
    rf_model.compile(metrics=["mae"])
    return rf_model

from keras_tuner.tuners import BayesianOptimization

# Create the tuner
tuner = BayesianOptimization(
    rf_model_builder,
    objective="val_mae",
    max_trials=10,
    directory="rf_models",
    project_name=f"rf_model_window_{i}",
)

tuner.search(
    train_df,
    validation_data=val_df,
    label="target",
)

# Retrieve the best model
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

def convert_to_dataframe(X, y):
    # Assuming X has shape (num_samples, num_timesteps, num_features)
    num_samples = X.shape[0]
    num_timesteps = X.shape[1]
    num_features = X.shape[2]
    X_flat = X.reshape((num_samples, num_timesteps * num_features))
    feature_names = [f'feature_{i}' for i in range(X_flat.shape[1])]
    df_X = pd.DataFrame(X_flat, columns=feature_names)
    df_y = pd.DataFrame(y, columns=['target'])
    df = pd.concat([df_X, df_y], axis=1)
    return df

# Save the model
rf_model.save(f'{report_dir}best_rf_model_window_{i}')

# Load the model
loaded_model = tf.keras.models.load_model(f'{report_dir}best_rf_model_window_{i}')

# Evaluate the previous LSTM model on the validation set
prev_lstm_model = tf.keras.models.load_model(f'{report_dir}best_model_window_{i-1}.h5')
prev_val_metrics = prev_lstm_model.evaluate(
    self.x_train[i][-validation_window_size:], 
    self.y_train[i][-validation_window_size:], 
    batch_size=int(self.config["model"]["BatchSizeValidation"]), 
    return_dict=True
)

# Compare validation MAE
if val_metrics['mae'] < prev_val_metrics['mae']:
    print("Random Forest model outperforms the previous LSTM model.")
    # Proceed with the Random Forest model
else:
    print("Previous LSTM model performs better.")
    # Consider using the LSTM model or further tuning


for i in range(num_windows):
    # Prepare data
    train_df = convert_to_dataframe(self.x_train[i][:-validation_window_size], self.y_train[i][:-validation_window_size])
    val_df = convert_to_dataframe(self.x_train[i][-validation_window_size:], self.y_train[i][-validation_window_size:])
    
    # Define and train the Random Forest model
    # (Include hyperparameter tuning if desired)
    
    # Save the best model
    rf_model.save(f'{report_dir}best_rf_model_window_{i}')
    
    # Optionally, compare with previous models and decide which to use

import shap

# Prepare data for SHAP
X_train_flat = self.x_train[i][:-validation_window_size].reshape((num_samples, -1))
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_flat)

# Plot SHAP values
shap.summary_plot(shap_values, X_train_flat)