import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            ConfusionMatrixDisplay, RocCurveDisplay)
from scipy import signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

import os
import mne
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt


# Sample EEG file loading (replace with your dataset)
def load_sample_eeg(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(0.5, 50., fir_design='firwin')  # Bandpass filter
    raw.notch_filter(50.)  # Notch filter (50Hz noise)
    picks = mne.pick_types(raw.info, eeg=True)  # Get EEG channels
    data, times = raw[picks, :]  # Extract EEG data
    return data, times

# Load a sample from TUH EEG Corpus
sample_file = "tuh_eeg_sample.edf"  # Replace with actual path
eeg_data, _ = load_sample_eeg(sample_file)
print(f"EEG Data Shape: {eeg_data.shape}")  # (channels, timepoints)


# =============================================
# 1. Data Loading and Preprocessing
# =============================================

class EEGDataPreprocessor:
    def __init__(self, sampling_rate=256, lowcut=0.5, highcut=50, notch_freq=50):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.scaler = StandardScaler()
        
    def bandpass_filter(self, data):
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def notch_filter(self, data):
        nyquist = 0.5 * self.sampling_rate
        freq = self.notch_freq / nyquist
        b, a = signal.iirnotch(freq, 30)
        return signal.filtfilt(b, a, data)
    
    def preprocess(self, eeg_data):
        # Apply filters
        filtered_data = self.bandpass_filter(eeg_data)
        filtered_data = self.notch_filter(filtered_data)
        
        # Normalize
        filtered_data = self.scaler.fit_transform(filtered_data)
        
        return filtered_data

# =============================================
# 2. Model Architecture Components
# =============================================

class LightweightCNN:
    
    def build(input_shape=(64, 64, 3)):
        model = models.Sequential([
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second conv block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third conv block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.15)
        ])
        
        return model

class NeuralODE(layers.Layer):
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
        # Dynamics function (MLP)
        self.dynamics = tf.keras.Sequential([
            layers.Dense(units, activation=self.activation),
            layers.Dense(units, activation=self.activation)
        ])
        
    def call(self, inputs, times):
        # Solve ODE using Runge-Kutta 4th order
        solution = tfp.math.ode.BDF().solve(
            self.dynamics,
            inputs,
            solution_times=times
        )
        return solution.states[-1]

class GNNLayer(layers.Layer):
    def __init__(self, units=64, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
        # Graph convolution weights
        self.weight = self.add_weight(
            shape=(units, units),
            initializer='he_normal',
            trainable=True
        )
        
    def call(self, inputs, adj_matrix):
        # inputs: (batch_size, num_nodes, input_dim)
        # adj_matrix: (batch_size, num_nodes, num_nodes)
        
        # Graph convolution operation
        output = tf.matmul(adj_matrix, inputs)  # Aggregate neighbor features
        output = tf.matmul(output, self.weight)  # Transform features
        output = self.activation(output)
        
        return output

# =============================================
# 3. Complete Model Architecture
# =============================================

class EpilepsyDetectionModel:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.lcnn = LightweightCNN()
        self.node = NeuralODE(units=64)
        self.gnn1 = GNNLayer(units=64)
        self.gnn2 = GNNLayer(units=64)
        
    def build_model(self, input_shapes):
        # Input layers
        eeg_input = layers.Input(shape=input_shapes['eeg'], name='eeg_input')
        adj_input = layers.Input(shape=input_shapes['adj'], name='adj_input')
        time_input = layers.Input(shape=input_shapes['time'], name='time_input')
        
        # 1. Spatial Feature Extraction (LCNN)
        spatial_features = self.lcnn.build()(eeg_input)
        
        # 2. Temporal Feature Extraction (NODE)
        # Reshape for NODE input
        temporal_input = layers.Reshape((-1, 64))(eeg_input)
        temporal_features = self.node(temporal_input, time_input)
        
        # 3. Functional Feature Extraction (GNN)
        functional_features = self.gnn1(adj_input)
        functional_features = self.gnn2(functional_features)
        functional_features = layers.GlobalAveragePooling1D()(functional_features)
        
        # Feature fusion with multi-head attention
        fused_features = layers.Concatenate()([spatial_features, 
                                             temporal_features, 
                                             functional_features])
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(fused_features, fused_features)
        attention = layers.LayerNormalization()(attention + fused_features)
        
        # Classification head
        output = layers.Dense(128, activation='relu')(attention)
        output = layers.Dropout(0.15)(output)
        output = layers.Dense(self.num_classes, activation='softmax')(output)
        
        # Create model
        model = tf.keras.Model(
            inputs=[eeg_input, adj_input, time_input],
            outputs=output
        )
        
        return model

# =============================================
# 4. Training and Evaluation
# =============================================

class EpilepsyDetectionPipeline:
    def __init__(self):
        self.preprocessor = EEGDataPreprocessor()
        self.model = EpilepsyDetectionModel()
        
    def prepare_data(self, eeg_data, labels, adj_matrices, times):
        # Preprocess EEG data
        processed_data = np.array([self.preprocessor.preprocess(sample) for sample in eeg_data])
        
        # Convert labels to categorical
        categorical_labels = to_categorical(labels)
        
        return processed_data, adj_matrices, times, categorical_labels
    
    def train(self, train_data, val_data, epochs=50, batch_size=32):
        # Unpack data
        (x_train, adj_train, time_train, y_train) = train_data
        (x_val, adj_val, time_val, y_val) = val_data
        
        # Build model
        input_shapes = {
            'eeg': x_train.shape[1:],
            'adj': adj_train.shape[1:],
            'time': time_train.shape[1:]
        }
        model = self.model.build_model(input_shapes)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        # Train model
        history = model.fit(
            [x_train, adj_train, time_train],
            y_train,
            validation_data=([x_val, adj_val, time_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return model, history
    
    def evaluate(self, model, test_data):
        # Unpack data
        x_test, adj_test, time_test, y_test = test_data
        
        # Predictions
        y_pred = model.predict([x_test, adj_test, time_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true_classes, y_pred_classes),
            'precision': precision_score(y_true_classes, y_pred_classes),
            'recall': recall_score(y_true_classes, y_pred_classes),
            'f1_score': f1_score(y_true_classes, y_pred_classes),
            'roc_auc': roc_auc_score(y_true_classes, y_pred[:, 1])
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        return metrics, cm
    
    def plot_metrics(self, history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred):
        RocCurveDisplay.from_predictions(y_true, y_pred)
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.show()

# =============================================
# 5. Main Execution
# =============================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = EpilepsyDetectionPipeline()
    
    # Example data preparation (replace with actual data loading)
    # For demonstration, we'll create synthetic data
    num_samples = 1000
    seq_length = 256
    num_channels = 23
    
    # Synthetic EEG data (replace with actual data loading)
    eeg_data = np.random.randn(num_samples, seq_length, num_channels)
    labels = np.random.randint(0, 2, size=num_samples)  # Binary classification
    adj_matrices = np.random.rand(num_samples, num_channels, num_channels)  # Random adjacency matrices
    times = np.linspace(0, 1, num=seq_length).reshape(1, -1).repeat(num_samples, axis=0)
    
    # Split data into train/val/test (70/15/15)
    split_idx1 = int(0.7 * num_samples)
    split_idx2 = int(0.85 * num_samples)
    
    train_data = (eeg_data[:split_idx1], adj_matrices[:split_idx1], 
                  times[:split_idx1], labels[:split_idx1])
    val_data = (eeg_data[split_idx1:split_idx2], adj_matrices[split_idx1:split_idx2],
                times[split_idx1:split_idx2], labels[split_idx1:split_idx2])
    test_data = (eeg_data[split_idx2:], adj_matrices[split_idx2:],
                 times[split_idx2:], labels[split_idx2:])
    
    # Preprocess data
    train_data = pipeline.prepare_data(*train_data)
    val_data = pipeline.prepare_data(*val_data)
    test_data = pipeline.prepare_data(*test_data)
    
    # Train model
    model, history = pipeline.train(train_data, val_data)
    
    # Evaluate model
    metrics, cm = pipeline.evaluate(model, test_data)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Plot results
    pipeline.plot_metrics(history)
    pipeline.plot_confusion_matrix(cm)
    pipeline.plot_roc_curve(np.argmax(test_data[3], axis=1), 
                           model.predict([test_data[0], test_data[1], test_data[2]])[:, 1])
