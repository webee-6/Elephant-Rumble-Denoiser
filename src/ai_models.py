"""
Deep Learning Models for Elephant Rumble Analysis

Includes:
- CNN for rumble classification
- RNN/LSTM for temporal modeling
- Autoencoder for unsupervised learning
- Siamese network for individual identification
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================================
# PyTorch Models
# ============================================================================

def create_rumble_cnn_pytorch(input_shape: Tuple[int, int],
                             num_classes: int = 10) -> 'torch.nn.Module':
    """
    Create CNN for rumble classification (PyTorch).
    
    Architecture:
    - Conv2D layers for feature extraction
    - BatchNorm + ReLU + MaxPool
    - Global average pooling
    - Dense classification head
    
    Args:
        input_shape: (freq_bins, time_frames) e.g., (128, 200)
        num_classes: Number of rumble classes
    
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        
        class RumbleCNN(nn.Module):
            def __init__(self, input_shape, num_classes):
                super(RumbleCNN, self).__init__()
                
                freq_bins, time_frames = input_shape
                
                # Convolutional blocks
                self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                # Global average pooling
                self.gap = nn.AdaptiveAvgPool2d(1)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                # x shape: (batch, 1, freq, time)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.gap(x)
                x = self.classifier(x)
                return x
        
        model = RumbleCNN(input_shape, num_classes)
        print(f"✅ Created PyTorch CNN")
        print(f"   Input: (batch, 1, {input_shape[0]}, {input_shape[1]})")
        print(f"   Output: (batch, {num_classes})")
        
        return model
    
    except ImportError:
        print("⚠️ PyTorch not installed. Install with: pip install torch")
        return None


def create_rumble_lstm_pytorch(input_dim: int = 20,
                               hidden_dim: int = 128,
                               num_layers: int = 2,
                               num_classes: int = 10) -> 'torch.nn.Module':
    """
    Create LSTM for temporal rumble modeling (PyTorch).
    
    Good for:
    - Call type classification
    - Individual identification from call patterns
    - Sequence-to-sequence tasks
    
    Args:
        input_dim: Feature dimension (e.g., 20 MFCCs)
        hidden_dim: LSTM hidden size
        num_layers: Number of LSTM layers
        num_classes: Output classes
    
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        
        class RumbleLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
                super(RumbleLSTM, self).__init__()
                
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.3 if num_layers > 1 else 0,
                    bidirectional=True
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                # x shape: (batch, time, features)
                lstm_out, (h_n, c_n) = self.lstm(x)
                
                # Use last hidden state from both directions
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
                
                output = self.classifier(h_last)
                return output
        
        model = RumbleLSTM(input_dim, hidden_dim, num_layers, num_classes)
        print(f"✅ Created PyTorch LSTM")
        return model
    
    except ImportError:
        print("⚠️ PyTorch not installed")
        return None


# ============================================================================
# TensorFlow/Keras Models
# ============================================================================

def create_rumble_cnn_keras(input_shape: Tuple[int, int],
                           num_classes: int = 10) -> 'tf.keras.Model':
    """
    Create CNN for rumble classification (Keras).
    
    Args:
        input_shape: (freq_bins, time_frames, channels)
        num_classes: Number of classes
    
    Returns:
        Keras model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Add channel dimension
        full_shape = (*input_shape, 1)
        
        model = keras.Sequential([
            # Input
            layers.Input(shape=full_shape),
            
            # Conv Block 1
            layers.Conv2D(32, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(2),
            
            # Conv Block 2
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(2),
            
            # Conv Block 3
            layers.Conv2D(128, 3, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(2),
            
            # Global pooling
            layers.GlobalAveragePooling2D(),
            
            # Classification head
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Created Keras CNN")
        model.summary()
        
        return model
    
    except ImportError:
        print("⚠️ TensorFlow not installed. Install with: pip install tensorflow")
        return None


def create_autoencoder_keras(input_shape: Tuple[int, int],
                            latent_dim: int = 64) -> 'tf.keras.Model':
    """
    Create autoencoder for unsupervised rumble learning.
    
    Use cases:
    - Feature learning from unlabeled data
    - Anomaly detection (unusual calls)
    - Dimensionality reduction
    - Data compression
    
    Args:
        input_shape: (freq_bins, time_frames)
        latent_dim: Bottleneck dimension
    
    Returns:
        Keras autoencoder model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        
        full_shape = (*input_shape, 1)
        
        # Encoder
        encoder_input = layers.Input(shape=full_shape)
        x = layers.Conv2D(32, 3, strides=2, padding='same')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Flatten()(x)
        latent = layers.Dense(latent_dim, name='latent')(x)
        
        encoder = keras.Model(encoder_input, latent, name='encoder')
        
        # Decoder
        latent_input = layers.Input(shape=(latent_dim,))
        
        # Calculate decoder start shape
        freq_bins, time_frames = input_shape
        dec_freq = freq_bins // 8
        dec_time = time_frames // 8
        
        x = layers.Dense(dec_freq * dec_time * 128)(latent_input)
        x = layers.Reshape((dec_freq, dec_time, 128))(x)
        
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        decoder_output = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
        
        decoder = keras.Model(latent_input, decoder_output, name='decoder')
        
        # Full autoencoder
        autoencoder_output = decoder(encoder(encoder_input))
        autoencoder = keras.Model(encoder_input, autoencoder_output, name='autoencoder')
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        print(f"✅ Created Keras Autoencoder")
        print(f"   Latent dim: {latent_dim}")
        
        return autoencoder
    
    except ImportError:
        print("⚠️ TensorFlow not installed")
        return None


# ============================================================================
# Classical ML Models (scikit-learn)
# ============================================================================

def train_random_forest(features: np.ndarray,
                       labels: np.ndarray,
                       n_estimators: int = 100) -> 'sklearn.ensemble.RandomForestClassifier':
    """
    Train Random Forest classifier on extracted features.
    
    Good for:
    - Baseline model
    - Feature importance analysis
    - Small datasets
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Class labels
        n_estimators: Number of trees
    
    Returns:
        Trained RandomForest model
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        print("\n" + classification_report(y_test, y_pred))
        
        print(f"\n✅ Random Forest trained")
        print(f"   Train accuracy: {clf.score(X_train, y_train):.3f}")
        print(f"   Test accuracy: {clf.score(X_test, y_test):.3f}")
        
        return clf
    
    except ImportError:
        print("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
        return None


if __name__ == "__main__":
    # Example: Create models
    print("Creating example models...\n")
    
    # CNN for spectrograms
    cnn = create_rumble_cnn_pytorch(input_shape=(128, 200), num_classes=5)
    
    # LSTM for MFCCs
    lstm = create_rumble_lstm_pytorch(input_dim=20, num_classes=5)
    
    # Keras CNN
    keras_cnn = create_rumble_cnn_keras(input_shape=(128, 200), num_classes=5)
    
    # Autoencoder
    autoencoder = create_autoencoder_keras(input_shape=(128, 200), latent_dim=64)
