#!/usr/bin/env python3
"""
Complete Elephant Rumble Classifier Training & Testing
========================================================

Trains multiple models (Random Forest, SVM, CNN) and provides comprehensive evaluation.

Usage:
    python train_complete.py
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import joblib

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not installed. CNN training will be skipped.")
    print("   Install with: pip install torch")
    TORCH_AVAILABLE = False

# Audio processing
import librosa
import librosa.display

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

print("="*70)
print("🐘 ELEPHANT RUMBLE CLASSIFIER - COMPLETE TRAINING")
print("="*70)
print("\n✅ All libraries imported successfully!")
print(f"📦 NumPy: {np.__version__}")
print(f"📦 Pandas: {pd.__version__}")
if TORCH_AVAILABLE:
    print(f"📦 PyTorch: {torch.__version__}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 1: LOAD DATA")
print("="*70)

# Load cluster assignments from unsupervised learning
clusters_df = pd.read_csv('unsupervised_results/cluster_assignments.csv')

# Load features
features_raw = np.load('unsupervised_results/features_raw.npy')
features_normalized = np.load('unsupervised_results/features_normalized.npy')

# Check if PCA was applied
if Path('unsupervised_results/features_pca.npy').exists():
    features_pca = np.load('unsupervised_results/features_pca.npy')
    print(f"📊 PCA features loaded: {features_pca.shape}")
    features = features_pca
else:
    features = features_normalized

print(f"\n✅ Loaded data:")
print(f"   Files: {len(clusters_df)}")
print(f"   Features: {features.shape[1]} dimensions")
print(f"   Clusters: {clusters_df['cluster'].nunique()}")

# ============================================================================
# 2. EXPLORE CLUSTERS
# ============================================================================

print("\n" + "="*70)
print("STEP 2: CLUSTER ANALYSIS")
print("="*70)

# Cluster statistics
cluster_counts = clusters_df['cluster'].value_counts().sort_index()

print("\n📊 Cluster Distribution:")
print("="*50)
for cluster_id, count in cluster_counts.items():
    percentage = count / len(clusters_df) * 100
    print(f"Cluster {cluster_id}: {count:3d} rumbles ({percentage:5.1f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
cluster_counts.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[0].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Cluster ID', fontsize=12)
axes[0].set_ylabel('Number of Rumbles', fontsize=12)
axes[0].grid(True, alpha=0.3)

for i, (cluster_id, count) in enumerate(cluster_counts.items()):
    percentage = count / len(clusters_df) * 100
    axes[0].text(i, count + 1, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontweight='bold')

# Pie chart
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
axes[1].pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
           autopct='%1.1f%%', colors=colors[:len(cluster_counts)],
           startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Cluster Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()

# Create models directory if it doesn't exist
Path('models').mkdir(exist_ok=True)
plt.savefig('models/cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ You have {len(cluster_counts)} clusters")
if max(cluster_counts) / min(cluster_counts) < 2:
    print("   ✓ Balanced split!")
else:
    print(f"   ⚠️ Imbalanced: largest is {max(cluster_counts)/min(cluster_counts):.1f}x bigger")

# Show representative files
print("\n🎧 REPRESENTATIVE RUMBLES FROM EACH CLUSTER:")
print("-"*70)
for cluster_id in sorted(clusters_df['cluster'].unique()):
    cluster_files = clusters_df[clusters_df['cluster'] == cluster_id]['filename']
    print(f"\nCluster {cluster_id} ({len(cluster_files)} rumbles) - First 3:")
    for i, filename in enumerate(cluster_files.head(3), 1):
        print(f"  {i}. outputs/audio/{filename}")

# ============================================================================
# 3. CREATE LABELS
# ============================================================================

print("\n" + "="*70)
print("STEP 3: CREATE TRAINING LABELS")
print("="*70)

# Define cluster names (MODIFY THESE!)
cluster_names = {
    0: "Type A",  # Replace based on what you heard
    1: "Type B",  # Replace based on what you heard
    # Add more if you have more clusters:
    # 2: "Type C",
    # 3: "Type D",
}

# Update with actual number of clusters
for i in range(len(cluster_counts)):
    if i not in cluster_names:
        cluster_names[i] = f"Type {chr(65+i)}"  # A, B, C, D...

print("\n🏷️ CLUSTER LABELS:")
print("="*50)
for cluster_id, name in cluster_names.items():
    count = (clusters_df['cluster'] == cluster_id).sum()
    print(f"Cluster {cluster_id} = '{name}': {count} rumbles")

# Create labels.json
labels_dict = dict(zip(
    clusters_df['filename'],
    clusters_df['cluster']
))

# Save
Path('data').mkdir(exist_ok=True)
with open('data/labels.json', 'w') as f:
    json.dump(labels_dict, f, indent=2)

print("\n💾 Saved: data/labels.json")

# ============================================================================
# 4. PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 4: PREPARE TRAINING DATA")
print("="*70)

# Extract labels array
y = clusters_df['cluster'].values
X = features  # Already normalized/PCA'd

print(f"\n📊 Data Summary:")
print(f"   Total samples: {len(X)}")
print(f"   Feature dimensions: {X.shape[1]}")
print(f"   Number of classes: {len(np.unique(y))}")

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✅ Data split:")
print(f"   Training:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing:    {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 5. TRAIN RANDOM FOREST
# ============================================================================

print("\n" + "="*70)
print("STEP 5: TRAIN RANDOM FOREST")
print("="*70)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n🌲 Training Random Forest...")
rf_clf.fit(X_train, y_train)

y_train_pred_rf = rf_clf.predict(X_train)
y_test_pred_rf = rf_clf.predict(X_test)
y_test_proba_rf = rf_clf.predict_proba(X_test)

train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

print(f"\n✅ Random Forest Performance:")
print(f"   Train Accuracy: {train_acc_rf:.1%}")
print(f"   Test Accuracy:  {test_acc_rf:.1%}")

# ============================================================================
# 6. TRAIN SVM
# ============================================================================

print("\n" + "="*70)
print("STEP 6: TRAIN SVM")
print("="*70)

svm_clf = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True,
    random_state=42
)

print("\n🎯 Training SVM...")
svm_clf.fit(X_train, y_train)

y_train_pred_svm = svm_clf.predict(X_train)
y_test_pred_svm = svm_clf.predict(X_test)
y_test_proba_svm = svm_clf.predict_proba(X_test)

train_acc_svm = accuracy_score(y_train, y_train_pred_svm)
test_acc_svm = accuracy_score(y_test, y_test_pred_svm)

print(f"\n✅ SVM Performance:")
print(f"   Train Accuracy: {train_acc_svm:.1%}")
print(f"   Test Accuracy:  {test_acc_svm:.1%}")

# ============================================================================
# 7. TRAIN CNN (if PyTorch available)
# ============================================================================

if TORCH_AVAILABLE:
    print("\n" + "="*70)
    print("STEP 7: TRAIN CONVOLUTIONAL NEURAL NETWORK")
    print("="*70)
    
    # Define CNN Model
    class RumbleCNN(nn.Module):
        """1D CNN for rumble classification from features."""
        
        def __init__(self, input_dim, num_classes, dropout=0.5):
            super(RumbleCNN, self).__init__()
            
            # Reshape features to (batch, 1, features) for 1D conv
            self.conv1 = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.MaxPool1d(2)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.MaxPool1d(2)
            )
            
            # Calculate flattened size
            conv_output_size = input_dim // 4  # After 2 maxpool layers
            
            self.fc = nn.Sequential(
                nn.Linear(64 * conv_output_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            # x shape: (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            return x
    
    # Prepare data for PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    cnn_model = RumbleCNN(
        input_dim=X.shape[1],
        num_classes=len(cluster_names),
        dropout=0.5
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    print(f"\n🧠 Training CNN on {device}...")
    print(f"   Architecture: Conv1D → Conv1D → Dense → Output")
    print(f"   Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
    
    num_epochs = 100
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Train
        cnn_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = cnn_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        cnn_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = cnn_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(cnn_model.state_dict(), 'models/cnn_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    cnn_model.load_state_dict(torch.load('models/cnn_best.pth'))
    
    # Final evaluation
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(X_test_tensor)
        _, y_test_pred_cnn = outputs.max(1)
        y_test_proba_cnn = torch.softmax(outputs, dim=1).cpu().numpy()
        y_test_pred_cnn = y_test_pred_cnn.cpu().numpy()
    
    test_acc_cnn = accuracy_score(y_test, y_test_pred_cnn)
    train_outputs = cnn_model(X_train_tensor)
    _, y_train_pred_cnn = train_outputs.max(1)
    train_acc_cnn = accuracy_score(y_train, y_train_pred_cnn.cpu().numpy())
    
    print(f"\n✅ CNN Performance:")
    print(f"   Train Accuracy: {train_acc_cnn:.1%}")
    print(f"   Test Accuracy:  {test_acc_cnn:.1%}")
    print(f"   Best Val Accuracy: {best_val_acc:.1%}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('CNN Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('CNN Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('models/cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("\n" + "="*70)
    print("STEP 7: SKIPPED (PyTorch not available)")
    print("="*70)
    test_acc_cnn = 0
    train_acc_cnn = 0

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("STEP 8: MODEL COMPARISON")
print("="*70)

# Create comparison table
models_data = {
    'Model': ['Random Forest', 'SVM'],
    'Train Accuracy': [train_acc_rf, train_acc_svm],
    'Test Accuracy': [test_acc_rf, test_acc_svm],
    'Overfitting': [train_acc_rf - test_acc_rf, train_acc_svm - test_acc_svm]
}

if TORCH_AVAILABLE:
    models_data['Model'].append('CNN')
    models_data['Train Accuracy'].append(train_acc_cnn)
    models_data['Test Accuracy'].append(test_acc_cnn)
    models_data['Overfitting'].append(train_acc_cnn - test_acc_cnn)

comparison_df = pd.DataFrame(models_data)

print("\n📊 MODEL COMPARISON:")
print("="*60)
print(comparison_df.to_string(index=False))

# Determine best model
best_idx = comparison_df['Test Accuracy'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']
best_acc = comparison_df.loc[best_idx, 'Test Accuracy']

print(f"\n🏆 Best Model: {best_model_name} ({best_acc:.1%} test accuracy)")

# Select best model for analysis
if best_model_name == 'Random Forest':
    best_clf = rf_clf
    y_test_pred = y_test_pred_rf
    y_test_proba = y_test_proba_rf
elif best_model_name == 'SVM':
    best_clf = svm_clf
    y_test_pred = y_test_pred_svm
    y_test_proba = y_test_proba_svm
else:  # CNN
    best_clf = cnn_model
    y_test_pred = y_test_pred_cnn
    y_test_proba = y_test_proba_cnn

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison_df))
width = 0.35

bars1 = ax.bar(x - width/2, comparison_df['Train Accuracy'], width, 
               label='Train', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, comparison_df['Test Accuracy'], width,
               label='Test', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend(fontsize=11)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1%}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. DETAILED EVALUATION
# ============================================================================

print("\n" + "="*70)
print("STEP 9: DETAILED EVALUATION")
print("="*70)

# Classification report
print("\n📊 Classification Report:")
print("="*70)
report = classification_report(
    y_test, 
    y_test_pred,
    target_names=[cluster_names[i] for i in sorted(cluster_names.keys())],
    digits=3
)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    square=True,
    xticklabels=[cluster_names[i] for i in sorted(cluster_names.keys())],
    yticklabels=[cluster_names[i] for i in sorted(cluster_names.keys())],
    ax=ax,
    annot_kws={'size': 14, 'weight': 'bold'}
)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrix analysis
print("\n📊 Confusion Matrix Analysis:")
print("="*50)
for i, name in cluster_names.items():
    true_pos = cm[i, i]
    total = cm[i, :].sum()
    false_neg = total - true_pos
    false_pos = cm[:, i].sum() - true_pos
    
    print(f"\n{name}:")
    print(f"  Correctly classified: {true_pos}/{total} ({true_pos/total*100:.1f}%)")
    if false_neg > 0:
        print(f"  ⚠️ Missed (false negatives): {false_neg}")
    if false_pos > 0:
        print(f"  ⚠️ False alarms (false positives): {false_pos}")

# ROC Curve (binary classification only)
if len(cluster_names) == 2:
    print("\n📊 ROC Curve Analysis:")
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
    roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#3498db', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"   AUC Score: {roc_auc:.3f}")
    if roc_auc > 0.9:
        print("   ✅ Excellent discrimination!")
    elif roc_auc > 0.8:
        print("   ✅ Good discrimination")
    else:
        print("   ✓ Acceptable discrimination")

# Cross-validation (for non-neural models)
if best_model_name != 'CNN':
    print("\n🔄 Running 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"\n📊 CV Results:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.1%}")
    print(f"\n   Mean: {cv_scores.mean():.1%} (± {cv_scores.std():.1%})")

# Feature importance (Random Forest only)
if isinstance(best_clf, RandomForestClassifier):
    print("\n🔍 Feature Importance Analysis:")
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = min(15, len(importances))
    print(f"\n   Top {top_n} features:")
    for i in range(top_n):
        idx = indices[i]
        print(f"   {i+1:2d}. Feature {idx:3d}: {importances[idx]:.4f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices[:top_n]], color='#3498db', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f'Feature {indices[i]}' for i in range(top_n)])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# 10. SAVE MODELS AND PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("STEP 10: SAVE MODELS")
print("="*70)

Path('models').mkdir(exist_ok=True)

# Save best model
if best_model_name != 'CNN':
    joblib.dump(best_clf, 'models/best_classifier.pkl')
    print(f"\n💾 Saved: models/best_classifier.pkl")
else:
    # Already saved during training
    print(f"\n💾 Saved: models/cnn_best.pth")

# Save all models
joblib.dump(rf_clf, 'models/random_forest.pkl')
joblib.dump(svm_clf, 'models/svm.pkl')
print(f"💾 Saved: models/random_forest.pkl")
print(f"💾 Saved: models/svm.pkl")

# Save metadata
metadata = {
    'best_model': best_model_name,
    'best_test_accuracy': float(best_acc),
    'n_features': X.shape[1],
    'n_classes': len(cluster_names),
    'cluster_names': cluster_names,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'models': {
        'random_forest': {'train_acc': float(train_acc_rf), 'test_acc': float(test_acc_rf)},
        'svm': {'train_acc': float(train_acc_svm), 'test_acc': float(test_acc_svm)},
    }
}

if TORCH_AVAILABLE:
    metadata['models']['cnn'] = {'train_acc': float(train_acc_cnn), 'test_acc': float(test_acc_cnn)}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"💾 Saved: models/model_metadata.json")

# Save predictions
test_indices = np.arange(len(X))[len(X_train):]
test_filenames = clusters_df.iloc[test_indices]['filename'].values

predictions_df = pd.DataFrame({
    'filename': test_filenames,
    'true_cluster': y_test,
    'predicted_cluster': y_test_pred,
    'true_label': [cluster_names[i] for i in y_test],
    'predicted_label': [cluster_names[i] for i in y_test_pred],
    'correct': y_test == y_test_pred,
    'confidence': [y_test_proba[i, y_test_pred[i]] for i in range(len(y_test_pred))]
})

# Add probability columns
for i in sorted(cluster_names.keys()):
    predictions_df[f'prob_{cluster_names[i]}'] = y_test_proba[:, i]

predictions_df.to_csv('models/test_predictions.csv', index=False)
print(f"💾 Saved: models/test_predictions.csv")

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)

print(f"\n📊 Final Results:")
print(f"   Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_acc:.1%}")
print(f"   Total Samples: {len(X)}")
print(f"   Classes: {len(cluster_names)}")

print(f"\n💾 Saved Files:")
print(f"   ✅ models/best_classifier.pkl (or cnn_best.pth)")
print(f"   ✅ models/random_forest.pkl")
print(f"   ✅ models/svm.pkl")
print(f"   ✅ models/model_metadata.json")
print(f"   ✅ models/test_predictions.csv")
print(f"   ✅ data/labels.json")

print(f"\n📊 Visualizations:")
print(f"   ✅ models/cluster_distribution.png")
print(f"   ✅ models/model_comparison.png")
print(f"   ✅ models/confusion_matrix.png")
if TORCH_AVAILABLE:
    print(f"   ✅ models/cnn_training_history.png")
if len(cluster_names) == 2:
    print(f"   ✅ models/roc_curve.png")
if isinstance(best_clf, RandomForestClassifier):
    print(f"   ✅ models/feature_importance.png")

print(f"\n🚀 Usage:")
print(f"   ```python")
if best_model_name == 'CNN':
    print(f"   import torch")
    print(f"   model = RumbleCNN({X.shape[1]}, {len(cluster_names)})")
    print(f"   model.load_state_dict(torch.load('models/cnn_best.pth'))")
else:
    print(f"   import joblib")
    print(f"   clf = joblib.load('models/best_classifier.pkl')")
    print(f"   prediction = clf.predict(new_features)")
print(f"   ```")

print("\n" + "="*70)
print("🐘 Happy Classifying! 🎯")
print("="*70)
