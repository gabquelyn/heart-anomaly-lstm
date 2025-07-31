"""
Train model with the current real medical data (176K sequences)
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_current_real_data():
    """Load the current real medical data"""
    print("🏥 Loading current real medical data...")
    
    sequences_path = Path('data/processed/real_medical_sequences.npy')
    labels_path = Path('data/processed/real_medical_labels.npy')
    
    if not sequences_path.exists() or not labels_path.exists():
        print("❌ Real medical data not found!")
        raise FileNotFoundError("Real medical data files not found")
    
    sequences = np.load(sequences_path)
    labels = np.load(labels_path)
    
    print(f"✅ Loaded current real medical data:")
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Total sequences: {len(sequences):,}")
    print(f"   Normal cases: {np.sum(labels == 0):,} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    print(f"   Abnormal cases: {np.sum(labels == 1):,} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    
    return sequences, labels

def build_optimized_medical_model(input_shape):
    """Build optimized model for the current dataset size"""
    print("🧠 Building optimized medical model for 176K sequences...")
    
    inputs = Input(shape=input_shape, name='medical_input')
    
    # Enhanced CNN layers for large dataset
    conv1 = Conv1D(128, 7, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(2)(conv1)
    
    conv2 = Conv1D(256, 5, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling1D(2)(conv2)
    
    conv3 = Conv1D(128, 3, activation='relu', padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    
    # Enhanced LSTM layers
    lstm1 = LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(conv3)
    lstm1 = BatchNormalization()(lstm1)
    
    lstm2 = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    lstm3 = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1)(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    
    # Enhanced dense layers
    dense1 = Dense(128, activation='relu')(lstm3)
    dropout1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    
    dense3 = Dense(32, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(dense3)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', name='medical_output')(dropout3)
    
    model = Model(inputs=inputs, outputs=outputs, name='optimized_medical_lstm')
    
    # Compile with optimized settings for large dataset
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Slightly lower LR for stability
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_with_current_data():
    """Train model with current 176K real medical sequences"""
    print("🚀 Starting training with current real medical data...")
    print("📊 Dataset: 176,585 real patient sequences")
    
    # Load current data
    X, y = load_current_real_data()
    
    # Enhanced train/validation split for large dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y  # 15% test for large dataset
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train  # 15% validation
    )
    
    print(f"\n📈 Enhanced Data Split:")
    print(f"Training: {len(X_train):,} sequences ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val):,} sequences ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Testing: {len(X_test):,} sequences ({len(X_test)/len(X)*100:.1f}%)")
    
    # Normalize features
    print("\n⚙️  Normalizing medical features...")
    scaler = StandardScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit and transform
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    print("✅ Feature normalization complete")
    
    # Build optimized model
    model = build_optimized_medical_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    print(f"\n🏗️  Optimized Model Architecture:")
    model.summary()
    
    # Enhanced callbacks for large dataset
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # More patience for large dataset
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'models/best_medical_model_176k.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model with optimized settings
    print("\n🎯 Training optimized medical model on 176K sequences...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,  # Fewer epochs needed with large dataset
        batch_size=64,  # Larger batch size for efficiency
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Comprehensive evaluation
    print("\n🔬 Performing comprehensive medical evaluation...")
    
    # Test evaluation
    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
    test_accuracy = test_results[1]
    
    # Detailed predictions
    y_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate comprehensive metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    sensitivity = recall
    specificity = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Display results
    print(f"\n🏥 MEDICAL MODEL PERFORMANCE (176K Dataset)")
    print(f"{'='*70}")
    print(f"Test Accuracy:     {test_accuracy:.4f}")
    print(f"Sensitivity:       {sensitivity:.4f} (Ability to detect anomalies)")
    print(f"Specificity:       {specificity:.4f} (Ability to identify normal)")
    print(f"Precision:         {precision:.4f} (Positive predictive value)")
    print(f"F1 Score:          {f1:.4f} (Balanced performance)")
    print(f"AUC Score:         {auc_score:.4f} (Discrimination ability)")
    print(f"{'='*70}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🩺 CLINICAL INTERPRETATION (176K Dataset)")
    print(f"{'='*70}")
    print(f"True Positives:  {tp:4d} (Correctly identified anomalies)")
    print(f"True Negatives:  {tn:4d} (Correctly identified normal)")
    print(f"False Positives: {fp:4d} (False alarms)")
    print(f"False Negatives: {fn:4d} (Missed anomalies - CRITICAL)")
    print(f"{'='*70}")
    
    # Save model and comprehensive metrics
    save_model_and_metrics_176k(
        model, scaler, history, test_accuracy, precision, recall, 
        sensitivity, specificity, f1, auc_score, cm, len(X)
    )
    
    print(f"\n🎉 Training completed successfully with 176K real medical sequences!")
    return model, scaler, history

def save_model_and_metrics_176k(model, scaler, history, accuracy, precision, recall, 
                               sensitivity, specificity, f1, auc, cm, total_sequences):
    """Save model and metrics for 176K dataset"""
    print("💾 Saving optimized medical model and metrics...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save('models/heart_anomaly_medical_lstm_176k.keras')
    joblib.dump(scaler, 'models/scaler_medical_176k.pkl')
    
    # Save scaler parameters for Node.js
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist()
    }
    
    with open('models/scaler_params_medical_176k.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    # Comprehensive metrics
    tn, fp, fn, tp = cm.ravel()
    
    medical_metrics = {
        'model_info': {
            'name': 'Medical Heart Anomaly LSTM - 176K Dataset',
            'version': '2.0',
            'trained_on': 'Real Medical Data (MIT-BIH + AFDB)',
            'total_sequences': total_sequences,
            'training_date': pd.Timestamp.now().isoformat(),
            'total_parameters': model.count_params()
        },
        'dataset_info': {
            'total_sequences': total_sequences,
            'databases': ['MIT-BIH Arrhythmia', 'MIT-BIH Atrial Fibrillation'],
            'normal_cases': int(tn + fp),
            'abnormal_cases': int(tp + fn),
            'real_data_percentage': 100.0
        },
        'performance_metrics': {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_score': float(auc)
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'clinical_assessment': {
            'clinical_readiness': sensitivity >= 0.95 and specificity >= 0.85,
            'sensitivity_adequate': sensitivity >= 0.95,
            'specificity_adequate': specificity >= 0.85,
            'dataset_size': 'Large (176K+ sequences)',
            'data_quality': 'High - Real PhysioNet data'
        }
    }
    
    with open('models/medical_metrics_176k.json', 'w') as f:
        json.dump(medical_metrics, f, indent=2)
    
    print("✅ Model and metrics saved successfully!")
    print("\nFiles created:")
    print("- models/heart_anomaly_medical_lstm_176k.keras")
    print("- models/scaler_medical_176k.pkl")
    print("- models/scaler_params_medical_176k.json")
    print("- models/medical_metrics_176k.json")

if __name__ == "__main__":
    train_with_current_data()
