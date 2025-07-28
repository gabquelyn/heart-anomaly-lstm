"""
Quick preview of medical data without full export
"""
import numpy as np
import pandas as pd
from pathlib import Path

def quick_preview():
    """Quick preview of the medical data"""
    print("👀 QUICK MEDICAL DATA PREVIEW")
    print("=" * 40)
    
    # Load data
    sequences = np.load('data/processed/real_medical_sequences.npy')
    labels = np.load('data/processed/real_medical_labels.npy')
    
    print(f"📊 Dataset Overview:")
    print(f"   Shape: {sequences.shape}")
    print(f"   Total sequences: {len(sequences):,}")
    print(f"   Normal: {np.sum(labels == 0):,} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    print(f"   Abnormal: {np.sum(labels == 1):,} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    
    # Show first few sequences
    print(f"\n📋 First 3 sequences (first 10 timesteps each):")
    feature_names = ['ECG', 'HR', 'SpO2', 'BP_Sys', 'BP_Dia']
    
    for seq_idx in range(min(3, len(sequences))):
        sequence = sequences[seq_idx]
        label = labels[seq_idx]
        
        print(f"\nSequence {seq_idx} ({'Abnormal' if label == 1 else 'Normal'}):")
        
        # Create a mini DataFrame for this sequence
        seq_data = []
        for t in range(min(10, len(sequence))):
            row = {'Timestep': t}
            for feat_idx, feat_name in enumerate(feature_names):
                row[feat_name] = f"{sequence[t, feat_idx]:.3f}"
            seq_data.append(row)
        
        df_seq = pd.DataFrame(seq_data)
        print(df_seq.to_string(index=False))
    
    # Feature statistics
    print(f"\n📈 Feature Statistics (across all sequences):")
    for feat_idx, feat_name in enumerate(feature_names):
        feat_data = sequences[:, :, feat_idx].flatten()
        print(f"   {feat_name}:")
        print(f"     Range: [{np.min(feat_data):.3f}, {np.max(feat_data):.3f}]")
        print(f"     Mean: {np.mean(feat_data):.3f} ± {np.std(feat_data):.3f}")

if __name__ == "__main__":
    quick_preview()
