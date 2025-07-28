"""
Export real medical data to CSV for reference and analysis
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def export_sample_data_to_csv(sample_size=1000):
    """Export a sample of the real medical data to CSV"""
    print("📊 EXPORTING REAL MEDICAL DATA TO CSV")
    print("=" * 50)
    
    # Load the real medical data
    sequences_path = Path('data/processed/real_medical_sequences.npy')
    labels_path = Path('data/processed/real_medical_labels.npy')
    
    if not sequences_path.exists() or not labels_path.exists():
        print("❌ Real medical data not found!")
        return
    
    sequences = np.load(sequences_path)
    labels = np.load(labels_path)
    
    print(f"✅ Loaded data: {sequences.shape[0]:,} sequences")
    
    # Sample data for CSV export
    if len(sequences) > sample_size:
        # Stratified sampling to maintain class balance
        normal_indices = np.where(labels == 0)[0]
        abnormal_indices = np.where(labels == 1)[0]
        
        normal_sample_size = int(sample_size * 0.88)  # Maintain ~88% normal
        abnormal_sample_size = sample_size - normal_sample_size
        
        normal_sample = np.random.choice(normal_indices, 
                                       min(normal_sample_size, len(normal_indices)), 
                                       replace=False)
        abnormal_sample = np.random.choice(abnormal_indices, 
                                         min(abnormal_sample_size, len(abnormal_indices)), 
                                         replace=False)
        
        sample_indices = np.concatenate([normal_sample, abnormal_sample])
        np.random.shuffle(sample_indices)
        
        sample_sequences = sequences[sample_indices]
        sample_labels = labels[sample_indices]
    else:
        sample_sequences = sequences
        sample_labels = labels
        sample_indices = np.arange(len(sequences))
    
    print(f"📋 Exporting {len(sample_sequences):,} sequences to CSV...")
    
    # Create comprehensive CSV data
    csv_data = []
    feature_names = ['ECG', 'Heart_Rate', 'SpO2', 'BP_Systolic', 'BP_Diastolic']
    
    for seq_idx, (sequence, label) in enumerate(zip(sample_sequences, sample_labels)):
        original_idx = sample_indices[seq_idx] if len(sample_indices) == len(sample_sequences) else seq_idx
        
        # Calculate sequence statistics
        seq_stats = {}
        for feat_idx, feat_name in enumerate(feature_names):
            feat_data = sequence[:, feat_idx]
            seq_stats.update({
                f'{feat_name}_mean': np.mean(feat_data),
                f'{feat_name}_std': np.std(feat_data),
                f'{feat_name}_min': np.min(feat_data),
                f'{feat_name}_max': np.max(feat_data),
                f'{feat_name}_median': np.median(feat_data)
            })
        
        # Add sequence metadata
        row_data = {
            'sequence_id': original_idx,
            'label': int(label),
            'label_text': 'Abnormal' if label == 1 else 'Normal',
            **seq_stats
        }
        
        csv_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    
    # Export to CSV
    csv_path = 'data/processed/medical_data_sample.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Sample data exported to: {csv_path}")
    print(f"📊 CSV contains {len(df)} sequences with statistical summaries")
    
    return df

def export_detailed_sequences_csv(num_sequences=50):
    """Export detailed timestep-by-timestep data for specific sequences"""
    print(f"\n📋 EXPORTING DETAILED SEQUENCES (Timestep-by-timestep)")
    print("=" * 60)
    
    # Load data
    sequences = np.load('data/processed/real_medical_sequences.npy')
    labels = np.load('data/processed/real_medical_labels.npy')
    
    # Select diverse examples
    normal_indices = np.where(labels == 0)[0]
    abnormal_indices = np.where(labels == 1)[0]
    
    # Get examples from each class
    normal_examples = np.random.choice(normal_indices, min(25, len(normal_indices)), replace=False)
    abnormal_examples = np.random.choice(abnormal_indices, min(25, len(abnormal_indices)), replace=False)
    
    selected_indices = np.concatenate([normal_examples, abnormal_examples])
    
    detailed_data = []
    feature_names = ['ECG', 'Heart_Rate', 'SpO2', 'BP_Systolic', 'BP_Diastolic']
    
    for seq_idx in selected_indices:
        sequence = sequences[seq_idx]
        label = labels[seq_idx]
        
        for timestep in range(len(sequence)):
            row = {
                'sequence_id': seq_idx,
                'timestep': timestep,
                'label': int(label),
                'label_text': 'Abnormal' if label == 1 else 'Normal'
            }
            
            # Add feature values for this timestep
            for feat_idx, feat_name in enumerate(feature_names):
                row[feat_name] = sequence[timestep, feat_idx]
            
            detailed_data.append(row)
    
    # Create detailed DataFrame
    detailed_df = pd.DataFrame(detailed_data)
    
    # Export detailed CSV
    detailed_csv_path = 'data/processed/medical_data_detailed.csv'
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"✅ Detailed sequences exported to: {detailed_csv_path}")
    print(f"📊 Contains {len(selected_indices)} sequences × 100 timesteps = {len(detailed_df):,} rows")
    
    return detailed_df

def create_data_analysis_report(df):
    """Create a comprehensive data analysis report"""
    print(f"\n📈 CREATING DATA ANALYSIS REPORT")
    print("=" * 40)
    
    # Basic statistics
    print(f"📊 Dataset Overview:")
    print(f"   Total sequences: {len(df):,}")
    print(f"   Normal cases: {len(df[df['label'] == 0]):,} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
    print(f"   Abnormal cases: {len(df[df['label'] == 1]):,} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
    
    # Feature analysis
    feature_names = ['ECG', 'Heart_Rate', 'SpO2', 'BP_Systolic', 'BP_Diastolic']
    
    print(f"\n📋 Feature Statistics:")
    for feature in feature_names:
        mean_col = f'{feature}_mean'
        if mean_col in df.columns:
            normal_mean = df[df['label'] == 0][mean_col].mean()
            abnormal_mean = df[df['label'] == 1][mean_col].mean()
            
            print(f"   {feature}:")
            print(f"     Normal avg:   {normal_mean:.2f}")
            print(f"     Abnormal avg: {abnormal_mean:.2f}")
            print(f"     Difference:   {abs(abnormal_mean - normal_mean):.2f}")
    
    # Create analysis report file
    report_path = 'data/processed/data_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("REAL MEDICAL DATA ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"  Total sequences: {len(df):,}\n")
        f.write(f"  Normal cases: {len(df[df['label'] == 0]):,} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)\n")
        f.write(f"  Abnormal cases: {len(df[df['label'] == 1]):,} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)\n\n")
        
        f.write("Feature Analysis:\n")
        for feature in feature_names:
            mean_col = f'{feature}_mean'
            if mean_col in df.columns:
                normal_stats = df[df['label'] == 0][mean_col].describe()
                abnormal_stats = df[df['label'] == 1][mean_col].describe()
                
                f.write(f"\n{feature}:\n")
                f.write(f"  Normal - Mean: {normal_stats['mean']:.2f}, Std: {normal_stats['std']:.2f}\n")
                f.write(f"  Abnormal - Mean: {abnormal_stats['mean']:.2f}, Std: {abnormal_stats['std']:.2f}\n")
    
    print(f"✅ Analysis report saved to: {report_path}")

def view_sample_data(df, num_rows=20):
    """Display sample data in a readable format"""
    print(f"\n👀 SAMPLE DATA PREVIEW (First {num_rows} rows)")
    print("=" * 80)
    
    # Display basic info
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nFirst {num_rows} rows:")
    print(df.head(num_rows).to_string(index=False))
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nBasic statistics:")
    print(df.describe())

def create_visualizations(df):
    """Create visualizations of the medical data"""
    print(f"\n📊 CREATING DATA VISUALIZATIONS")
    print("=" * 40)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    feature_names = ['ECG', 'Heart_Rate', 'SpO2', 'BP_Systolic', 'BP_Diastolic']
    
    for i, feature in enumerate(feature_names):
        row = i // 3
        col = i % 3
        
        mean_col = f'{feature}_mean'
        if mean_col in df.columns:
            # Distribution comparison
            normal_data = df[df['label'] == 0][mean_col]
            abnormal_data = df[df['label'] == 1][mean_col]
            
            axes[row, col].hist(normal_data, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
            axes[row, col].hist(abnormal_data, bins=30, alpha=0.7, label='Abnormal', color='red', density=True)
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(f'{feature} Mean Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(feature_names) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    viz_path = 'data/processed/medical_data_visualizations.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {viz_path}")

def main():
    """Main function to export and analyze medical data"""
    print("🏥 MEDICAL DATA CSV EXPORT & ANALYSIS TOOL")
    print("=" * 60)
    
    # Export sample data
    df_sample = export_sample_data_to_csv(sample_size=1000)
    
    if df_sample is not None:
        # View sample data
        view_sample_data(df_sample)
        
        # Create analysis report
        create_data_analysis_report(df_sample)
        
        # Create visualizations
        create_visualizations(df_sample)
        
        # Export detailed sequences
        df_detailed = export_detailed_sequences_csv(num_sequences=50)
        
        print(f"\n🎉 DATA EXPORT COMPLETE!")
        print(f"📁 Files created:")
        print(f"   - data/processed/medical_data_sample.csv (1,000 sequence summaries)")
        print(f"   - data/processed/medical_data_detailed.csv (50 sequences × 100 timesteps)")
        print(f"   - data/processed/data_analysis_report.txt")
        print(f"   - data/processed/medical_data_visualizations.png")
        
        print(f"\n📋 How to use the CSV files:")
        print(f"   1. Open medical_data_sample.csv in Excel/Google Sheets for overview")
        print(f"   2. Open medical_data_detailed.csv to see timestep-by-timestep data")
        print(f"   3. Read data_analysis_report.txt for statistical summary")
        print(f"   4. View medical_data_visualizations.png for data distributions")

if __name__ == "__main__":
    main()
