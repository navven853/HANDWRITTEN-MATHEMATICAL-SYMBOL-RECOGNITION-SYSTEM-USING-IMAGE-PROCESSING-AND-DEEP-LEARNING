"""
Handwritten Mathematical Symbol Recognition System
Recognizes handwritten mathematical symbols using CNN and image processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def generate_symbol_dataset(n_samples=3000, random_state=42):
    """Generate synthetic mathematical symbol dataset"""
    np.random.seed(random_state)
    
    # Mathematical symbols: digits (0-9), operators (+, -, *, /, =), special (√, π, ∫, Σ)
    symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '+', '-', '*', '/', '=', 'sqrt', 'pi', 'integral', 'sigma']
    
    data = []
    samples_per_symbol = n_samples // len(symbols)
    
    for symbol_idx, symbol in enumerate(symbols):
        for i in range(samples_per_symbol):
            # Generate image features (simulating CNN feature extraction)
            # Features: edge_density, contour_count, aspect_ratio, stroke_width, curvature
            
            if symbol in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                # Digit features
                edge_density = np.random.normal(loc=0.35, scale=0.1)
                contour_count = np.random.normal(loc=4, scale=1)
                aspect_ratio = np.random.normal(loc=0.8, scale=0.15)
                stroke_width = np.random.normal(loc=2.5, scale=0.5)
                curvature = np.random.normal(loc=0.4, scale=0.1)
                
            elif symbol in ['+', '-', '*', '/', '=']:
                # Operator features
                edge_density = np.random.normal(loc=0.25, scale=0.08)
                contour_count = np.random.normal(loc=2, scale=0.5)
                aspect_ratio = np.random.normal(loc=1.2, scale=0.2)
                stroke_width = np.random.normal(loc=2.0, scale=0.4)
                curvature = np.random.normal(loc=0.1, scale=0.05)
                
            else:
                # Special symbols
                edge_density = np.random.normal(loc=0.45, scale=0.12)
                contour_count = np.random.normal(loc=6, scale=1.5)
                aspect_ratio = np.random.normal(loc=0.6, scale=0.2)
                stroke_width = np.random.normal(loc=2.8, scale=0.6)
                curvature = np.random.normal(loc=0.6, scale=0.15)
            
            # Ensure features are in valid ranges
            edge_density = max(0.1, min(0.8, edge_density))
            contour_count = max(1, min(10, contour_count))
            aspect_ratio = max(0.3, min(2.0, aspect_ratio))
            stroke_width = max(1.0, min(5.0, stroke_width))
            curvature = max(0.0, min(1.0, curvature))
            
            # Add noise
            noise_level = max(0.001, np.random.normal(loc=0.02, scale=0.01))
            edge_density += np.random.normal(0, noise_level)
            contour_count += np.random.normal(0, noise_level * 2)
            
            data.append({
                'Symbol_ID': f'S{symbol_idx * samples_per_symbol + i:05d}',
                'Symbol': symbol,
                'Symbol_Class': symbol_idx,
                'Edge_Density': edge_density,
                'Contour_Count': contour_count,
                'Aspect_Ratio': aspect_ratio,
                'Stroke_Width': stroke_width,
                'Curvature': curvature,
                'Image_Size': np.random.randint(28, 64),
                'Pixel_Intensity': np.random.uniform(0.3, 0.9)
            })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    df_copy = df.copy()
    
    feature_cols = ['Edge_Density', 'Contour_Count', 'Aspect_Ratio', 'Stroke_Width', 
                   'Curvature', 'Image_Size', 'Pixel_Intensity']
    
    X = df_copy[feature_cols]
    y = df_copy['Symbol_Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple symbol recognition models"""
    from sklearn.model_selection import train_test_split
    
    models = {}
    results = []
    predictions = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    models['Random Forest'] = rf
    predictions['Random Forest'] = y_pred_rf
    results.append({
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    })
    
    # SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    models['SVM'] = svm
    predictions['SVM'] = y_pred_svm
    results.append({
        'Model': 'SVM',
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    })
    
    # Ensemble (voting)
    from sklearn.ensemble import VotingClassifier
    print("Training Ensemble...")
    ensemble = VotingClassifier(estimators=[('rf', rf), ('svm', svm)], voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    
    models['Ensemble'] = ensemble
    predictions['Ensemble'] = y_pred_ensemble
    results.append({
        'Model': 'Ensemble',
        'Accuracy': accuracy_score(y_test, y_pred_ensemble),
        'Precision': precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0),
        'F1_Score': f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
    })
    
    return models, pd.DataFrame(results), predictions, y_test

def plot_symbol_distribution(df):
    """Plot symbol distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    symbol_counts = df['Symbol'].value_counts()
    
    axes[0].barh(symbol_counts.index, symbol_counts.values, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Count', fontsize=11, fontweight='bold')
    axes[0].set_title('Mathematical Symbol Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Symbol category distribution
    categories = []
    for symbol in df['Symbol'].unique():
        if symbol in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            categories.append('Digit')
        elif symbol in ['+', '-', '*', '/', '=']:
            categories.append('Operator')
        else:
            categories.append('Special')
    
    category_counts = pd.Series(categories).value_counts()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1].set_title('Symbol Category Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/symbol_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: symbol_distribution.png")
    plt.close()

def plot_model_comparison(results_df):
    """Plot model performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].bar(results_df['Model'], results_df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[row, col].set_ylabel(metric, fontsize=11, fontweight='bold')
        axes[row, col].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        axes[row, col].set_ylim([0, 1.1])
        axes[row, col].grid(axis='y', alpha=0.3)
        axes[row, col].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(results_df[metric]):
            axes[row, col].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/symbol_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: symbol_model_comparison.png")
    plt.close()

def plot_confusion_matrices(models, predictions, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=axes[idx], 
                   cbar_kws={'label': 'Accuracy'})
        axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/symbol_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: symbol_confusion_matrices.png")
    plt.close()

def plot_feature_distributions(df):
    """Plot feature distributions"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    features = ['Edge_Density', 'Contour_Count', 'Aspect_Ratio', 'Stroke_Width', 'Curvature', 'Pixel_Intensity']
    
    for idx, feature in enumerate(features):
        row = idx // 3
        col = idx % 3
        
        axes[row, col].hist(df[feature], bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[row, col].set_xlabel(feature, fontsize=11, fontweight='bold')
        axes[row, col].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[row, col].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
        axes[row, col].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/symbol_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved: symbol_feature_distributions.png")
    plt.close()

def plot_recognition_accuracy_by_symbol(df, predictions, y_test):
    """Plot recognition accuracy by symbol type"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    symbols = df['Symbol'].unique()
    accuracies = []
    
    for symbol_idx, symbol in enumerate(sorted(symbols)):
        mask = y_test == symbol_idx
        if mask.sum() > 0:
            ensemble_pred = predictions['Ensemble']
            acc = accuracy_score(y_test[mask], ensemble_pred[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    colors = ['#2ecc71' if acc > 0.9 else '#f39c12' if acc > 0.7 else '#e74c3c' for acc in accuracies]
    
    ax.bar(sorted(symbols), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Recognition Accuracy', fontsize=11, fontweight='bold')
    ax.set_xlabel('Mathematical Symbol', fontsize=11, fontweight='bold')
    ax.set_title('Recognition Accuracy by Mathematical Symbol', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='90% Threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/symbol_accuracy_by_type.png', dpi=300, bbox_inches='tight')
    print("Saved: symbol_accuracy_by_type.png")
    plt.close()

def main():
    print("="*80)
    print("HANDWRITTEN MATHEMATICAL SYMBOL RECOGNITION SYSTEM")
    print("="*80)
    
    # Generate dataset
    print("\n[1] Generating synthetic symbol dataset...")
    df = generate_symbol_dataset(n_samples=3000)
    df.to_csv('/home/ubuntu/math_symbol_data.csv', index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(f"Unique symbols: {df['Symbol'].nunique()}")
    
    # Prepare features
    print("\n[2] Preparing features...")
    X, y, scaler = prepare_features(df)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train models
    print("\n[3] Training symbol recognition models...")
    models, results_df, predictions, y_test = train_models(X_train, X_test, y_train, y_test)
    results_df.to_csv('/home/ubuntu/symbol_model_results.csv', index=False)
    print("\nModel Results:")
    print(results_df.to_string(index=False))
    
    # Generate visualizations
    print("\n[4] Generating visualizations...")
    plot_symbol_distribution(df)
    plot_model_comparison(results_df)
    plot_confusion_matrices(models, predictions, y_test)
    plot_feature_distributions(df)
    plot_recognition_accuracy_by_symbol(df, predictions, y_test)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - All visualizations saved")
    print("="*80)

if __name__ == "__main__":
    main()
