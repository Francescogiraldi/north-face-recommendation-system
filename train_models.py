#!/usr/bin/env python3
"""
The North Face Product Recommendation System - Model Training Script

This script trains all ML models and saves them using joblib for later use in the Streamlit app.
Run this script first to generate the trained models before running the Streamlit application.
"""

import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Create models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the North Face catalog data"""
    print("📁 Loading North Face catalog...")
    
    try:
        df = pd.read_csv("northface_catalog.csv")
        print(f"✅ Loaded {len(df)} products")
        
        # Text preprocessing function
        def clean_text(text):
            if pd.isna(text):
                return ""
            # Convert to lowercase and remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', str(text).lower())
            # Remove special characters and digits
            text = re.sub(r'[^a-z\s]', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        
        # Apply text cleaning
        df['description_clean'] = df['description'].apply(clean_text)
        print("✅ Text preprocessing completed")
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def train_tfidf_vectorizer(df):
    """Train and save TF-IDF vectorizer"""
    print("🔤 Training TF-IDF vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit the vectorizer and transform the data
    tfidf_matrix = vectorizer.fit_transform(df['description_clean'])
    
    print(f"✅ TF-IDF matrix created: {tfidf_matrix.shape}")
    
    # Save the vectorizer and matrix
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, MODELS_DIR / "tfidf_matrix.joblib")
    
    print(f"💾 TF-IDF vectorizer saved to {MODELS_DIR / 'tfidf_vectorizer.joblib'}")
    print(f"💾 TF-IDF matrix saved to {MODELS_DIR / 'tfidf_matrix.joblib'}")
    
    return tfidf_matrix, vectorizer

def train_clustering_model(tfidf_matrix, eps=0.45, min_samples=5):
    """Train and save DBSCAN clustering model"""
    print(f"🔍 Training DBSCAN clustering model (eps={eps}, min_samples={min_samples})...")
    
    # Train DBSCAN clustering
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = clustering_model.fit_predict(tfidf_matrix.toarray())
    
    # Calculate clustering statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_percentage = (n_noise / len(cluster_labels)) * 100
    
    print(f"✅ Clustering completed:")
    print(f"   📊 Number of clusters: {n_clusters}")
    print(f"   🔇 Noise points: {n_noise} ({noise_percentage:.1f}%)")
    
    # Save the clustering model and labels
    joblib.dump(clustering_model, MODELS_DIR / "dbscan_model.joblib")
    joblib.dump(cluster_labels, MODELS_DIR / "cluster_labels.joblib")
    
    print(f"💾 DBSCAN model saved to {MODELS_DIR / 'dbscan_model.joblib'}")
    print(f"💾 Cluster labels saved to {MODELS_DIR / 'cluster_labels.joblib'}")
    
    return clustering_model, cluster_labels

def train_topic_model(tfidf_matrix, n_components=15):
    """Train and save topic modeling (TruncatedSVD) model"""
    print(f"🎯 Training topic modeling (TruncatedSVD) with {n_components} topics...")
    
    # Train TruncatedSVD for topic modeling
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    topic_matrix = svd_model.fit_transform(tfidf_matrix)
    
    print(f"✅ Topic modeling completed: {topic_matrix.shape}")
    print(f"   📊 Explained variance ratio: {svd_model.explained_variance_ratio_.sum():.3f}")
    
    # Save the topic model and matrix
    joblib.dump(svd_model, MODELS_DIR / "svd_topic_model.joblib")
    joblib.dump(topic_matrix, MODELS_DIR / "topic_matrix.joblib")
    
    print(f"💾 SVD topic model saved to {MODELS_DIR / 'svd_topic_model.joblib'}")
    print(f"💾 Topic matrix saved to {MODELS_DIR / 'topic_matrix.joblib'}")
    
    return svd_model, topic_matrix

def create_similarity_matrix(tfidf_matrix):
    """Pre-compute and save cosine similarity matrix for faster recommendations"""
    print("🔄 Computing cosine similarity matrix...")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print(f"✅ Similarity matrix computed: {similarity_matrix.shape}")
    
    # Save the similarity matrix
    joblib.dump(similarity_matrix, MODELS_DIR / "similarity_matrix.joblib")
    
    print(f"💾 Similarity matrix saved to {MODELS_DIR / 'similarity_matrix.joblib'}")
    
    return similarity_matrix

def save_processed_data(df):
    """Save the processed dataframe"""
    print("💾 Saving processed data...")
    
    # Save the processed dataframe
    df.to_csv(MODELS_DIR / "processed_data.csv", index=False)
    joblib.dump(df, MODELS_DIR / "processed_dataframe.joblib")
    
    print(f"💾 Processed data saved to {MODELS_DIR / 'processed_data.csv'}")
    print(f"💾 Processed dataframe saved to {MODELS_DIR / 'processed_dataframe.joblib'}")

def create_model_metadata(df, tfidf_matrix, cluster_labels, topic_matrix, vectorizer, clustering_model, svd_model):
    """Create and save model metadata for the Streamlit app"""
    print("📋 Creating model metadata...")
    
    metadata = {
        'dataset_info': {
            'n_products': len(df),
            'avg_description_length': df['description'].str.len().mean(),
            'data_columns': list(df.columns)
        },
        'tfidf_info': {
            'matrix_shape': tfidf_matrix.shape,
            'n_features': tfidf_matrix.shape[1],
            'vocabulary_size': len(vectorizer.vocabulary_)
        },
        'clustering_info': {
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise_points': list(cluster_labels).count(-1),
            'noise_percentage': (list(cluster_labels).count(-1) / len(cluster_labels)) * 100,
            'eps': clustering_model.eps,
            'min_samples': clustering_model.min_samples
        },
        'topic_modeling_info': {
            'n_topics': svd_model.n_components,
            'explained_variance_ratio': svd_model.explained_variance_ratio_.sum(),
            'topic_matrix_shape': topic_matrix.shape
        },
        'model_files': {
            'tfidf_vectorizer': 'tfidf_vectorizer.joblib',
            'tfidf_matrix': 'tfidf_matrix.joblib',
            'dbscan_model': 'dbscan_model.joblib',
            'cluster_labels': 'cluster_labels.joblib',
            'svd_topic_model': 'svd_topic_model.joblib',
            'topic_matrix': 'topic_matrix.joblib',
            'similarity_matrix': 'similarity_matrix.joblib',
            'processed_dataframe': 'processed_dataframe.joblib'
        }
    }
    
    # Save metadata
    joblib.dump(metadata, MODELS_DIR / "model_metadata.joblib")
    
    print(f"💾 Model metadata saved to {MODELS_DIR / 'model_metadata.joblib'}")
    
    return metadata

def optimize_clustering_parameters(tfidf_matrix, df):
    """Find optimal clustering parameters"""
    print("🔧 Optimizing clustering parameters...")
    
    best_params = None
    best_score = 0
    
    eps_values = [0.35, 0.40, 0.45, 0.50, 0.55]
    min_samples_values = [3, 4, 5, 6]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = clustering.fit_predict(tfidf_matrix.toarray())
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_percentage = (n_noise / len(labels)) * 100
            
            # Score based on number of clusters (10-20 ideal) and low noise (<30%)
            if 10 <= n_clusters <= 20 and noise_percentage < 30:
                score = n_clusters * (1 - noise_percentage/100)
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    print(f"   ✅ Better params found: eps={eps}, min_samples={min_samples}, clusters={n_clusters}, noise={noise_percentage:.1f}%")
    
    if best_params:
        print(f"🎯 Optimal parameters: {best_params}")
        return best_params['eps'], best_params['min_samples']
    else:
        print("⚠️ Using default parameters: eps=0.45, min_samples=5")
        return 0.45, 5

def main():
    """Main training function"""
    print("🏔️ THE NORTH FACE ML MODELS TRAINING")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Train TF-IDF vectorizer
    tfidf_matrix, vectorizer = train_tfidf_vectorizer(df)
    
    # Optimize clustering parameters
    optimal_eps, optimal_min_samples = optimize_clustering_parameters(tfidf_matrix, df)
    
    # Train clustering model with optimal parameters
    clustering_model, cluster_labels = train_clustering_model(
        tfidf_matrix, eps=optimal_eps, min_samples=optimal_min_samples
    )
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Train topic modeling
    svd_model, topic_matrix = train_topic_model(tfidf_matrix)
    
    # Create similarity matrix
    similarity_matrix = create_similarity_matrix(tfidf_matrix)
    
    # Save processed data
    save_processed_data(df)
    
    # Create and save metadata
    metadata = create_model_metadata(
        df, tfidf_matrix, cluster_labels, topic_matrix, 
        vectorizer, clustering_model, svd_model
    )
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("\n📊 Training Summary:")
    print(f"   📁 Products processed: {metadata['dataset_info']['n_products']}")
    print(f"   🔤 TF-IDF features: {metadata['tfidf_info']['n_features']}")
    print(f"   📊 Clusters created: {metadata['clustering_info']['n_clusters']}")
    print(f"   🔇 Noise percentage: {metadata['clustering_info']['noise_percentage']:.1f}%")
    print(f"   🎯 Topics extracted: {metadata['topic_modeling_info']['n_topics']}")
    print(f"   📈 Explained variance: {metadata['topic_modeling_info']['explained_variance_ratio']:.3f}")
    
    print(f"\n💾 All models saved in: {MODELS_DIR.absolute()}")
    print("\n🚀 You can now run the Streamlit app:")
    print("   streamlit run app.py")
    print("="*60)

if __name__ == "__main__":
    main()