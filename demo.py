#!/usr/bin/env python3
"""
The North Face Product Recommendation System - Demo Script

This script demonstrates the core ML functionality without the Streamlit interface.
It shows how the clustering, recommendation system, and topic modeling work.
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

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

def create_tfidf_matrix(df):
    """Create TF-IDF matrix from cleaned descriptions"""
    print("🔤 Creating TF-IDF matrix...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['description_clean'])
    print(f"✅ TF-IDF matrix created: {tfidf_matrix.shape}")
    
    return tfidf_matrix, vectorizer

def perform_clustering(tfidf_matrix, eps=0.45, min_samples=5):
    """Perform DBSCAN clustering on TF-IDF matrix"""
    print(f"🔍 Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    noise_percentage = (n_noise / len(cluster_labels)) * 100
    
    print(f"✅ Clustering completed:")
    print(f"   📊 Number of clusters: {n_clusters}")
    print(f"   🔇 Noise points: {n_noise} ({noise_percentage:.1f}%)")
    
    return cluster_labels

def perform_topic_modeling(tfidf_matrix, n_components=15):
    """Perform topic modeling using TruncatedSVD"""
    print(f"🎯 Performing topic modeling ({n_components} topics)...")
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    topic_matrix = svd.fit_transform(tfidf_matrix)
    
    print(f"✅ Topic modeling completed: {topic_matrix.shape}")
    
    return topic_matrix, svd

def find_similar_products(product_id, df, cluster_labels, tfidf_matrix, k=5):
    """Find similar products based on clustering or cosine similarity"""
    if product_id not in df.index:
        return []
    
    product_cluster = cluster_labels[product_id]
    
    # If product is in a cluster (not noise), return products from same cluster
    if product_cluster != -1:
        same_cluster_products = df[cluster_labels == product_cluster].index.tolist()
        same_cluster_products = [pid for pid in same_cluster_products if pid != product_id]
        return same_cluster_products[:k]
    
    # If product is noise, use cosine similarity
    product_vector = tfidf_matrix[product_id]
    similarities = cosine_similarity(product_vector, tfidf_matrix).flatten()
    similarities[product_id] = 0  # Exclude the product itself
    
    similar_indices = similarities.argsort()[::-1][:k]
    return similar_indices.tolist()

def search_products_by_keyword(keyword, df, top_k=10):
    """Search products by keyword in description"""
    mask = df['description'].str.contains(keyword, case=False, na=False)
    matching_products = df[mask].index.tolist()
    
    if not matching_products:
        # Fuzzy matching as fallback
        descriptions = df['description'].str.lower().tolist()
        close_matches = get_close_matches(keyword.lower(), descriptions, n=top_k, cutoff=0.6)
        matching_products = [df.index[i] for i, desc in enumerate(descriptions) if desc in close_matches]
    
    return matching_products[:top_k]

def display_product(df, product_id, title="Product"):
    """Display product information"""
    if product_id in df.index:
        product = df.loc[product_id]
        print(f"\n{title}:")
        print(f"  🆔 ID: {product_id}")
        if 'cluster' in df.columns:
            print(f"  📊 Cluster: {product['cluster']}")
        print(f"  📝 Description: {product['description'][:200]}...")
        print("-" * 80)
    else:
        print(f"❌ Product ID {product_id} not found")

def demo_clustering_analysis(df, tfidf_matrix, cluster_labels):
    """Demonstrate clustering analysis"""
    print("\n" + "="*80)
    print("📈 CLUSTERING ANALYSIS DEMO")
    print("="*80)
    
    df['cluster'] = cluster_labels
    
    # Show cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("\n📊 Cluster Distribution:")
    for cluster_id, count in cluster_counts.head(10).items():
        if cluster_id == -1:
            print(f"   Noise: {count} products")
        else:
            print(f"   Cluster {cluster_id}: {count} products")
    
    # Show sample products from largest clusters
    top_clusters = cluster_counts[cluster_counts.index != -1].head(3).index
    
    for cluster_id in top_clusters:
        cluster_products = df[df['cluster'] == cluster_id]
        print(f"\n🔍 Sample products from Cluster {cluster_id}:")
        for i, (idx, product) in enumerate(cluster_products.head(3).iterrows()):
            print(f"   {i+1}. ID {idx}: {product['description'][:100]}...")

def demo_recommendation_system(df, cluster_labels, tfidf_matrix):
    """Demonstrate recommendation system"""
    print("\n" + "="*80)
    print("💡 RECOMMENDATION SYSTEM DEMO")
    print("="*80)
    
    # Test with a few sample products
    test_products = [1, 5, 10, 15, 20]
    
    for product_id in test_products:
        if product_id in df.index:
            print(f"\n🎯 Testing recommendations for Product {product_id}:")
            display_product(df, product_id, "Selected Product")
            
            similar_products = find_similar_products(product_id, df, cluster_labels, tfidf_matrix)
            
            if similar_products:
                print("\n💡 Recommended Products:")
                for i, sim_id in enumerate(similar_products, 1):
                    print(f"   {i}. ID {sim_id}: {df.loc[sim_id, 'description'][:100]}...")
            else:
                print("   ❌ No similar products found")
            
            print("\n" + "-"*60)

def demo_keyword_search(df, cluster_labels, tfidf_matrix):
    """Demonstrate keyword search functionality"""
    print("\n" + "="*80)
    print("🔍 KEYWORD SEARCH DEMO")
    print("="*80)
    
    test_keywords = ["jacket", "shorts", "fleece", "alpine", "waterproof"]
    
    for keyword in test_keywords:
        print(f"\n🔍 Searching for '{keyword}':")
        matching_products = search_products_by_keyword(keyword, df)
        
        if matching_products:
            print(f"   ✅ Found {len(matching_products)} products")
            
            # Show first match and its recommendations
            first_match = matching_products[0]
            display_product(df, first_match, f"First match for '{keyword}'")
            
            similar_products = find_similar_products(first_match, df, cluster_labels, tfidf_matrix, k=3)
            if similar_products:
                print(f"   💡 Similar products:")
                for sim_id in similar_products:
                    print(f"      - ID {sim_id}: {df.loc[sim_id, 'description'][:80]}...")
        else:
            print(f"   ❌ No products found for '{keyword}'")

def demo_topic_modeling(tfidf_matrix, vectorizer, svd_model):
    """Demonstrate topic modeling"""
    print("\n" + "="*80)
    print("🎯 TOPIC MODELING DEMO")
    print("="*80)
    
    feature_names = vectorizer.get_feature_names_out()
    n_top_words = 8
    
    print("\n📋 Discovered Topics:")
    
    for topic_idx, topic in enumerate(svd_model.components_[:10]):  # Show first 10 topics
        top_words_idx = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"   Topic {topic_idx}: {', '.join(top_words)}")

def main():
    """Main demo function"""
    print("🏔️ THE NORTH FACE PRODUCT RECOMMENDATION SYSTEM - DEMO")
    print("="*80)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    if df is None:
        return
    
    # Create TF-IDF matrix
    tfidf_matrix, vectorizer = create_tfidf_matrix(df)
    
    # Perform clustering
    cluster_labels = perform_clustering(tfidf_matrix)
    
    # Perform topic modeling
    topic_matrix, svd_model = perform_topic_modeling(tfidf_matrix)
    
    # Run demos
    demo_clustering_analysis(df, tfidf_matrix, cluster_labels)
    demo_recommendation_system(df, cluster_labels, tfidf_matrix)
    demo_keyword_search(df, cluster_labels, tfidf_matrix)
    demo_topic_modeling(tfidf_matrix, vectorizer, svd_model)
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETED!")
    print("\n💡 To run the full interactive Streamlit application:")
    print("   streamlit run app.py")
    print("="*80)

if __name__ == "__main__":
    main()