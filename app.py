import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import get_close_matches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Models directory
MODELS_DIR = Path("models")

# Configure Streamlit page
st.set_page_config(
    page_title="The North Face - Product Recommendation System",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E8B57;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
    .product-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #228B22;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_trained_models():
    """Load all pre-trained models and data"""
    try:
        if not MODELS_DIR.exists():
            st.error("❌ Models directory not found! Please run 'python train_models.py' first to train the models.")
            return None
        
        # Check if all required model files exist
        required_files = [
            "processed_dataframe.joblib",
            "tfidf_vectorizer.joblib", 
            "tfidf_matrix.joblib",
            "cluster_labels.joblib",
            "svd_topic_model.joblib",
            "topic_matrix.joblib",
            "similarity_matrix.joblib",
            "model_metadata.joblib"
        ]
        
        missing_files = [f for f in required_files if not (MODELS_DIR / f).exists()]
        if missing_files:
            st.error(f"❌ Missing model files: {missing_files}. Please run 'python train_models.py' first.")
            return None
        
        # Load all models and data
        models = {
            'df': joblib.load(MODELS_DIR / "processed_dataframe.joblib"),
            'vectorizer': joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib"),
            'tfidf_matrix': joblib.load(MODELS_DIR / "tfidf_matrix.joblib"),
            'cluster_labels': joblib.load(MODELS_DIR / "cluster_labels.joblib"),
            'svd_model': joblib.load(MODELS_DIR / "svd_topic_model.joblib"),
            'topic_matrix': joblib.load(MODELS_DIR / "topic_matrix.joblib"),
            'similarity_matrix': joblib.load(MODELS_DIR / "similarity_matrix.joblib"),
            'metadata': joblib.load(MODELS_DIR / "model_metadata.joblib")
        }
        
        return models
    except Exception as e:
        st.error(f"Error loading trained models: {e}")
        return None

def retrain_clustering_with_params(tfidf_matrix, eps=0.45, min_samples=5):
    """Retrain DBSCAN clustering with custom parameters (for interactive analysis)"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())
    return cluster_labels

def retrain_topic_modeling_with_params(tfidf_matrix, n_components=15):
    """Retrain topic modeling with custom parameters (for interactive analysis)"""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    topic_matrix = svd.fit_transform(tfidf_matrix)
    return topic_matrix, svd

def create_wordcloud(text_data, title="Word Cloud"):
    """Create and display a word cloud"""
    if not text_data or len(text_data.strip()) == 0:
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    return fig

def find_similar_products(product_id, df, cluster_labels, similarity_matrix, k=5):
    """Find similar products based on clustering or pre-computed cosine similarity"""
    if product_id not in df.index:
        return []
    
    product_cluster = cluster_labels[product_id]
    
    # If product is in a cluster (not noise), return products from same cluster
    if product_cluster != -1:
        same_cluster_products = df[cluster_labels == product_cluster].index.tolist()
        same_cluster_products = [pid for pid in same_cluster_products if pid != product_id]
        return same_cluster_products[:k]
    
    # If product is noise, use pre-computed cosine similarity
    similarities = similarity_matrix[product_id].copy()
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🏔️ The North Face Product Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load pre-trained models
    with st.spinner("Loading pre-trained models..."):
        models = load_trained_models()
    
    if models is None:
        st.error("Failed to load trained models. Please run 'python train_models.py' first to train the models.")
        st.info("💡 **How to train models:**\n1. Open terminal in the project directory\n2. Run: `python train_models.py`\n3. Wait for training to complete\n4. Refresh this page")
        return
    
    # Extract models and data
    df = models['df']
    vectorizer = models['vectorizer']
    tfidf_matrix = models['tfidf_matrix']
    cluster_labels = models['cluster_labels']
    svd_model = models['svd_model']
    topic_matrix = models['topic_matrix']
    similarity_matrix = models['similarity_matrix']
    metadata = models['metadata']
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Data Overview", "🔍 Product Search & Recommendations", "📈 Clustering Analysis", "🎯 Topic Modeling"]
    )
    
    # Display model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.markdown(f"**Products:** {metadata['dataset_info']['n_products']}")
    st.sidebar.markdown(f"**Features:** {metadata['tfidf_info']['n_features']}")
    st.sidebar.markdown(f"**Clusters:** {metadata['clustering_info']['n_clusters']}")
    st.sidebar.markdown(f"**Topics:** {metadata['topic_modeling_info']['n_topics']}")
    
    if page == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Products", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_desc_length = df['description'].str.len().mean()
            st.metric("Avg Description Length", f"{avg_desc_length:.0f} chars")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("TF-IDF Features", metadata['tfidf_info']['n_features'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📋 Sample Products")
        sample_df = df[['description']].head(10)
        sample_df['description'] = sample_df['description'].str[:200] + "..."
        st.dataframe(sample_df, use_container_width=True)
        
        # Description length distribution
        st.markdown("### 📏 Description Length Distribution")
        desc_lengths = df['description'].str.len()
        fig = px.histogram(x=desc_lengths, nbins=30, title="Distribution of Product Description Lengths")
        fig.update_layout(xaxis_title="Description Length (characters)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Product Search & Recommendations":
        st.markdown('<h2 class="sub-header">🔍 Product Search & Recommendations</h2>', unsafe_allow_html=True)
        
        # Use pre-trained clustering results
        # cluster_labels are already loaded and assigned to df
        
        # Search options
        search_type = st.radio("Search by:", ["Product ID", "Keyword"])
        
        if search_type == "Product ID":
            product_id = st.number_input(
                "Enter Product ID:", 
                min_value=int(df.index.min()), 
                max_value=int(df.index.max()), 
                value=int(df.index.min())
            )
            
            if st.button("Get Recommendations"):
                if product_id in df.index:
                    # Display selected product
                    st.markdown("### 🎯 Selected Product")
                    selected_product = df.loc[product_id]
                    
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    st.write(f"**Product ID:** {product_id}")
                    st.write(f"**Cluster:** {selected_product['cluster']}")
                    st.write(f"**Description:** {selected_product['description'][:300]}...")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Get recommendations
                    similar_products = find_similar_products(product_id, df, cluster_labels, similarity_matrix)
                    
                    if similar_products:
                        st.markdown("### 💡 Recommended Products")
                        for i, sim_id in enumerate(similar_products, 1):
                            sim_product = df.loc[sim_id]
                            st.markdown('<div class="product-card">', unsafe_allow_html=True)
                            st.write(f"**#{i} - Product ID:** {sim_id}")
                            st.write(f"**Cluster:** {sim_product['cluster']}")
                            st.write(f"**Description:** {sim_product['description'][:200]}...")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No similar products found.")
                else:
                    st.error("Product ID not found in the catalog.")
        
        else:  # Keyword search
            keyword = st.text_input("Enter keyword to search:")
            
            if keyword and st.button("Search Products"):
                matching_products = search_products_by_keyword(keyword, df)
                
                if matching_products:
                    st.markdown(f"### 🔍 Search Results for '{keyword}'")
                    
                    if len(matching_products) == 1:
                        # Single result - show recommendations
                        product_id = matching_products[0]
                        selected_product = df.loc[product_id]
                        
                        st.markdown('<div class="product-card">', unsafe_allow_html=True)
                        st.write(f"**Product ID:** {product_id}")
                        st.write(f"**Description:** {selected_product['description'][:300]}...")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Get recommendations
                        similar_products = find_similar_products(product_id, df, cluster_labels, similarity_matrix)
                        
                        if similar_products:
                            st.markdown("### 💡 Recommended Similar Products")
                            for i, sim_id in enumerate(similar_products, 1):
                                sim_product = df.loc[sim_id]
                                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                                st.write(f"**#{i} - Product ID:** {sim_id}")
                                st.write(f"**Description:** {sim_product['description'][:200]}...")
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        # Multiple results - show list
                        st.write(f"Found {len(matching_products)} products:")
                        for product_id in matching_products:
                            product = df.loc[product_id]
                            st.markdown('<div class="product-card">', unsafe_allow_html=True)
                            st.write(f"**Product ID:** {product_id}")
                            st.write(f"**Description:** {product['description'][:200]}...")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.info("💡 Select a specific Product ID above to get recommendations.")
                else:
                    st.warning(f"No products found for keyword '{keyword}'.")
    
    elif page == "📈 Clustering Analysis":
        st.markdown('<h2 class="sub-header">📈 Clustering Analysis</h2>', unsafe_allow_html=True)
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Epsilon (eps)", 0.1, 1.0, 0.45, 0.05)
        with col2:
            min_samples = st.slider("Min Samples", 3, 10, 5)
        
        if st.button("Run Clustering Analysis"):
            with st.spinner("Performing clustering..."):
                cluster_labels_new = retrain_clustering_with_params(tfidf_matrix, eps, min_samples)
            
            df_temp = df.copy()
            df_temp['cluster'] = cluster_labels_new
            
            # Clustering statistics
            n_clusters = len(set(cluster_labels_new)) - (1 if -1 in cluster_labels_new else 0)
            n_noise = list(cluster_labels_new).count(-1)
            noise_percentage = (n_noise / len(cluster_labels_new)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", n_clusters)
            with col2:
                st.metric("Noise Points", n_noise)
            with col3:
                st.metric("Noise Percentage", f"{noise_percentage:.1f}%")
            
            # Cluster distribution
            st.markdown("### 📊 Cluster Distribution")
            cluster_counts = pd.Series(cluster_labels_new).value_counts().sort_index()
            
            fig = px.bar(
                x=cluster_counts.index, 
                y=cluster_counts.values,
                title="Products per Cluster",
                labels={'x': 'Cluster ID', 'y': 'Number of Products'}
            )
            fig.update_layout(xaxis_title="Cluster ID", yaxis_title="Number of Products")
            st.plotly_chart(fig, use_container_width=True)
            
            # Word clouds for top clusters
            st.markdown("### ☁️ Cluster Word Clouds")
            
            top_clusters = cluster_counts.head(6).index.tolist()
            if -1 in top_clusters:
                top_clusters.remove(-1)  # Remove noise cluster
            
            cols = st.columns(2)
            for i, cluster_id in enumerate(top_clusters[:4]):
                cluster_products = df_temp[df_temp['cluster'] == cluster_id]
                cluster_text = ' '.join(cluster_products['description_clean'].tolist())
                
                fig = create_wordcloud(cluster_text, f"Cluster {cluster_id} ({len(cluster_products)} products)")
                if fig:
                    with cols[i % 2]:
                        st.pyplot(fig)
                        plt.close()
    
    elif page == "🎯 Topic Modeling":
        st.markdown('<h2 class="sub-header">🎯 Topic Modeling Analysis</h2>', unsafe_allow_html=True)
        
        # Topic modeling parameters
        n_topics = st.slider("Number of Topics", 5, 25, 15)
        
        if st.button("Run Topic Modeling"):
            with st.spinner("Performing topic modeling..."):
                topic_matrix_new, svd_model_new = retrain_topic_modeling_with_params(tfidf_matrix, n_topics)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Display top words for each topic
            st.markdown("### 📋 Top Words per Topic")
            
            n_top_words = 10
            topics_data = []
            
            for topic_idx, topic in enumerate(svd_model_new.components_):
                top_words_idx = topic.argsort()[::-1][:n_top_words]
                top_words = [feature_names[i] for i in top_words_idx]
                topics_data.append({
                    'Topic': f'Topic {topic_idx}',
                    'Top Words': ', '.join(top_words)
                })
            
            topics_df = pd.DataFrame(topics_data)
            st.dataframe(topics_df, use_container_width=True)
            
            # Topic distribution visualization
            st.markdown("### 📊 Topic Weights Distribution")
            
            # Calculate average topic weights
            avg_topic_weights = topic_matrix_new.mean(axis=0)
            
            fig = px.bar(
                x=[f'Topic {i}' for i in range(len(avg_topic_weights))],
                y=avg_topic_weights,
                title="Average Topic Weights Across All Products"
            )
            fig.update_layout(xaxis_title="Topics", yaxis_title="Average Weight")
            st.plotly_chart(fig, use_container_width=True)
            
            # Word clouds for top topics
            st.markdown("### ☁️ Topic Word Clouds")
            
            top_topic_indices = avg_topic_weights.argsort()[::-1][:4]
            
            cols = st.columns(2)
            for i, topic_idx in enumerate(top_topic_indices):
                topic = svd_model_new.components_[topic_idx]
                top_words_idx = topic.argsort()[::-1][:50]
                top_words = [feature_names[j] for j in top_words_idx]
                topic_text = ' '.join(top_words)
                
                fig = create_wordcloud(topic_text, f"Topic {topic_idx}")
                if fig:
                    with cols[i % 2]:
                        st.pyplot(fig)
                        plt.close()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>" +
        "🏔️ The North Face Product Recommendation System | Built with Streamlit" +
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()