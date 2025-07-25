import streamlit as st
import pandas as pd
import numpy as np
import re
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
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('northface_catalog.csv')
        
        # Basic preprocessing
        df = df.dropna(subset=['description'])
        df = df.reset_index(drop=True)
        
        # Clean descriptions
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        df['description'] = df['description'].apply(clean_text)
        df = df[df['description'].str.len() > 10]  # Remove very short descriptions
        df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def train_models(df):
    """Train all ML models on-the-fly"""
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(df['description'])
        
        # DBSCAN Clustering
        clustering = DBSCAN(eps=0.45, min_samples=5, metric='cosine')
        cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())
        
        # Topic Modeling with TruncatedSVD
        svd_model = TruncatedSVD(n_components=15, random_state=42)
        topic_matrix = svd_model.fit_transform(tfidf_matrix)
        
        # Cosine Similarity Matrix (for smaller datasets)
        if len(df) <= 1000:  # Only compute for smaller datasets to avoid memory issues
            similarity_matrix = cosine_similarity(tfidf_matrix)
        else:
            similarity_matrix = None
        
        # Calculate metadata
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = (cluster_labels == -1).sum() / len(cluster_labels)
        explained_variance = svd_model.explained_variance_ratio_.sum()
        
        metadata = {
            'dataset_info': {
                'n_products': len(df),
                'avg_description_length': df['description'].str.len().mean()
            },
            'tfidf_info': {
                'n_features': tfidf_matrix.shape[1],
                'vocabulary_size': len(vectorizer.vocabulary_)
            },
            'clustering_info': {
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'eps': 0.45,
                'min_samples': 5
            },
            'topic_modeling_info': {
                'n_topics': 15,
                'explained_variance_ratio': explained_variance
            }
        }
        
        return {
            'df': df,
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'cluster_labels': cluster_labels,
            'svd_model': svd_model,
            'topic_matrix': topic_matrix,
            'similarity_matrix': similarity_matrix,
            'metadata': metadata
        }
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None

def find_similar_products(product_id, df, cluster_labels, similarity_matrix=None, tfidf_matrix=None, k=5):
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
    if similarity_matrix is not None:
        similarities = similarity_matrix[product_id].copy()
        similarities[product_id] = 0  # Exclude the product itself
        similar_indices = similarities.argsort()[::-1][:k]
        return similar_indices.tolist()
    elif tfidf_matrix is not None:
        # Compute similarity on-the-fly for this product
        product_vector = tfidf_matrix[product_id]
        similarities = cosine_similarity(product_vector, tfidf_matrix).flatten()
        similarities[product_id] = 0  # Exclude the product itself
        similar_indices = similarities.argsort()[::-1][:k]
        return similar_indices.tolist()
    
    return []

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

def main():
    # Header
    st.markdown('<h1 class="main-header">🏔️ The North Face Product Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load and preprocess data
    with st.spinner("Loading dataset..."):
        df = load_and_preprocess_data()
    
    if df is None:
        st.error("Failed to load the dataset. Please ensure 'northface_catalog.csv' is in the project directory.")
        return
    
    # Train models
    with st.spinner("Training ML models... This may take a moment."):
        models = train_models(df)
    
    if models is None:
        st.error("Failed to train models.")
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
    st.sidebar.markdown(f"**Noise Ratio:** {metadata['clustering_info']['noise_ratio']:.1%}")
    
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
                    similar_products = find_similar_products(
                        product_id, df, cluster_labels, similarity_matrix, tfidf_matrix
                    )
                    
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
                        similar_products = find_similar_products(
                            product_id, df, cluster_labels, similarity_matrix, tfidf_matrix
                        )
                        
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
        
        # Display clustering results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Number of Clusters", metadata['clustering_info']['n_clusters'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            noise_ratio = metadata['clustering_info']['noise_ratio']
            st.metric("Noise Ratio", f"{noise_ratio:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cluster distribution
        st.markdown("### 📊 Cluster Distribution")
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_counts.index, 
            y=cluster_counts.values,
            title="Number of Products per Cluster",
            labels={'x': 'Cluster ID', 'y': 'Number of Products'}
        )
        fig.update_layout(xaxis_title="Cluster ID (-1 = Noise)", yaxis_title="Number of Products")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample products from each cluster
        st.markdown("### 🔍 Sample Products by Cluster")
        selected_cluster = st.selectbox(
            "Select a cluster to explore:",
            sorted(set(cluster_labels))
        )
        
        cluster_products = df[df['cluster'] == selected_cluster]
        if len(cluster_products) > 0:
            st.write(f"**Cluster {selected_cluster}** contains {len(cluster_products)} products:")
            
            # Show sample products
            sample_size = min(5, len(cluster_products))
            sample_products = cluster_products.sample(n=sample_size, random_state=42)
            
            for idx, (_, product) in enumerate(sample_products.iterrows(), 1):
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                st.write(f"**Product {idx} (ID: {product.name}):**")
                st.write(f"{product['description'][:200]}...")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Word cloud for cluster
            if st.button(f"Generate Word Cloud for Cluster {selected_cluster}"):
                cluster_text = " ".join(cluster_products['description'].tolist())
                wordcloud_fig = create_wordcloud(cluster_text, f"Cluster {selected_cluster} Word Cloud")
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
        else:
            st.warning(f"No products found in cluster {selected_cluster}.")
    
    elif page == "🎯 Topic Modeling":
        st.markdown('<h2 class="sub-header">🎯 Topic Modeling Analysis</h2>', unsafe_allow_html=True)
        
        # Display topic modeling results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Number of Topics", metadata['topic_modeling_info']['n_topics'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            explained_var = metadata['topic_modeling_info']['explained_variance_ratio']
            st.metric("Explained Variance", f"{explained_var:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Topic analysis
        st.markdown("### 📋 Topic Analysis")
        
        # Get feature names and topic components
        feature_names = vectorizer.get_feature_names_out()
        
        # Display top words for each topic
        n_top_words = 10
        
        for topic_idx in range(min(5, svd_model.n_components)):  # Show first 5 topics
            st.markdown(f"#### Topic {topic_idx + 1}")
            
            # Get top words for this topic
            top_words_idx = svd_model.components_[topic_idx].argsort()[::-1][:n_top_words]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [svd_model.components_[topic_idx][i] for i in top_words_idx]
            
            # Create a horizontal bar chart
            fig = px.bar(
                x=top_scores,
                y=top_words,
                orientation='h',
                title=f"Top {n_top_words} Words in Topic {topic_idx + 1}",
                labels={'x': 'Importance Score', 'y': 'Words'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Product-topic distribution
        st.markdown("### 📊 Product-Topic Distribution")
        
        # Find dominant topic for each product
        dominant_topics = topic_matrix.argmax(axis=1)
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        
        fig = px.bar(
            x=[f"Topic {i+1}" for i in topic_counts.index],
            y=topic_counts.values,
            title="Number of Products per Dominant Topic",
            labels={'x': 'Topic', 'y': 'Number of Products'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample products for selected topic
        st.markdown("### 🔍 Sample Products by Topic")
        selected_topic = st.selectbox(
            "Select a topic to explore:",
            range(svd_model.n_components),
            format_func=lambda x: f"Topic {x + 1}"
        )
        
        # Get products where this topic is dominant
        topic_products_mask = dominant_topics == selected_topic
        topic_products = df[topic_products_mask]
        
        if len(topic_products) > 0:
            st.write(f"**Topic {selected_topic + 1}** is dominant in {len(topic_products)} products:")
            
            # Show sample products
            sample_size = min(5, len(topic_products))
            sample_products = topic_products.sample(n=sample_size, random_state=42)
            
            for idx, (_, product) in enumerate(sample_products.iterrows(), 1):
                topic_score = topic_matrix[product.name, selected_topic]
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                st.write(f"**Product {idx} (ID: {product.name})** - Topic Score: {topic_score:.3f}")
                st.write(f"{product['description'][:200]}...")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"No products found with Topic {selected_topic + 1} as dominant.")

if __name__ == "__main__":
    main()