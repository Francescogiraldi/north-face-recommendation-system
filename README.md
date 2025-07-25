# 🏔️ The North Face Product Recommendation System

A comprehensive Streamlit application that integrates machine learning models for product clustering, recommendation system, and topic modeling based on The North Face product catalog.

## 🚀 Features

### 📊 Data Overview
- Dataset statistics and metrics
- Sample product display
- Description length distribution analysis

### 🔍 Product Search & Recommendations
- **Product ID Search**: Find products by their unique ID
- **Keyword Search**: Search products using natural language keywords
- **Smart Recommendations**: Get similar products using:
  - DBSCAN clustering for products in the same cluster
  - Cosine similarity for outlier products
- **Fuzzy Matching**: Intelligent search with typo tolerance

### 📈 Clustering Analysis
- **DBSCAN Clustering**: Group similar products based on description similarity
- **Interactive Parameters**: Adjust epsilon and min_samples in real-time
- **Cluster Visualization**: Bar charts showing product distribution
- **Word Clouds**: Visual representation of dominant words in each cluster
- **Clustering Metrics**: Number of clusters, noise points, and percentages

### 🎯 Topic Modeling
- **TruncatedSVD (LSA)**: Extract latent topics from product descriptions
- **Configurable Topics**: Choose the number of topics to extract
- **Topic Analysis**: View top words for each discovered topic
- **Topic Distribution**: Visualize topic weights across the catalog
- **Topic Word Clouds**: Visual representation of each topic's vocabulary

## 🛠️ Technical Implementation

### Machine Learning Models Explained

#### 🔤 **TF-IDF Vectorization**
**What it does**: Converts product descriptions (text) into numbers that computers can understand.
- **How it works**: Analyzes which words are important in each product description
- **Example**: "waterproof jacket" becomes a numerical vector highlighting these key terms
- **Why it's useful**: Allows mathematical comparison between product descriptions

#### 📊 **DBSCAN Clustering** 
**What it does**: Automatically groups similar products together without knowing how many groups to make.
- **How it works**: Finds products with similar descriptions and puts them in the same cluster
- **Example**: All winter jackets end up in one group, hiking boots in another
- **Parameters**:
  - `eps` (0.45): How close products need to be to group together
  - `min_samples` (5): Minimum products needed to form a group
- **Why it's special**: Can identify "noise" products that don't fit any group

#### 🎯 **TruncatedSVD (Topic Modeling)**
**What it does**: Discovers hidden themes or topics across all product descriptions.
- **How it works**: Finds patterns of words that often appear together
- **Example**: Might discover topics like "Winter Gear", "Hiking Equipment", "Casual Wear"
- **Output**: Shows the most important words for each discovered topic
- **Why it's useful**: Helps understand the main categories in your product catalog

#### 🔍 **Cosine Similarity**
**What it does**: Measures how similar two products are based on their descriptions.
- **How it works**: Calculates the angle between product vectors (closer angle = more similar)
- **Example**: Two hiking backpacks will have high similarity (small angle)
- **When it's used**: For products that don't fit into any cluster ("noise" products)
- **Result**: Provides backup recommendations when clustering doesn't help

### 🚀 **How These Models Work Together**

1. **Text Processing**: Product descriptions → TF-IDF vectors
2. **Grouping**: DBSCAN finds natural product clusters
3. **Recommendations**: 
   - If product is in a cluster → recommend from same cluster
   - If product is "noise" → use cosine similarity
4. **Topic Discovery**: TruncatedSVD reveals hidden themes
5. **Visualization**: Word clouds show what makes each group unique

### UI/UX Features
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Navigation**: Sidebar-based section selection
- **Real-time Processing**: Cached computations for fast response
- **Visual Analytics**: Charts, graphs, and word clouds
- **Professional Styling**: Custom CSS for modern appearance
- **Error Handling**: Graceful error messages and fallbacks

## 📋 Prerequisites

- Python 3.8 or higher
- The North Face catalog data (`northface_catalog.csv`)

## 🔧 Installation

1. **Clone or download the project files**:
   ```bash
   # Ensure you have these files:
   # - app.py
   # - requirements.txt
   # - northface_catalog.csv
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if needed):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

## 🚀 Running the Application

1. **Navigate to the project directory**:
   ```bash
   cd PROJET_NORTH_FACE
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to:
   ```
   http://localhost:8501
   ```

## 📖 Usage Guide

### Running the Application
```bash
# First time setup - train the models
python train_models.py

# Then run the Streamlit app
streamlit run app.py
```

### 📱 **Application Sections Explained**

#### 📊 **1. Data Overview**
**What you'll see**: Basic statistics about your product catalog
- **Total Products**: How many products are in the dataset
- **Average Description Length**: How detailed product descriptions are
- **TF-IDF Features**: Number of unique word patterns found
- **Sample Products**: Preview of actual product data
- **Description Length Chart**: Visual distribution of description lengths

**Why it's useful**: Understand your data quality and coverage before diving into analysis.

#### 🔍 **2. Product Search & Recommendations**
**What you can do**: Find products and get personalized recommendations

**Search Methods**:
- **By Product ID**: Enter a number (0-499) to find a specific product
- **By Keyword**: Type words like "jacket", "waterproof", "hiking" to find matching products

**What you'll get**:
- **Product Details**: Full description of the selected product
- **Cluster Info**: Which group this product belongs to (or if it's unique)
- **5 Recommendations**: Similar products you might also like
- **Recommendation Logic**: Explanation of why these products were suggested

**Pro Tip**: Try searching for "waterproof", "fleece", or "hiking" to see different product categories!

#### 📈 **3. Clustering Analysis** 
**What you can do**: Experiment with how products are grouped together

**Interactive Controls**:
- **Eps Slider (0.1-1.0)**: How strict the grouping is
  - Lower = stricter groups, more noise
  - Higher = looser groups, less noise
- **Min Samples (2-10)**: Minimum products needed to form a group

**What you'll see**:
- **Cluster Statistics**: Number of groups found and noise percentage
- **Cluster Distribution Chart**: Size of each product group
- **Word Clouds**: Visual representation of what makes each cluster unique

**Try This**: Start with eps=0.3 for strict grouping, then try eps=0.7 for looser grouping!

#### 🎯 **4. Topic Modeling**
**What you can do**: Discover hidden themes across all products

**Interactive Controls**:
- **Number of Topics (5-25)**: How many themes to discover

**What you'll see**:
- **Top Words per Topic**: The most important words defining each theme
- **Topic Weights Chart**: How prominent each topic is across all products
- **Topic Word Clouds**: Visual representation of each theme

**Example Topics You Might Find**:
- Topic 1: "jacket, waterproof, breathable, rain" (Rain Gear)
- Topic 2: "hiking, boot, trail, outdoor" (Hiking Equipment)
- Topic 3: "fleece, warm, insulation, cold" (Insulation Layer)

### 🎛️ **Interactive Features**
- **Real-time Analysis**: Changes update immediately as you adjust parameters
- **Visual Feedback**: Charts and word clouds help you understand the data
- **Export Ready**: All visualizations can be saved or shared
- **Mobile Friendly**: Works on tablets and phones too

### 🚀 **Getting Started - Step by Step**

1. **📊 Start with Data Overview**: Get familiar with your product catalog
2. **🔍 Try Product Search**: Search for "jacket" or "hiking" to see recommendations in action
3. **📈 Experiment with Clustering**: Try eps=0.3 vs eps=0.7 to see the difference
4. **🎯 Explore Topics**: Start with 10 topics, then try 20 to see more detailed themes

### 💡 **Pro Tips for Best Results**

#### **For Clustering Analysis**:
- **Start Conservative**: Begin with eps=0.3, min_samples=5
- **Noise is Normal**: 60-80% noise is typical for diverse product catalogs
- **Sweet Spot**: eps=0.4-0.5 usually gives good balance of clusters vs noise

#### **For Topic Modeling**:
- **Start Small**: Begin with 10 topics to see broad themes
- **Go Detailed**: Try 15-20 topics for more specific categories
- **Watch the Words**: Good topics have coherent, related words

#### **For Product Search**:
- **Try Different Keywords**: "waterproof", "insulated", "lightweight", "durable"
- **Explore Product IDs**: Random numbers like 42, 156, 299 to discover new products
- **Check Cluster Info**: See if recommended products make sense based on the cluster

## 🎯 Key Insights & Model Performance

### 🎯 **Current Model Results**
- **Products Analyzed**: 500 North Face products
- **Features Extracted**: 5,000 TF-IDF features from descriptions
- **Clusters Found**: 15 distinct product groups
- **Noise Level**: 69.8% (products that don't fit standard categories)
- **Topics Discovered**: 15 main themes with 26.2% variance explained

### 🔍 **What These Numbers Mean**

#### **15 Clusters Found**
- The algorithm automatically discovered 15 natural groupings in the product catalog
- Examples might include: "Winter Jackets", "Hiking Boots", "Base Layers", etc.
- Each cluster represents products with very similar descriptions

#### **69.8% Noise Level**
- This means about 70% of products are unique and don't fit into standard groups
- **Why this happens**: North Face has many specialized, unique products
- **Is this bad?** No! It shows product diversity and innovation
- **How we handle it**: Use cosine similarity for personalized recommendations

#### **26.2% Variance Explained (Topics)**
- Topic modeling captured about 26% of the variation in product descriptions
- **What this means**: The 15 topics represent the main themes reasonably well
- **Room for improvement**: Could increase topics for more detailed analysis

### 🚀 **Recommendation Quality**
- **Hybrid Approach**: Combines clustering + similarity for best results
- **Fast Performance**: Pre-computed similarity matrix for instant recommendations
- **Consistent Results**: Same recommendations across app sessions
- **Scalable**: Easy to retrain with new products

The application reveals several interesting patterns in The North Face catalog:

1. **Product Categories**: Natural clustering around outdoor activities
2. **Seasonal Patterns**: Different clusters for summer/winter gear
3. **Technical Features**: Clusters based on fabric technology and features
4. **Activity-Based**: Groupings around climbing, hiking, casual wear

## 🔍 Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **CSV file not found**:
   - Ensure `northface_catalog.csv` is in the same directory as `app.py`

3. **Slow performance**:
   - The app uses caching for better performance
   - First run may be slower due to model training

4. **Memory issues**:
   - Reduce the number of TF-IDF features in the code
   - Use a smaller subset of the data for testing

## 🛡️ Performance Optimizations

- **Streamlit Caching**: All expensive computations are cached
- **Efficient Algorithms**: Optimized clustering and similarity calculations
- **Lazy Loading**: Models are trained only when needed
- **Memory Management**: Efficient data structures and processing

## 🔮 Future Enhancements

- **Advanced Filtering**: Price, category, and feature-based filters
- **User Profiles**: Personalized recommendations
- **A/B Testing**: Compare different recommendation algorithms
- **Real-time Updates**: Live catalog integration
- **Export Features**: Save recommendations and analysis results

## 📊 Technical Specifications

- **Framework**: Streamlit 1.28+
- **ML Libraries**: scikit-learn, NLTK
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Data Processing**: Pandas, NumPy
- **Styling**: Custom CSS with responsive design

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements!

---

**Built with ❤️ using Streamlit and scikit-learn**