# 🎬 Movie Recommendation System using DBSCAN & IMDB

A content-based movie recommendation system that uses **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) to cluster movies by their content similarity, built on metadata from the **IMDb dataset**.

---

## 🎯 Project Objective

The goal of this project is to build a **recommendation system** that groups movies based on similarity using **unsupervised learning**, specifically the **DBSCAN clustering algorithm**.  
This allows recommending movies based purely on **content-based features**, not on popularity or user ratings.

---

## 📦 Dataset

- Source: IMDb Top 1000 Movies Dataset  
- Features used:
  - Title
  - Genre
  - Overview
  - Director
  - Star1, Star2, Star4
  - IMDb Rating

---

## 🧠 Techniques Used

- **Natural Language Processing (NLP)**:
  - Combine metadata into a unified `content` field
  - TF-IDF vectorization (`max_features=300`)
- **Numerical Features**:
  - IMDb Rating added to feature matrix
- **Preprocessing**:
  - Feature normalization with `StandardScaler`
  - Dimensionality reduction using `PCA(n_components=50)`
- **Clustering**:
  - DBSCAN clustering (with Euclidean distance)
- **Recommendation Logic**:
  - Use `cosine_similarity` within cluster for ranking

---

## 🚀 Features

- Input a movie title
- Recommend movies from the **same DBSCAN cluster**
- Filter out "noise" (movies that don't belong to any cluster)
- No need for user ratings or collaborative filtering

---

## ⚙️ How It Works

1. 🔄 **Combine metadata**: Genre, Overview, Director, Cast → `content`  
2. 🧠 **Vectorize**: Use `TfidfVectorizer` to convert text to vectors  
3. 📊 **Add IMDb Rating** to feature matrix  
4. 📏 **Normalize** features using `StandardScaler`  
5. 🔻 **Reduce dimensions** using `PCA`  
6. 🔍 **Cluster** with `DBSCAN (eps=10, min_samples=10)`  
7. 🎯 **Recommend**:
   - Find cluster of input movie  
   - Compute `cosine_similarity` with others in same cluster  
   - Return top-N recommendations

---
