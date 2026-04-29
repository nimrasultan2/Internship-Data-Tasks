# Movie Recommendation System
## Recommends 5 similar movies based on a selected title using content-based filtering.

1. Dataset of 65 movies including 15 Pakistani films
2. Used TF-IDF to convert movie genres, director, and tags into numbers
3. Used cosine similarity to find the most similar movies
4. Streamlit GUI with a dropdown, recommend button, and similarity score bar chart

### Libraries Used

| Library | Purpose |
| :--- | :--- |
| **pandas** | Data handling and management |
| **numpy** | Numerical operations and dataset generation |
| **scikit-learn** | ML models, TF-IDF, and Feature Selection (RFE) |
| **pickle** | Saving and loading the trained model |
| **streamlit** | Creating the Graphical User Interface (GUI) |
