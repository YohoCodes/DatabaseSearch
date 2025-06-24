**SearchEngine**

A fast, flexible semantic search engine for large tabular datasets, built with Streamlit and Sentence Transformers.

**Features**

Semantic Search: Find the most relevant rows in your dataset using natural language queries, powered by transformer-based embeddings.
Customizable Columns: Choose which columns to search by and which columns to return as results.
Large Dataset Support: Efficiently handles large CSVs (e.g., 100,000+ rows) with vectorized search.
Streamlit Web App: User-friendly web interface for interactive searching and result display.
Caching: Uses Streamlit's resource caching to avoid redundant computation and speed up repeated searches.

**Usage**

1. Prepare your data: Place your CSV file(s) in the project directory.
2. Configure columns: Edit main.py to set which columns to search and return.
3. Run the app with terminal:
streamlit run main.py
4. Search: Enter a query in the web interface and view the most relevant results.

*Example*

Search Amazon product data by title and return bullet points, description, and product length.
Search Seth Godin blog posts by title and return content and metadata.

**Requirements**
Python 3.8+
pandas
numpy
streamlit
sentence-transformers
License
MIT License

*Note:*
This project is ideal for anyone who wants to add semantic search to their tabular data with minimal setup and a modern web UI.