# SearchEngine

A fast, flexible semantic search engine for large tabular datasets, built with Streamlit and Sentence Transformers.

## Features

- **Semantic Search**: Find the most relevant rows in your dataset using natural language queries, powered by transformer-based embeddings.
- **Customizable Columns**: Choose which columns to search by and which columns to return as results.
- **Large Dataset Support**: Efficiently handles large CSVs (e.g., 100,000+ rows) with vectorized search.
- **Streamlit Web App**: User-friendly web interface for interactive searching and result display.
- **Caching**: Uses Streamlit's resource caching to avoid redundant computation and speed up repeated searches.
- **Embedding Cache**: Optionally caches embeddings to disk for faster startup.
- **Configurable**: Uses YAML configuration files for easy customization.
- **Error Handling**: Robust error handling and logging.
- **Type Hints**: Comprehensive type annotations for better code quality.

## Project Structure

```
SearchEngine/
├── .cache/               # Cached embeddings (created at runtime)
├── logs/                 # Log files (created at runtime)
├── operations/           # Core search functionality
│   ├── __init__.py
│   └── search.py         # Search engine implementation
├── tests/                # Unit tests
│   ├── __init__.py
│   └── test_search.py    # Tests for Search class
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── config.py         # Configuration handling
│   └── logging.py        # Logging setup
├── .gitignore            # Git ignore file
├── config.yaml           # Application configuration
├── main.py               # Streamlit web interface
├── README.md             # Project documentation
├── requirements.txt      # Project dependencies
└── run_tests.py          # Script to run tests
```

## Usage

1. Prepare your data:

   - Place your CSV file(s) in the project directory.
   - Update the `config.yaml` file with your dataset information.

2. Initialize your environment and install dependencies:

   ```bash
   $ python -m venv .venv
   $ source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   $ pip install --upgrade pip
   $ pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   $ streamlit run main.py
   ```

4. Search:

   - Enter a natural language query in the web interface and view the most relevant results.
   - Use the sidebar to adjust the number of results or select a different dataset.

## Configuration

The application is configured through the `config.yaml` file:

```yaml
search_engine:
  type: "Amazon"  # Default dataset
  by_column: "TITLE"  # Column to search by
  num_results: 5  # Number of results to show

datasets:
  amazon:
    file: "Amazon.csv"
    columns: [...]
    default_result_columns: [...]
  
  sethgodin:
    file: "SethGodin.csv"
    columns: [...]
    default_result_columns: [...]

model:
  name: "all-MiniLM-L6-v2"  # Sentence transformer model
  cache_embeddings: true  # Whether to cache embeddings
  cache_dir: ".cache"  # Cache directory
```

## Testing

Run the tests using:

```bash
$ python run_tests.py
```

Or with pytest:

```bash
$ pytest
```

### Example Use Cases

- Search Amazon product data by title and return bullet points, description, and product length.
- Search Seth Godin blog posts by title and return content and metadata.

## Requirements

- Python 3.8+
- pandas
- numpy
- streamlit
- sentence-transformers
- pyyaml

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License.

---

> **Note:** This project is ideal for anyone who wants to add semantic search to their tabular data with minimal setup and a modern web UI.
