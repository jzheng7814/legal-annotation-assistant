# Legal Annotation Assistant

A Streamlit-based tool for annotating legal case documents, with semantic search capabilities for finding relevant text chunks across document collections.

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (optional but recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd legal-annotation-assistant
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

The tool expects the following files in the working directory:

- `case_summary.txt`: Plain text file containing the case summary (line breaks represent section divides)
- `enhanced_chunks.csv`: CSV file with chunked text data and page numbers
- `document_mapping.json`: JSON file mapping document IDs to file paths
- `chunk_embeddings.pkl`: Pickle file containing pre-computed embeddings

The preprocessing scripts (`preprocess.py` and `precompute_embeddings.py`) can help you generate these files from your document collection.

## Preprocessing

### Preparing Documents

1. Place your PDF documents in a `documents/` directory
2. Create a document mapping file with IDs and paths
3. Run the preprocessing script to annotate chunks with page numbers:
   ```bash
   python preprocess.py
   ```

### Computing Embeddings

After preprocessing, generate embeddings for semantic search:
```bash
python precompute_embeddings.py
```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your default web browser (typically at http://localhost:8501)

### Workflow

1. **Annotation Tab**:
   - Navigate through case summary sections
   - Add claims from each section
   - View semantically similar chunks from the document collection
   - Click on chunks to view them in the document viewer

2. **Search Tab**:
   - Search for specific text in all chunks
   - Jump directly to chunks by ID
   - View search results and open them in the document viewer

3. **Document Viewer** (right panel):
   - Browse available documents
   - Navigate through document pages
   - See chunk annotations on each page
   - Highlight specific chunks

## System Architecture

- `app.py`: Main Streamlit application
- `preprocess.py`: Augment chunk CSV with document page for app display
- `precompute_embeddings.py`: Generates embeddings for semantic search
- `chunk_suggestion_engines.py`: Implements chunk suggestion engines
