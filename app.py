import streamlit as st
import pandas as pd
import json
import numpy as np
import re  # Add import for regex functions
from sentence_transformers import SentenceTransformer
import fitz
from PIL import Image, ImageDraw
import io
import pickle
import os
from chunk_suggestion_engines import EmbeddingsSimilarityEngine

# ────────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Legal Annotation Assistant", layout="wide")

THEME_SAFE_CSS = """
<style>
div.chunk-box {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    padding: 10px;
    border-left: 3px solid #0078ff;
    font-style: italic;
    border-radius: 4px;
    line-height: 1.4;
}

/* Style for chunk navigator */
div.chunk-navigator {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

div.chunk-card {
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
    cursor: pointer;
    transition: all 0.2s ease;
}

div.chunk-card:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    background-color: var(--secondary-background-color);
    transform: translateY(-2px);
}

div.chunk-content {
    margin-bottom: 10px;
}

/* Pagination indicators */
div.pagination-dots {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin: 10px 0;
}

div.dot {
    height: 8px;
    width: 8px;
    border-radius: 50%;
    background-color: rgba(128, 128, 128, 0.3);
}

div.active-dot {
    background-color: #0078ff;
}

/* Search results styling */
div.search-result {
    margin-bottom: 10px;
    padding: 8px;
    border-radius: 4px;
    border-left: 3px solid #0078ff;
    background-color: var(--secondary-background-color);
    cursor: pointer;
}

div.search-result:hover {
    background-color: var(--secondary-background-color);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

/* Tabs styling */
div.tab-container {
    display: flex;
    border-bottom: 1px solid #ccc;
    margin-bottom: 15px;
}

div.tab {
    padding: 8px 16px;
    cursor: pointer;
    border-radius: 4px 4px 0 0;
    margin-right: 4px;
}

div.active-tab {
    background-color: var(--secondary-background-color);
    border: 1px solid #ccc;
    border-bottom: none;
}

/* PDF container with relative positioning for annotations */
div.pdf-container {
    position: relative;
    width: 100%;
}

/* Ensure stButton elements don't expand too much */
div.chunk-card div.stButton {
    display: inline-block;
    width: auto;
}
</style>
"""

st.markdown(THEME_SAFE_CSS, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# DEFAULT STATE
# ────────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "current_section_idx": 0,
    "claimed_texts": {},
    "current_doc": None,
    "current_doc_name": None,
    "current_page": 0,
    "current_chunk_text": None,
    "page_num_input": 1,
    "chunk_index": {},  # To track which chunk we're viewing for each claim
    "engine_type": "embeddings",  # Default suggestion engine
    "active_tab": "annotation",  # Active tab: annotation, search
    "search_query": "",
    "search_results": [],
    "jump_to_chunk_id": "",  # For direct chunk ID lookup
}

for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Suppress PyTorch-related warnings with Streamlit's watcher
import warnings
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer("all-mpnet-base-v2")

PERSISTENCE_FILE = "annotation_state.json"

# ────────────────────────────────────────────────────────────────────────────────
# PERSISTENCE HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def save_session_state():
    data_to_save = {
        "current_section_idx": st.session_state.current_section_idx,
        "claimed_texts": st.session_state.claimed_texts,
    }   
    with open(PERSISTENCE_FILE, "w") as fp:
        json.dump(data_to_save, fp)


def load_session_state():
    if not os.path.exists(PERSISTENCE_FILE):
        return
    try:
        with open(PERSISTENCE_FILE, "r") as fp:
            saved = json.load(fp)
        st.session_state.current_section_idx = saved.get("current_section_idx", 0)
        st.session_state.claimed_texts = {int(k): v for k, v in saved.get("claimed_texts", {}).items()}
    except Exception as exc:
        st.error(f"Failed to load saved state → {exc}")

load_session_state()

# ────────────────────────────────────────────────────────────────────────────────
# DATA LOADERS (CACHED)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    with open("case_summary.txt") as fp:
        summary = [line.strip() for line in fp]
    chunks = pd.read_csv("enhanced_chunks.csv")
    with open("document_mapping.json") as fp:
        mapping = json.load(fp)
    with open("chunk_embeddings.pkl", "rb") as fp:
        embeds = pickle.load(fp)
    return summary, chunks, mapping, embeds

# ────────────────────────────────────────────────────────────────────────────────
# SUGGESTION ENGINE FACTORY
# ────────────────────────────────────────────────────────────────────────────────

def get_suggestion_engine(engine_type, model, embeddings):
    """Factory function to create the appropriate suggestion engine."""
    if engine_type == "embeddings":
        return EmbeddingsSimilarityEngine(model, embeddings)
    # Add more engine types here as they are implemented
    else:
        # Default to embeddings engine if type is not recognized
        return EmbeddingsSimilarityEngine(model, embeddings)

# ────────────────────────────────────────────────────────────────────────────────
# PDF PROCESSING AND ANNOTATION
# ────────────────────────────────────────────────────────────────────────────────

def normalize_text(text):
    """Normalize text for better matching, copied from preprocess.py"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_text_on_page(doc_path, page_num, text):
    """Find bounding boxes for text on a specific page with improved matching."""
    try:
        doc = fitz.open(doc_path)
        page = doc[page_num]
        
        # Clean text for search
        text = text.strip()
        if not text:
            return []
        
        # Get original page text
        page_text = page.get_text()
        
        # Try direct search first (fastest and most accurate if it works)
        text_instances = page.search_for(text)
        
        # If direct search fails, try with normalized text
        if not text_instances:
            # Normalize the search text
            clean_text = normalize_text(text)
            
            # Get text words - will be used if we need to search for smaller chunks
            text_words = clean_text.split()
            
            # If text is long, try with first 10-15 words
            if len(text_words) > 15:
                search_text = ' '.join(text_words[:15])
                # Convert back to original case and format for searching
                pattern = re.compile(re.escape(search_text), re.IGNORECASE)
                for match in pattern.finditer(normalize_text(page_text)):
                    # Find the corresponding text in the original page
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Use the match position to find in original text
                    original_match_text = page_text[start_pos:end_pos + 50]  # Add some buffer
                    
                    # Now search for this in the page
                    found = page.search_for(original_match_text[:30])  # Use first 30 chars
                    if found:
                        text_instances.extend(found)
            
            # If still no results, try with first 5-7 words
            if not text_instances and len(text_words) > 7:
                search_text = ' '.join(text_words[:7])
                # Similar to above but with shorter text
                pattern = re.compile(re.escape(search_text), re.IGNORECASE)
                for match in pattern.finditer(normalize_text(page_text)):
                    start_pos = match.start()
                    original_match_text = page_text[start_pos:start_pos + 30]
                    found = page.search_for(original_match_text[:20])
                    if found:
                        text_instances.extend(found)
        
        # Get page dimensions for normalization
        page_rect = page.rect
        
        # Return normalized coordinates
        results = []
        for rect in text_instances:
            # Normalize coordinates relative to page size
            norm_rect = {
                "x0": rect.x0 / page_rect.width,
                "y0": rect.y0 / page_rect.height,
                "x1": rect.x1 / page_rect.width,
                "y1": rect.y1 / page_rect.height,
            }
            results.append(norm_rect)
        
        return results
    except Exception as e:
        st.error(f"Error finding text: {e}")
        return []

def render_pdf_page_with_annotations(path, page_num, chunks_df=None):
    """Render a PDF page with chunk numbers to the left of each chunk's first word,
    using original chunk IDs with sequential disambiguation."""
    try:
        doc = fitz.open(path)
        page = doc[page_num]
        
        # Create pixmap with higher resolution for better quality
        zoom = 1.5
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Add chunk annotations to the left of each chunk if chunks_df is provided
        if chunks_df is not None:
            # Get document ID from path
            doc_id = get_doc_id_from_path(path)
            if doc_id is not None:
                # Find chunks on this page
                page_chunks = chunks_df[
                    (chunks_df['doc_id'] == doc_id) & 
                    (chunks_df['page_num'] == page_num)
                ]
                
                # Dictionary to track positions where text has been found to handle duplicates
                found_text_positions = {}
                
                # For each chunk, find its starting position on the page
                for _, chunk in page_chunks.iterrows():
                    chunk_text = chunk['chunk']
                    chunk_id = int(chunk['index'])
                    
                    # Get the first few words of the chunk to search for
                    words = chunk_text.split()
                    first_words = ' '.join(words[:min(5, len(words))])
                    
                    # Find positions using the improved text finding function
                    positions = find_text_on_page(path, page_num, first_words)
                    
                    if positions:
                        # For text disambiguation purposes, normalize the text
                        normalized_text = normalize_text(first_words)
                        
                        # Convert to image coordinates - use the first position by default
                        x0 = positions[0]["x0"] * img_width
                        y0 = positions[0]["y0"] * img_height
                        
                        # Check if this text has been found before at a similar position
                        found_at_duplicate = False
                        for prev_text, prev_positions in found_text_positions.items():
                            if normalized_text == prev_text:
                                for prev_pos in prev_positions:
                                    # If positions are very close, consider them duplicates
                                    if abs(y0 - prev_pos['y0']) < 20:  # within 20px vertically
                                        # Skip if current chunk ID is lower than previous one
                                        if chunk_id < prev_pos['chunk_id']:
                                            found_at_duplicate = True
                                            break
                        
                        # Skip if this is a duplicate and should be disambiguated
                        if found_at_duplicate:
                            continue
                        
                        # Record this position for future disambiguation
                        if normalized_text not in found_text_positions:
                            found_text_positions[normalized_text] = []
                        found_text_positions[normalized_text].append({
                            'chunk_id': chunk_id,
                            'x0': x0,
                            'y0': y0
                        })
                        
                        # Calculate position to the left of the first word
                        label_x = max(5, x0 - 20)  # 20 pixels to the left, but minimum 5px from edge
                        label_y = y0  # Same vertical position as text
                        
                        # Scale font size based on page height, but keep it small
                        font_size = max(9, int(img_height / 100))  # Minimum size of 9px
                        
                        try:
                            # Try to use a font that will be readable at small sizes
                            from PIL import ImageFont
                            try:
                                font = ImageFont.truetype("Arial", font_size)
                            except IOError:
                                font = ImageFont.load_default()
                            
                            # Draw a small background for better readability
                            text = f"#{chunk_id}"
                            
                            # In newer PIL versions, use this:
                            if hasattr(font, "getbbox"):
                                text_width, text_height = font.getbbox(text)[2:4]
                            else:
                                # In older PIL versions, fall back to textsize if available
                                try:
                                    text_width, text_height = draw.textsize(text, font=font)
                                except:
                                    # If all fails, approximate
                                    text_width, text_height = len(text) * font_size * 0.6, font_size * 1.2
                            
                            # Draw background rectangle (white with transparency)
                            draw.rectangle(
                                [label_x - 1, label_y - 1, label_x + text_width + 1, label_y + text_height + 1],
                                fill=(255, 255, 255, 180)
                            )
                            
                            # Draw the chunk ID text
                            draw.text(
                                (label_x, label_y),
                                text,
                                fill=(0, 0, 255, 230),  # Blue, more visible
                                font=font
                            )
                        except Exception as font_error:
                            # Fallback for font issues
                            draw.text(
                                (label_x, label_y),
                                text,
                                fill=(0, 0, 255, 230)  # Blue with high opacity
                            )
        
        return img
    except Exception as e:
        st.error(f"PDF render error → {e}")
        return None

def find_text_chunks_on_page(doc_path, page_num, chunks_df):
    """Find chunks that appear on the specified page."""
    doc_id = get_doc_id_from_path(doc_path)
    if doc_id is None:
        return pd.DataFrame()
        
    matching_chunks = chunks_df[
        (chunks_df['doc_id'] == doc_id) & 
        (chunks_df['page_num'] == page_num)
    ]
    return matching_chunks

def get_doc_id_from_path(doc_path):
    """Extract document ID from path."""
    # Assuming path format like "documents/12345.pdf"
    try:
        filename = os.path.basename(doc_path)
        doc_id = int(os.path.splitext(filename)[0])
        return doc_id
    except Exception:
        # Return a fallback value if extraction fails
        return None

# ────────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def find_relevant_chunks(text, chunks_df, chunk_embeds, mapping, k=20):
    """Legacy wrapper function for backward compatibility."""
    engine = get_suggestion_engine(st.session_state.engine_type, st.session_state.model, chunk_embeds)
    return engine.find_relevant_chunks(text, chunks_df, mapping, k)

def open_document(record):
    """Open a document and reset view state."""
    st.session_state.current_doc = record["path"]
    st.session_state.current_doc_name = record["name"]
    st.session_state.current_page = 0
    st.session_state.page_num_input = 1
    st.session_state.current_chunk_text = None
    st.session_state.highlight_chunks = []  # Clear highlights

def jump_to_chunk(chunk_id, chunks_df, mapping):
    """Jump to a specific chunk by ID."""
    try:
        # Convert to int if it's a string
        chunk_id = int(chunk_id)
        
        # Find the chunk in the dataframe
        chunk = chunks_df[chunks_df['index'] == chunk_id]
        
        if not chunk.empty:
            row = chunk.iloc[0]
            doc_id = int(row['doc_id'])
            page_num = int(row['page_num'])
            chunk_text = row['chunk']
            
            # Find the document record
            doc_record = next((doc for doc in mapping if doc['id'] == doc_id), None)
            
            if doc_record:
                # Open the document
                open_document(doc_record)
                
                # Set page and chunk text
                st.session_state.current_page = page_num
                st.session_state.page_num_input = page_num + 1
                st.session_state.current_chunk_text = chunk_text
                
                return True
            
        return False
    except Exception as e:
        st.error(f"Error jumping to chunk: {e}")
        return False

def render_chunk_navigator(chunks, claim_index, mapping, chunks_df):
    """Render a chunk at a time with navigation controls and auto-jump."""
    if not chunks:
        st.info("No relevant chunks found.")
        return
    
    # Initialize chunk index for this claim if not already done
    if claim_index not in st.session_state.chunk_index:
        st.session_state.chunk_index[claim_index] = 0
    
    curr_idx = st.session_state.chunk_index[claim_index]
    
    # Ensure index is valid (might be out of range if chunks list changed)
    if curr_idx >= len(chunks):
        st.session_state.chunk_index[claim_index] = 0
        curr_idx = 0
    
    # Get current chunk
    ch = chunks[curr_idx]
    
    # Auto-jump to the document when the chunk index changes
    # This ensures we're always looking at the right document
    if hasattr(st.session_state, 'last_viewed_chunk') and st.session_state.last_viewed_chunk.get(claim_index) != ch['index']:
        jump_to_chunk(ch['index'], chunks_df, mapping)
    
    # Store current chunk index for comparison next time
    if not hasattr(st.session_state, 'last_viewed_chunk'):
        st.session_state.last_viewed_chunk = {}
    st.session_state.last_viewed_chunk[claim_index] = ch['index']
    
    # Create a card for the current chunk
    with st.container():
        # Display chunk header
        st.markdown(f"**Chunk #{ch['index']} · Doc {ch['doc_id']} ({ch['doc_name']}) · Page {ch['page_num'] + 1} · Similarity {ch['similarity']:.3f}**")
        
        # Display chunk content
        st.markdown(f"<div class='chunk-box'>{ch['chunk']}</div>", unsafe_allow_html=True)
        
        # Navigation controls
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Previous button
        if col1.button("← Previous", key=f"prev_{claim_index}", disabled=(curr_idx == 0)):
            st.session_state.chunk_index[claim_index] = max(0, curr_idx - 1)
            st.rerun()
        
        # Pagination indicator
        pagination_html = "<div class='pagination-dots'>"
        for i in range(len(chunks)):
            if i == curr_idx:
                pagination_html += "<div class='dot active-dot'></div>"
            else:
                pagination_html += "<div class='dot'></div>"
        pagination_html += "</div>"
        col2.markdown(pagination_html, unsafe_allow_html=True)
        col2.markdown(f"<div style='text-align: center;'>{curr_idx + 1} of {len(chunks)}</div>", unsafe_allow_html=True)
        
        # Next button
        if col3.button("Next →", key=f"next_{claim_index}", disabled=(curr_idx == len(chunks) - 1)):
            st.session_state.chunk_index[claim_index] = min(len(chunks) - 1, curr_idx + 1)
            st.rerun()

def search_chunks(query, chunks_df, mapping, limit=50):
    """Search for chunks containing the given text."""
    if not query:
        return []
    
    # Simple case-insensitive substring search
    mask = chunks_df['chunk'].str.contains(query, case=False, na=False)
    results = chunks_df[mask].head(limit)
    
    # Format results for display
    formatted_results = []
    for _, row in results.iterrows():
        doc_id = int(row['doc_id'])
        doc_name = next((d['name'] for d in mapping if d['id'] == doc_id), "Unknown")
        formatted_results.append({
            'index': int(row['index']),
            'chunk': row['chunk'],
            'doc_id': doc_id,
            'doc_name': doc_name,
            'page_num': int(row['page_num']),
        })
    
    return formatted_results

# ────────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ────────────────────────────────────────────────────────────────────────────────

def cb_prev_section():
    if st.session_state.current_section_idx > 0:
        st.session_state.current_section_idx -= 1
        save_session_state()

def cb_next_section(total):
    if st.session_state.current_section_idx < total - 1:
        st.session_state.current_section_idx += 1
        save_session_state()

def cb_select_doc():
    # Get the selected document and open it
    rec = next(r for r in st.session_state.doc_mapping if r['name'] == st.session_state.selected_doc_name)
    open_document(rec)

def cb_page_input_change():
    # User typed a page number → sync current_page
    page = st.session_state.page_num_input - 1
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.session_state.highlight_chunks = []  # Clear highlights when changing page

def cb_search_chunks():
    # Perform chunk search
    query = st.session_state.search_query
    if query:
        results = search_chunks(query, st.session_state.chunks_df, st.session_state.doc_mapping)
        st.session_state.search_results = results
    else:
        st.session_state.search_results = []

def cb_jump_to_chunk():
    # Jump to specific chunk by ID
    chunk_id = st.session_state.jump_to_chunk_id
    if chunk_id:
        success = jump_to_chunk(chunk_id, st.session_state.chunks_df, st.session_state.doc_mapping)
        if not success:
            st.error(f"Chunk #{chunk_id} not found")

def cb_search_result_click(result):
    # Handle search result click
    jump_to_chunk(result['index'], st.session_state.chunks_df, st.session_state.doc_mapping)

def cb_set_active_tab(tab):
    st.session_state.active_tab = tab

# ────────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ────────────────────────────────────────────────────────────────────────────────

def render_tab_bar():
    """Render a tab bar for switching between app sections."""
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📝 Annotation", 
                    use_container_width=True,
                    type="primary" if st.session_state.active_tab == "annotation" else "secondary"):
            st.session_state.active_tab = "annotation"
            st.rerun()
    
    with col2:
        if st.button("🔍 Chunk Search", 
                    use_container_width=True,
                    type="primary" if st.session_state.active_tab == "search" else "secondary"):
            st.session_state.active_tab = "search"
            st.rerun()
            
    st.markdown("---")

def render_annotation_tab(summary, chunks_df, mapping, embeds):
    """Render the annotation tab."""
    st.subheader("Annotation")

    nav_prev, nav_mid, nav_next = st.columns([1, 2, 1])
    nav_prev.button("← Previous", on_click=cb_prev_section)
    total = len(summary)
    nav_mid.markdown(f"### Section {st.session_state.current_section_idx + 1} of {total}")
    nav_next.button("Next →", on_click=cb_next_section, args=(total,))
    st.progress(st.session_state.current_section_idx / max(1, total - 1))

    sect = summary[st.session_state.current_section_idx]
    st.text_area("Current section", sect, height=100)

    with st.form(key=f"cl_form_{st.session_state.current_section_idx}"):
        cl = st.text_input("Add a claim from this section", key="claim_input")
        if st.form_submit_button("Add claim") and cl:
            st.session_state.claimed_texts.setdefault(st.session_state.current_section_idx, [])
            if cl not in st.session_state.claimed_texts[st.session_state.current_section_idx]:
                st.session_state.claimed_texts[st.session_state.current_section_idx].append(cl)
                save_session_state()

    claims = st.session_state.claimed_texts.get(st.session_state.current_section_idx, [])
    for i, text in enumerate(claims):
        with st.expander(f"🔍 Claim {i + 1}: {text}", expanded=False):
            cols = st.columns([5, 1])
            cols[0].info(text)
            if cols[1].button("❌", key=f"del_{i}"):
                claims.pop(i)
                save_session_state()
                st.rerun()

            # Create the suggestion engine
            engine = get_suggestion_engine(st.session_state.engine_type, st.session_state.model, embeds)
            rels = engine.find_relevant_chunks(text, chunks_df, mapping)
            st.markdown("#### Suggested chunks")
            
            # Render the chunk navigator component
            render_chunk_navigator(rels, i, mapping, chunks_df)

def render_search_tab(chunks_df, mapping):
    """Render the chunk search tab."""
    st.subheader("Chunk Search")
    
    # Text search
    with st.form(key="search_form"):
        st.text_input("Search for text in chunks:", key="search_query")
        st.form_submit_button("Search", on_click=cb_search_chunks)
    
    # Direct chunk ID lookup
    with st.form(key="chunk_id_form"):
        st.number_input("Jump to chunk by ID:", min_value=0, step=1, key="jump_to_chunk_id")
        st.form_submit_button("Jump to Chunk", on_click=cb_jump_to_chunk)
    
    # Display search results
    if st.session_state.search_results:
        st.markdown(f"### Found {len(st.session_state.search_results)} results")
        
        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                st.markdown(f"**Chunk #{result['index']} · {result['doc_name']} · Page {result['page_num'] + 1}**")
                st.markdown(f"<div class='chunk-box'>{result['chunk'][:200]}{'...' if len(result['chunk']) > 200 else ''}</div>", unsafe_allow_html=True)
                
                if st.button("View in Document", key=f"view_result_{i}"):
                    cb_search_result_click(result)
                    st.rerun()
                
                st.markdown("---")

def render_document_viewer(chunks_df, mapping):
    """Render the document viewer panel."""
    st.subheader("Document viewer")

    with st.expander("📚 Document browser", expanded=False):
        names = [d["name"] for d in mapping]
        
        # If no document is currently selected, select the first one
        if not st.session_state.current_doc_name:
            default_doc = mapping[0]
            st.session_state.current_doc = default_doc["path"]
            st.session_state.current_doc_name = default_doc["name"]
            st.session_state.selected_doc_name = default_doc["name"]
        
        # Get the index of the currently selected document
        idx = names.index(st.session_state.current_doc_name)
        
        # Display radio buttons for selection
        selected = st.radio(
            "Select a document", 
            names, 
            index=idx, 
            key="selected_doc_name", 
            on_change=cb_select_doc
        )

    if st.session_state.current_doc:
        st.markdown(f"**{st.session_state.current_doc_name}**")
        
        # Show the current chunk being looked for
        if st.session_state.current_chunk_text:
            st.markdown("**Looking for:**")
            st.markdown(f"<div class='chunk-box'>{st.session_state.current_chunk_text}</div>", unsafe_allow_html=True)

        try:
            pdf = fitz.open(st.session_state.current_doc)
            total_pages = len(pdf)

            col_a, col_b, col_c = st.columns([1, 2, 1])

            prev_clicked = col_a.button("← Prev page")
            next_clicked = col_c.button("Next page →")

            if prev_clicked and st.session_state.current_page > 0:
                st.session_state.current_page -= 1
                st.session_state.highlight_chunks = []  # Clear highlights
            if next_clicked and st.session_state.current_page < total_pages - 1:
                st.session_state.current_page += 1
                st.session_state.highlight_chunks = []  # Clear highlights

            # Always keep numeric widget in sync with current_page
            st.session_state.page_num_input = st.session_state.current_page + 1

            col_b.number_input(
                "Page #",
                min_value=1,
                max_value=total_pages,
                key="page_num_input",
                step=1,
                on_change=cb_page_input_change,
            )

            # Find highlights for the current chunk text if available
            if st.session_state.current_chunk_text and not st.session_state.highlight_chunks:
                highlights = find_text_on_page(
                    st.session_state.current_doc,
                    st.session_state.current_page,
                    st.session_state.current_chunk_text
                )
                st.session_state.highlight_chunks = highlights

            # Render the page with chunk annotations only
            img = render_pdf_page_with_annotations(
                st.session_state.current_doc,
                st.session_state.current_page,
                chunks_df
            )
            
            if img:
                st.image(img, use_container_width=True)
                
                # Show which chunks are on this page
                page_chunks = find_text_chunks_on_page(
                    st.session_state.current_doc,
                    st.session_state.current_page,
                    chunks_df
                )
        
        except Exception as e:
            st.error(f"Error loading document: {e}")
    else:
        st.info("Select a document from the browser to start reading.")

# ────────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ────────────────────────────────────────────────────────────────────────────────

def main():
    summary, chunks_df, mapping, embeds = load_data()
    
    # Store data in session state for callbacks
    st.session_state.doc_mapping = mapping
    st.session_state.chunks_df = chunks_df
    
    # Render tabs
    render_tab_bar()
    
    # Split the layout
    left, right = st.columns([1, 1])
    
    # ───────────────── LEFT SIDE ─────────────────
    with left:
        if st.session_state.active_tab == "annotation":
            render_annotation_tab(summary, chunks_df, mapping, embeds)
        elif st.session_state.active_tab == "search":
            render_search_tab(chunks_df, mapping)
    
    # ───────────────── RIGHT SIDE ─────────────────
    with right:
        render_document_viewer(chunks_df, mapping)


if __name__ == "__main__":
    main()