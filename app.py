"""
╔══════════════════════════════════════════════════════════════════════╗
║         MOVIE RECOMMENDATION SYSTEM - STREAMLIT APP                  ║
║      Cross-Domain Recommendation: Books → Movies                     ║
╚══════════════════════════════════════════════════════════════════════╝

Run: streamlit run ui_version_07.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import hashlib
from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Dict
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

# ── Groq Whisper API (fast, free, no heavy local model) ─────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def get_groq_api_key():
    # 1. Try environment variable
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    # 2. Try api_key.txt file
    if os.path.exists("api_key.txt"):
        try:
            with open("api_key.txt", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if "gsk_" in content:
                    if "=" in content:
                        return content.split("=")[-1].strip()
                    return content
        except Exception:
            pass
    return ""

GROQ_API_KEY = get_groq_api_key()

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .rating-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

class BranchBModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, shared_dim=128, dropout=0.4):
        super().__init__()
        
        self.user_shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        self.user_specific_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        self.user_decoder = nn.Sequential(
            nn.Linear(shared_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(shared_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_vector, item_vector):
        user_shared = self.user_shared_encoder(user_vector)
        user_specific = self.user_specific_encoder(user_vector)
        
        user_combined = torch.cat([user_shared, user_specific], dim=1)
        user_reconstructed = self.user_decoder(user_combined)
        
        item_features = self.item_encoder(item_vector)
        
        combined_features = torch.cat([user_shared, item_features], dim=1)
        rating_raw = self.predictor(combined_features).squeeze(-1)
        
        rating_pred = rating_raw * 4.0 + 1.0
        
        return rating_pred, user_shared, user_specific, user_reconstructed

# ══════════════════════════════════════════════════════════════════════
# LOAD MODELS AND DATA
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_bert_model():
    """Load DistilBERT for encoding reviews"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    bert_model.eval()
    return tokenizer, bert_model

@st.cache_resource
def load_recommendation_model():
    """Load trained Branch B model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BranchBModel(
        input_dim=768,
        hidden_dim=256,
        shared_dim=128,
        dropout=0.4
    ).to(device)
    
    try:
        checkpoint = torch.load('best_branch_b_v2.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device, checkpoint
    except FileNotFoundError:
        st.error("⚠️ Model file 'best_branch_b_v2.pt' not found!")
        return None, device, None

@st.cache_resource
def load_movie_embeddings():
    """Load pre-computed movie embeddings"""
    try:
        with open('data/embeddings/test_item_embeddings.pkl', 'rb') as f:
            movie_embeddings = pickle.load(f)
        
        # Load movie metadata if available
        try:
            movie_metadata = pd.read_pickle('data/aggregated/test_pairs.pkl')
            return movie_embeddings, movie_metadata
        except:
            return movie_embeddings, None
            
    except FileNotFoundError:
        st.error("⚠️ Movie embeddings not found at 'data/embeddings/test_item_embeddings.pkl'")
        return None, None

@st.cache_resource
def load_movie_titles():
    """Load movie titles mapping from ASIN to Title"""
    titles = {}
    try:
        df = pd.read_pickle('data/preprocessed/test_movies_preprocessed.pkl')
        titles = dict(zip(df['asin'], df['title']))
    except Exception as e:
        st.warning(f"⚠️ Could not load movie titles from 'data/preprocessed/test_movies_preprocessed.pkl': {e}")
        
    cache_path = 'data/processed/asin_to_movie_title.json'
    if os.path.exists(cache_path):
        try:
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                titles.update(cache)
        except Exception:
            pass
            
    return titles

@st.cache_resource
def load_groq_client():
    """Initialize Groq client"""
    if not GROQ_AVAILABLE:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        st.warning(f"⚠️ Could not load Groq client: {e}")
        return None

def transcribe_audio(audio_bytes: bytes, client) -> str:
    """
    Transcribe audio using Groq Whisper API.
    Extremely fast, free, no local model needed.
    """
    import subprocess
    import shutil

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw_tmp:
        raw_tmp.write(audio_bytes)
        raw_path = raw_tmp.name

    resampled_path = raw_path + "_16k.wav"

    try:
        # Convert to 16kHz mono WAV using ffmpeg if available
        ffmpeg_cmd = shutil.which("ffmpeg")
        if ffmpeg_cmd:
            try:
                subprocess.run(
                    [ffmpeg_cmd, "-y", "-i", raw_path,
                     "-ar", "16000", "-ac", "1",
                     "-acodec", "pcm_s16le",
                     "-f", "wav", resampled_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                upload_path = resampled_path
            except Exception as e:
                # Fallback to raw path if ffmpeg errors out
                upload_path = raw_path
        else:
            # Fallback to raw path directly if ffmpeg is missing
            upload_path = raw_path

        # Send to Groq Whisper API
        with open(upload_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read()),
                model="whisper-large-v3",
                language="en",
                response_format="text"
            )

        text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        return text

    finally:
        for path in [raw_path, resampled_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass

# ══════════════════════════════════════════════════════════════════════
# ENCODING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def encode_reviews(reviews: List[str], tokenizer, bert_model) -> torch.Tensor:
    """
    Convert text reviews to embeddings using DistilBERT
    """
    embeddings = []
    
    with torch.no_grad():
        for review in reviews:
            # Tokenize
            inputs = tokenizer(
                review,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get BERT embeddings
            outputs = bert_model(**inputs)
            
            # Mean pooling over tokens
            embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
            embeddings.append(embedding)
    
    # Aggregate all reviews (mean pooling)
    if embeddings:
        aggregated = torch.cat(embeddings, dim=0).mean(dim=0)  # [768]
        return aggregated
    else:
        return torch.zeros(768)

def get_recommendations(
    user_reviews: List[str],
    movie_embeddings: Dict,
    model,
    tokenizer,
    bert_model,
    device,
    top_k: int = 10,
    min_rating: float = 3.5
) -> List[Dict]:
    """
    Generate movie recommendations based on book reviews
    """
    
    # 1. Encode user reviews
    user_embedding = encode_reviews(user_reviews, tokenizer, bert_model)
    user_vector = user_embedding.unsqueeze(0).to(device)  # [1, 768]
    
    # 2. Predict ratings for all movies
    recommendations = []
    
    with torch.no_grad():
        for movie_id, movie_emb in movie_embeddings.items():
            # Convert movie embedding to tensor
            if isinstance(movie_emb, np.ndarray):
                movie_emb = torch.from_numpy(movie_emb).float()
            
            if movie_emb.dim() == 1:
                movie_emb = movie_emb.unsqueeze(0)
            
            movie_vector = movie_emb.mean(dim=0).unsqueeze(0).to(device)  # [1, 768]
            
            # Predict rating
            rating_pred, _, _, _ = model(user_vector, movie_vector)
            predicted_rating = rating_pred.item()
            
            # Filter by minimum rating
            if predicted_rating >= min_rating:
                recommendations.append({
                    'movie_id': movie_id,
                    'predicted_rating': predicted_rating
                })
    
    # 3. Sort by rating (descending)
    recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    # 4. Return top K
    return recommendations[:top_k]

def fetch_movie_title_amazon(asin: str) -> str:
    """Fetch movie title directly from Amazon product page"""
    import urllib.request
    import re
    import html
    import time
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    url = f"https://www.amazon.com/dp/{asin}"
    req = urllib.request.Request(url, headers=headers)
    try:
        time.sleep(0.5) # Prevent rate limiting
        with urllib.request.urlopen(req, timeout=8) as response:
            html_content = response.read().decode('utf-8')
            title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                raw_title = title_match.group(1).strip()
                # Check for robot checks
                if any(x in raw_title.lower() for x in ["captcha", "robot", "api-services-support"]):
                    return None
                    
                # Clean title
                cleaned = html.unescape(raw_title)
                cleaned = re.sub(r'^Amazon\.com\s*:\s*', '', cleaned, flags=re.IGNORECASE)
                parts = cleaned.split(' : ')
                if parts:
                    cleaned = parts[0]
                return cleaned.strip()
    except Exception:
        pass
    return None

def fetch_movie_title_ddg(asin: str) -> str:
    """Fetch movie title from DuckDuckGo HTML search for the ASIN"""
    import urllib.request
    import urllib.parse
    import re
    import html
    import time
    try:
        from bs4 import BeautifulSoup
        BS_AVAILABLE = True
    except ImportError:
        BS_AVAILABLE = False

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    query = f"{asin} amazon"
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers=headers)
    
    try:
        time.sleep(0.5) # Prevent rate limiting
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode('utf-8')
            if BS_AVAILABLE:
                soup = BeautifulSoup(content, 'html.parser')
                results = [r.get_text(strip=True) for r in soup.find_all('a', class_='result__a')]
            else:
                results = re.findall(r'<a class="result__a"[^>]*>\s*(.*?)\s*</a>', content, re.DOTALL)
                results = [re.sub(r'<[^>]*>', '', r).strip() for r in results]
            
            for t_text in results:
                t_lower = t_text.lower()
                # Skip ad titles or generic search pages
                if any(x in t_lower for x in ["order online", "shopping", "selection of products", "duckduckgo", "spend less. smile more."]):
                    continue
                if t_text.strip() in ["Amazon.com", "Amazon"]:
                    continue
                
                # Check if it relates to Amazon or IMDb
                if "amazon" in t_lower or "imdb" in t_lower or asin in t_lower:
                    cleaned = t_text
                    for suffix in [
                        " - amazon.com", " - Amazon.com", " - Amazon", " | Amazon", " : Amazon",
                        " - IMDb", " | IMDb", " - Amazon.co.jp", " - Amazon.co.uk", " - Amazon.ca"
                    ]:
                        cleaned = cleaned.replace(suffix, "")
                    cleaned = html.unescape(cleaned).strip()
                    if cleaned and len(cleaned) > 3:
                        return cleaned
            
            # Fallback to the first non-ad result if we didn't find amazon/imdb in the title
            for t_text in results:
                t_lower = t_text.lower()
                if not any(x in t_lower for x in ["order online", "shopping", "selection", "duckduckgo"]):
                    cleaned = t_text
                    for suffix in [" - amazon.com", " - Amazon.com", " - Amazon", " | Amazon", " - IMDb"]:
                        cleaned = cleaned.replace(suffix, "")
                    cleaned = html.unescape(cleaned).strip()
                    if cleaned:
                        return cleaned
    except Exception:
        pass
        
    return None

def is_generic_title(title: str) -> bool:
    """Helper to detect if a title is a review headline or placeholder instead of a movie title"""
    import re
    title_clean = title.strip().lower().rstrip('.!')
    if len(title_clean) < 4:
        return True
        
    generic_words = {
        "great", "good", "excellent", "love", "awesome", "perfect", "ok", "okay", 
        "bad", "terrible", "worst", "waste", "money", "disappointed", "recommend", 
        "watch", "fun", "boring", "nice", "cool", "wonderful", "amazing", "product", 
        "item", "dvd", "movie", "book", "stars", "five", "one", "favorite", "classic", 
        "video", "show", "film", "series", "season"
    }
    
    # Split title into words
    words = re.findall(r'\b\w+\b', title_clean)
    if not words:
        return True
        
    # If all words in the title are generic review keywords
    generic_count = sum(1 for w in words if w in generic_words or w in ["this", "that", "it", "the", "a", "an", "is", "for", "ever", "highly", "very"])
    if generic_count == len(words):
        return True
        
    # Specific known review headlines
    specific_generics = {"christian keyes", "great bookgreat book great book"}
    if title_clean in specific_generics:
        return True
        
    return False

def resolve_and_cache_title(asin: str, local_title: str, titles_cache: dict) -> str:
    """Check cache, then if generic, resolve online and cache the correct movie title"""
    cache_path = 'data/processed/asin_to_movie_title.json'
    
    # 1. Check if we already have it in local JSON cache file
    if os.path.exists(cache_path):
        try:
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                json_cache = json.load(f)
                if asin in json_cache:
                    return json_cache[asin]
        except Exception:
            pass
            
    # 2. Check if local_title is NOT generic. If so, it is correct, we cache it and return it
    if not is_generic_title(local_title):
        try:
            import json
            json_cache = {}
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    json_cache = json.load(f)
            json_cache[asin] = local_title
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(json_cache, f, indent=4, ensure_ascii=False)
            titles_cache[asin] = local_title # update memory cache
        except Exception:
            pass
        return local_title
        
    # 3. If local_title IS generic, resolve online (Try Amazon Direct first, then DDG)
    resolved = fetch_movie_title_amazon(asin)
    if not resolved:
        resolved = fetch_movie_title_ddg(asin)
        
    if not resolved:
        resolved = f"Movie {asin}"
        
    try:
        import json
        json_cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                json_cache = json.load(f)
        json_cache[asin] = resolved
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(json_cache, f, indent=4, ensure_ascii=False)
        titles_cache[asin] = resolved # update memory cache
    except Exception:
        pass
        
    return resolved

# ══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get personalized movie recommendations based on your book preferences!</p>', unsafe_allow_html=True)
    
    # Load models and titles mapping
    with st.spinner('Loading models & metadata... This may take a moment on first run.'):
        tokenizer, bert_model = load_bert_model()
        model, device, checkpoint = load_recommendation_model()
        movie_embeddings, movie_metadata = load_movie_embeddings()
        movie_titles = load_movie_titles()
        client = load_groq_client() if GROQ_AVAILABLE else None
    
    if model is None or movie_embeddings is None:
        st.error("❌ Failed to load required models or data. Please check file paths.")
        return
    
    # Show model info
    with st.sidebar:
        st.header("📊 Model Information")
        if checkpoint:
            st.info(f"""
            **Model Status:** ✅ Loaded
            
            **Training Epoch:** {checkpoint['epoch'] + 1}
            
            **Validation RMSE:** {checkpoint['val_rmse']:.4f}
            
            **Validation MAE:** {checkpoint['val_mae']:.4f}
            
            **Prediction Std:** {checkpoint['val_pred_std']:.4f}
            """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        top_k = st.slider("Number of recommendations", 5, 20, 10)
        min_rating = st.slider("Minimum rating threshold", 1.0, 5.0, 3.5, 0.1)

        st.markdown("---")
        st.header("🎤 Speech Input")
        if GROQ_AVAILABLE and client:
            st.success("Groq Whisper ready ✅")
            st.caption("Use the 🎤 mic button next to book titles and review boxes to speak your input.")
        elif GROQ_AVAILABLE:
            st.warning("Groq installed but client failed to load. Check your API key.")
        else:
            st.error("Groq not installed.\nRun: `pip install groq`")
            st.caption("After installing, restart the app to enable speech input.")
    
    # Main content
    st.markdown("## 📚 Tell us about books you've enjoyed")
    
    st.markdown('<div class="info-box">💡 <b>Tip:</b> Add reviews of books you loved or hated. The more detail, the better the recommendations!</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'book_reviews' not in st.session_state:
        st.session_state.book_reviews = [
            {'book': '', 'review': '', 'rating': 5}
        ]

    # Book review inputs
    for idx, review_data in enumerate(st.session_state.book_reviews):
        # Initialize widget keys in session state if not already set
        if f"book_{idx}" not in st.session_state:
            st.session_state[f"book_{idx}"] = review_data['book']
        if f"review_{idx}" not in st.session_state:
            st.session_state[f"review_{idx}"] = review_data['review']
        if f"rating_{idx}" not in st.session_state:
            st.session_state[f"rating_{idx}"] = review_data['rating']

        # ─── Speech Transcription (PRE-RENDERING to avoid "cannot modify after instantiation" error) ───
        if GROQ_AVAILABLE and client:
            # 1. Book Title Transcription
            audio_book_key = f"audio_book_{idx}"
            if audio_book_key in st.session_state and st.session_state[audio_book_key] is not None:
                audio_data = st.session_state[audio_book_key]
                audio_bytes = audio_data.getvalue()
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                cache_key = f"audio_book_hash_{idx}"

                if st.session_state.get(cache_key) != audio_hash:
                    st.session_state[cache_key] = audio_hash
                    with st.spinner("🎙️ Transcribing book title..."):
                        try:
                            text = transcribe_audio(audio_bytes, client)
                            if text:
                                text_clean = text.strip().rstrip('.')
                                st.session_state[f"book_{idx}"] = text_clean
                                st.session_state.book_reviews[idx]['book'] = text_clean
                                st.success(f"✅ Transcribed Book: *{text_clean}*")
                                st.rerun()
                            else:
                                st.warning("Could not detect speech for book title. Please try again.")
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")

            # 2. Review Transcription
            audio_review_key = f"audio_review_{idx}"
            if audio_review_key in st.session_state and st.session_state[audio_review_key] is not None:
                audio_data = st.session_state[audio_review_key]
                audio_bytes = audio_data.getvalue()
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                cache_key = f"audio_review_hash_{idx}"

                if st.session_state.get(cache_key) != audio_hash:
                    st.session_state[cache_key] = audio_hash
                    with st.spinner("🎙️ Transcribing review..."):
                        try:
                            text = transcribe_audio(audio_bytes, client)
                            if text:
                                st.session_state[f"review_{idx}"] = text
                                st.session_state.book_reviews[idx]['review'] = text
                                st.success(f"✅ Transcribed Review: *{text[:80]}{'...' if len(text) > 80 else ''}*")
                                st.rerun()
                            else:
                                st.warning("Could not detect speech for review. Please try again.")
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")

        with st.container():
            col1, col2, col3, col4 = st.columns([3, 6, 2, 1])

            with col1:
                book = st.text_input(
                    "Book Title",
                    key=f"book_{idx}",
                    placeholder="e.g., Harry Potter"
                )

                # 🎤 Speech input for Book Title
                if GROQ_AVAILABLE and client:
                    st.audio_input(
                        "🎤 Speak book title",
                        key=f"audio_book_{idx}",
                    )

            with col2:
                review = st.text_area(
                    "Your Review",
                    key=f"review_{idx}",
                    placeholder="What did you think? Be descriptive! (or use 🎤 to speak)",
                    height=100
                )

                # 🎤 Speech input for Review
                if GROQ_AVAILABLE and client:
                    st.audio_input(
                        "🎤 Speak your review",
                        key=f"audio_review_{idx}",
                    )

            with col3:
                rating = st.select_slider(
                    "Your Rating",
                    options=[1, 2, 3, 4, 5],
                    key=f"rating_{idx}"
                )

            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"delete_{idx}"):
                    if len(st.session_state.book_reviews) > 1:
                        # Pop the item
                        st.session_state.book_reviews.pop(idx)
                        
                        # Shift widget keys down to align correctly
                        for i in range(idx, len(st.session_state.book_reviews)):
                            st.session_state[f"book_{i}"] = st.session_state.get(f"book_{i+1}", "")
                            st.session_state[f"review_{i}"] = st.session_state.get(f"review_{i+1}", "")
                            st.session_state[f"rating_{i}"] = st.session_state.get(f"rating_{i+1}", 5)
                            st.session_state[f"audio_book_{i}"] = st.session_state.get(f"audio_book_{i+1}", None)
                            st.session_state[f"audio_review_{i}"] = st.session_state.get(f"audio_review_{i+1}", None)
                            st.session_state[f"audio_book_hash_{i}"] = st.session_state.get(f"audio_book_hash_{i+1}", None)
                            st.session_state[f"audio_review_hash_{i}"] = st.session_state.get(f"audio_review_hash_{i+1}", None)
                        
                        # Pop the last (now unused) keys
                        last_idx = len(st.session_state.book_reviews)
                        st.session_state.pop(f"book_{last_idx}", None)
                        st.session_state.pop(f"review_{last_idx}", None)
                        st.session_state.pop(f"rating_{last_idx}", None)
                        st.session_state.pop(f"audio_book_{last_idx}", None)
                        st.session_state.pop(f"audio_review_{last_idx}", None)
                        st.session_state.pop(f"audio_book_hash_{last_idx}", None)
                        st.session_state.pop(f"audio_review_hash_{last_idx}", None)
                        
                        st.rerun()

            # Keep the main session state dictionary up to date with any manual edits
            st.session_state.book_reviews[idx] = {
                'book': book,
                'review': review,
                'rating': rating
            }

        st.markdown("---")
    
    # Add review button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➕ Add Another Book", use_container_width=True):
            st.session_state.book_reviews.append({
                'book': '',
                'review': '',
                'rating': 5
            })
            st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Get recommendations button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        recommend_button = st.button("🎬 Get Movie Recommendations", use_container_width=True, type="primary")
    
    # Generate recommendations
    if recommend_button:
        # Validate input
        valid_reviews = [
            r['review'] for r in st.session_state.book_reviews 
            if r['book'].strip() and r['review'].strip()
        ]
        
        if len(valid_reviews) == 0:
            st.error("❌ Please add at least one book review before getting recommendations!")
        else:
            with st.spinner('🔮 Analyzing your preferences and generating recommendations...'):
                # Get recommendations
                recommendations = get_recommendations(
                    user_reviews=valid_reviews,
                    movie_embeddings=movie_embeddings,
                    model=model,
                    tokenizer=tokenizer,
                    bert_model=bert_model,
                    device=device,
                    top_k=top_k,
                    min_rating=min_rating
                )
                
                if len(recommendations) == 0:
                    st.warning(f"No movies found with rating ≥ {min_rating:.1f}. Try lowering the threshold.")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("## 🎯 Your Personalized Movie Recommendations")
                    st.markdown(f"*Based on {len(valid_reviews)} book review(s)*")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display recommendations
                    for idx, rec in enumerate(recommendations, 1):
                        rating = rec['predicted_rating']
                        stars = "⭐" * int(round(rating))
                        asin = rec['movie_id']
                        amazon_url = f"https://www.amazon.com/dp/{asin}"
                        movie_title = resolve_and_cache_title(asin, movie_titles.get(asin, f"Movie {asin}"), movie_titles)
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <h3 style="margin-bottom: 0.2rem;">#{idx} {movie_title}</h3>
                            <p style="margin: 0.2rem 0 0.8rem 0; opacity: 0.75; font-size: 0.85rem;">ASIN: {asin} | <a href="{amazon_url}" target="_blank" style="color: white; text-decoration: underline;">View Movie on Amazon ↗</a></p>
                            <div class="rating-badge">{rating:.2f} {stars}</div>
                            <p style="margin-top: 1rem; opacity: 0.9;">
                                Based on your book preferences, we predict you'll rate this movie 
                                <b>{rating:.2f} stars</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Statistics
                    st.markdown("<br>", unsafe_allow_html=True)
                    avg_rating = np.mean([r['predicted_rating'] for r in recommendations])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Recommendations", len(recommendations))
                    with col2:
                        st.metric("⭐ Average Rating", f"{avg_rating:.2f}")
                    with col3:
                        st.metric("🎯 Threshold", f"{min_rating:.1f}+")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><b>Cross-Domain Recommendation System</b></p>
        <p>Transferring preferences from Books → Movies using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
