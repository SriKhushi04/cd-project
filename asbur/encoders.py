import os
import json
import hashlib
import urllib.request
import urllib.error
import numpy as np


class BaseEncoder:
    """Abstract interface for text embedding encoders."""

    def encode(self, texts):
        """
        Encode a list of text strings into normalized float32 numpy embeddings.
        Returns array of shape (len(texts), embedding_dim).
        """
        raise NotImplementedError


class SentenceTransformerEncoder(BaseEncoder):
    """Local Sentence-BERT encoder using sentence-transformers."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=256, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class GeminiEncoder(BaseEncoder):
    """
    Optional Google Gemini Embedding API encoder.
    Uses model 'text-embedding-004' via REST API.
    
    Includes:
    - Environment variable lookup for GEMINI_API_KEY (or .env file)
    - Local disk caching to avoid redundant API calls and save quota
    - Rate limit (429) & network error handling
    - Strict privacy controls (only enabled when configured)
    """

    def __init__(
        self,
        api_key=None,
        model_name="text-embedding-004",
        cache_dir=".cache/gemini_embeddings",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Load .env file if available and key is not set
        if not self.api_key and os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        self.api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set the GEMINI_API_KEY environment variable "
                "or create a .env file with GEMINI_API_KEY=your_key."
            )

        self.model_name = model_name
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, text):
        if not self.cache_dir:
            return None
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.json")

    def _load_cached(self, text):
        cache_path = self._get_cache_path(text)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return np.array(data["embedding"], dtype=np.float32)
            except Exception:
                return None
        return None

    def _save_cached(self, text, embedding):
        cache_path = self._get_cache_path(text)
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"text": text, "embedding": embedding.tolist()}, f)
            except Exception:
                pass

    def _encode_single(self, text):
        cached = self._load_cached(text)
        if cached is not None:
            return cached

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                values = result["embedding"]["values"]
                vec = np.array(values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._save_cached(text, vec)
                return vec
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RuntimeError(
                    "Gemini API rate limit exceeded (HTTP 429). Please retry after a brief pause."
                ) from e
            elif e.code == 400 or e.code == 403:
                raise ValueError(
                    f"Gemini API authentication/request error (HTTP {e.code}): {e.reason}"
                ) from e
            else:
                raise RuntimeError(f"Gemini API HTTP Error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error connecting to Gemini API: {e.reason}") from e

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            vec = self._encode_single(text)
            embeddings.append(vec)

        return np.asarray(embeddings, dtype=np.float32)


def get_encoder(encoder_spec=None, **kwargs):
    """
    Factory function to instantiate an encoder.
    encoder_spec can be:
    - BaseEncoder instance (returned as is)
    - "gemini" -> GeminiEncoder
    - model name string (e.g. "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformerEncoder
    """
    if isinstance(encoder_spec, BaseEncoder):
        return encoder_spec

    if isinstance(encoder_spec, str):
        if encoder_spec.lower() == "gemini":
            return GeminiEncoder(**kwargs)
        return SentenceTransformerEncoder(model_name=encoder_spec)

    return SentenceTransformerEncoder()

