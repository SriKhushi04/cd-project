import unittest
import pandas as pd
import numpy as np
from asbur.model import ASBUR
from asbur.encoders import BaseEncoder, SentenceTransformerEncoder, get_encoder


class MockEncoder(BaseEncoder):
    """Simple deterministic mock encoder for testing without neural net overhead."""

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            t = text.lower()
            # Map keywords to distinct orthogonal unit vectors
            if "music" in t or "guitar" in t or "jazz" in t:
                vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "robot" in t or "navigation" in t:
                vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            elif "neural" in t or "learning" in t or "vision" in t:
                vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            elif "sports" in t or "soccer" in t or "football" in t:
                vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        return np.asarray(embeddings, dtype=np.float32)


class TestASBUR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo_rows = [
            ("u1", "I build neural networks for image classification.", "machine-learning"),
            ("u1", "Computer vision can support medical diagnosis.", "machine-learning"),
            ("u2", "I play guitar and listen to jazz music.", "music"),
            ("u2", "Jazz improvisation is fascinating.", "music"),
            ("u3", "I study robotics and autonomous navigation.", "robotics"),
            ("u3", "Computer vision is useful for mobile robots.", "robotics"),
            ("u4", "I read about healthcare and medical imaging.", "machine-learning"),
        ]
        cls.df = pd.DataFrame(cls.demo_rows, columns=["user_id", "text", "community"])
        cls.df["timestamp"] = pd.date_range("2019-01-01", periods=len(cls.df), tz="UTC")

    def test_music_query(self):
        model = ASBUR(min_community_similarity=0.25).fit(self.df)
        predicted, sims = model.infer_community("I love playing guitar and listening to jazz music", return_similarities=True)
        self.assertEqual(predicted, "music")
        self.assertGreater(sims["music"], 0.25)

        results = model.rank("I love playing guitar and listening to jazz music", top_k=3)
        self.assertEqual(results[0]["user_id"], "u2")
        self.assertEqual(results[0]["inferred_community"], "music")

    def test_robotics_query(self):
        model = ASBUR(min_community_similarity=0.25).fit(self.df)
        predicted, sims = model.infer_community("Building autonomous mobile robots and sensors", return_similarities=True)
        self.assertEqual(predicted, "robotics")
        self.assertGreater(sims["robotics"], 0.25)

        results = model.rank("Building autonomous mobile robots and sensors", top_k=3)
        self.assertEqual(results[0]["user_id"], "u3")

    def test_machine_learning_query(self):
        model = ASBUR(min_community_similarity=0.25).fit(self.df)
        predicted, sims = model.infer_community("Deep learning algorithms for image recognition", return_similarities=True)
        self.assertEqual(predicted, "machine-learning")

        results = model.rank("Deep learning algorithms for image recognition", top_k=3)
        self.assertIn(results[0]["user_id"], ["u1", "u4"])

    def test_out_of_domain_sports_query(self):
        sports_query = "Who will win the football championship soccer match this weekend?"

        # Semantic fallback mode: returns 0.0 behavior score
        model_sem = ASBUR(min_community_similarity=0.35, fallback_behavior="semantic").fit(self.df)
        predicted, sims = model_sem.infer_community(sports_query, return_similarities=True)
        self.assertIsNone(predicted, f"Expected None for out-of-domain query, got {predicted} with similarities {sims}")
        results_sem = model_sem.rank(sports_query, top_k=3)
        self.assertIsNone(results_sem[0]["inferred_community"])
        self.assertEqual(results_sem[0]["behavior_score"], 0.0)

        # Activity fallback mode: provides defensible global activity prior
        model_act = ASBUR(min_community_similarity=0.35, fallback_behavior="activity").fit(self.df)
        results_act = model_act.rank(sports_query, top_k=3)
        self.assertIsNone(results_act[0]["inferred_community"])
        self.assertGreater(results_act[0]["behavior_score"], 0.0)

    def test_duplicate_embeddings_no_kmeans_warning(self):
        mock_enc = MockEncoder()
        # User with 3 duplicate posts
        dup_rows = [
            ("user_dup", "I play guitar jazz music.", "music"),
            ("user_dup", "I play guitar jazz music.", "music"),
            ("user_dup", "I play guitar jazz music.", "music"),
        ]
        df_dup = pd.DataFrame(dup_rows, columns=["user_id", "text", "community"])
        df_dup["timestamp"] = pd.date_range("2019-01-01", periods=len(df_dup), tz="UTC")
        model = ASBUR(encoder=mock_enc, max_prototypes=3).fit(df_dup)
        self.assertEqual(len(model.profiles["user_dup"]), 1)

    def test_precomputed_vectors_shape_validation(self):
        mock_enc = MockEncoder()
        valid_vecs = np.zeros((len(self.df), 4), dtype=np.float32)
        valid_vecs[:, 0] = 1.0
        model = ASBUR(encoder=mock_enc).fit(self.df, precomputed_vectors=valid_vecs)
        self.assertTrue(model.is_fitted)

        # Mismatched rows should raise ValueError
        invalid_vecs = np.zeros((len(self.df) - 1, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            ASBUR(encoder=mock_enc).fit(self.df, precomputed_vectors=invalid_vecs)

    def test_mock_encoder_isolation(self):
        mock_enc = MockEncoder()
        model = ASBUR(encoder=mock_enc, min_community_similarity=0.3).fit(self.df)
        
        predicted = model.infer_community("soccer football match")
        self.assertIsNone(predicted)

        predicted_music = model.infer_community("guitar jazz music")
        self.assertEqual(predicted_music, "music")

    def test_get_encoder_factory(self):
        enc = get_encoder("sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsInstance(enc, SentenceTransformerEncoder)


if __name__ == "__main__":
    unittest.main()

