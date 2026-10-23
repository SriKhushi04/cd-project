from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from asbur.encoders import get_encoder, BaseEncoder


class ASBUR:
    """
    ASBUR: Sentence-Driven Multi-Interest and
    Behavior-Aware User Recommendation.

    Semantic relevance:
        Maximum cosine similarity between a query and
        a user's interest prototypes.

    Behavioral relevance:
        Log-normalized activity in the inferred community.

    Final score:
        Adaptive weighted combination of semantic and
        behavioral relevance.
    """

    def __init__(
        self,
        encoder="sentence-transformers/all-MiniLM-L6-v2",
        max_prototypes=4,
        alpha_min=0.4,
        alpha_max=0.8,
        tau=5.0,
        min_community_similarity=0.35,
        random_state=42,
        batch_size=64,
        fallback_behavior="activity",
    ):
        if max_prototypes < 1:
            raise ValueError("max_prototypes must be at least 1.")
        if not (0 <= alpha_min <= alpha_max <= 1):
            raise ValueError(
                "Weights must satisfy 0 <= alpha_min <= alpha_max <= 1."
            )
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if fallback_behavior not in ("activity", "semantic"):
            raise ValueError("fallback_behavior must be 'activity' or 'semantic'.")

        self.encoder = get_encoder(encoder)
        self.max_prototypes = max_prototypes
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.tau = tau
        self.min_community_similarity = min_community_similarity
        self.random_state = random_state
        self.batch_size = batch_size
        self.fallback_behavior = fallback_behavior

        self.profiles = {}
        self.history_size = {}
        self.community_counts = {}
        self.community_vectors = {}
        self.users = []
        self.prototype_matrix = None
        self.user_proto_indices = {}
        self.is_fitted = False

    @staticmethod
    def _norm(x):
        """Normalize vectors safely along the final axis."""
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def fit(self, train_df, cache_path=None, precomputed_vectors=None):
        """Build user interest profiles and community statistics."""

        required = {"user_id", "text", "community"}
        missing = required - set(train_df.columns)

        if missing:
            raise ValueError(
                f"Training data is missing columns: {sorted(missing)}"
            )

        df = train_df.copy()
        df = df.dropna(subset=["user_id", "text", "community"])

        for column in ("user_id", "text", "community"):
            df[column] = df[column].astype(str).str.strip()

        df = df[
            (df["user_id"] != "")
            & (df["text"] != "")
            & (df["community"] != "")
        ].reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid training rows remain after cleaning.")

        import os
        vectors = None

        if precomputed_vectors is not None:
            vectors = np.asarray(precomputed_vectors, dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(df):
                raise ValueError(
                    f"precomputed_vectors shape {vectors.shape} does not match training dataframe length {len(df)}."
                )
        elif cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading cached training post embeddings from: {cache_path}")
                loaded = np.load(cache_path)
                if loaded.ndim == 2 and loaded.shape[0] == len(df) and loaded.shape[1] > 0:
                    vectors = loaded
                else:
                    print(f"Cache shape mismatch ({loaded.shape} vs expected ({len(df)}, dim)), re-encoding...")
                    vectors = None
            except Exception as e:
                print(f"Failed to load cache ({e}), re-encoding...")
                vectors = None

        if vectors is None:
            print(f"Encoding {len(df)} training posts...")
            vectors = self.encoder.encode(
                df["text"].tolist(),
                batch_size=max(256, self.batch_size),
                show_progress_bar=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, vectors)
                print(f"Saved training embeddings cache to: {cache_path}")

        self.users = sorted(df["user_id"].unique().tolist())
        self.profiles = {}
        self.history_size = {}

        # Construct multiple interest prototypes for each user.
        for user, group in df.groupby("user_id", sort=True):
            indices = group.index.to_numpy()
            user_vectors = vectors[indices]
            n_posts = len(user_vectors)

            # Avoid asking K-means for more clusters than distinct data points.
            unique_vectors = np.unique(user_vectors, axis=0)
            n_distinct = len(unique_vectors)

            k = min(
                self.max_prototypes,
                max(1, int(np.sqrt(n_posts))),
                n_distinct,
            )

            if k <= 1:
                centers = user_vectors.mean(axis=0, keepdims=True)
            else:
                clustering = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=self.random_state,
                )
                clustering.fit(user_vectors)
                centers = clustering.cluster_centers_

            self.profiles[str(user)] = self._norm(centers)
            self.history_size[str(user)] = n_posts

        # Vectorize prototype matrix for fast batch matrix multiplication during ranking
        all_centers = []
        self.user_proto_indices = {}
        curr = 0
        for u in self.users:
            centers = self.profiles[u]
            n_c = len(centers)
            all_centers.append(centers)
            self.user_proto_indices[u] = (curr, curr + n_c)
            curr += n_c

        if all_centers:
            self.prototype_matrix = np.vstack(all_centers)
        else:
            self.prototype_matrix = None

        # Count each user's activity in each community.
        counts = defaultdict(lambda: defaultdict(int))

        for row in df.itertuples(index=False):
            counts[str(row.user_id)][str(row.community)] += 1

        self.community_counts = {
            user: dict(counts[user]) for user in self.users
        }

        # Compute community representations combining semantic label prior with empirical post centroids.
        self.community_vectors = {}
        communities = df["community"].to_numpy()
        unique_communities = sorted(set(communities))

        # Encode community names (clearing underscores) to establish robust semantic anchors
        cleaned_names = [c.replace("_", " ") for c in unique_communities]
        name_vectors = self.encoder.encode(cleaned_names, show_progress_bar=False)

        for i, community in enumerate(unique_communities):
            community_posts = vectors[communities == community]
            name_vec = self._norm(name_vectors[i])
            if len(community_posts) > 0:
                mean_post_vector = self._norm(community_posts.mean(axis=0, keepdims=True))[0]
                # Blend 60% semantic label prior + 40% empirical post centroid to handle sparse communities
                blended = 0.6 * name_vec + 0.4 * mean_post_vector
                self.community_vectors[community] = self._norm(blended)
            else:
                self.community_vectors[community] = name_vec

        self.is_fitted = True

        print(
            f"ASBUR fitted: {len(self.users)} users, "
            f"{len(self.community_vectors)} communities."
        )

        return self

    def infer_community(self, query, return_similarities=False):
        """
        Predict the community most semantically similar to a query.

        If the maximum similarity is below self.min_community_similarity,
        returns None to indicate that no suitable community was found.

        Note: Cosine similarity alone is a heuristic threshold and not a
        formally calibrated probability distribution.
        """

        if not self.is_fitted:
            raise RuntimeError("Call fit() before infer_community().")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty sentence.")

        query_vector = self.encoder.encode([query.strip()])[0]

        similarities = {
            community: float(np.dot(query_vector, vector))
            for community, vector in self.community_vectors.items()
        }

        if not similarities:
            predicted = None
        else:
            best_community = max(similarities, key=similarities.get)
            max_sim = similarities[best_community]
            if self.min_community_similarity is not None and max_sim < self.min_community_similarity:
                predicted = None
            else:
                predicted = best_community

        sorted_sims = dict(
            sorted(
                similarities.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        if return_similarities:
            return predicted, sorted_sims

        return predicted

    def rank(self, query, top_k=10, exclude_user=None, debug=False):
        """Rank users for a sentence-driven recommendation query."""

        if not self.is_fitted:
            raise RuntimeError("Call fit() before rank().")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty sentence.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = query.strip()
        query_vector = self.encoder.encode([query])[0]

        # Batch matrix multiplication over all prototypes
        if self.prototype_matrix is not None and self.prototype_matrix.shape[0] > 0:
            proto_sims = np.dot(self.prototype_matrix, query_vector)
            use_matrix = True
        else:
            use_matrix = False

        # Infer community with similarity scores.
        community, community_similarities = self.infer_community(
            query, return_similarities=True
        )

        if debug:
            print("\nCommunity similarities:")
            for name, similarity in community_similarities.items():
                print(f"  {name}: {similarity:.4f}")
            if community is None:
                max_sim = max(community_similarities.values()) if community_similarities else 0.0
                print(
                    f"Selected community: None "
                    f"(Low confidence: max similarity {max_sim:.4f} < threshold {self.min_community_similarity})\n"
                    f"Fallback diagnostic: Query interests are not represented in training communities. "
                    f"Defaulting to pure semantic prototype matching.\n"
                )
            else:
                print(f"Selected community: {community}\n")

        max_history = max(self.history_size.values()) if self.history_size else 1
        history_denominator = np.log1p(max_history) if max_history > 0 else 1.0

        if community is not None:
            max_count = max(
                (
                    self.community_counts[user].get(community, 0)
                    for user in self.users
                ),
                default=0,
            )
            denominator = np.log1p(max_count) if max_count > 0 else 1.0
            best_sim = community_similarities.get(community, 1.0)
            conf = float(np.clip(
                (best_sim - self.min_community_similarity) / max(1.0 - self.min_community_similarity, 1e-6),
                0.0,
                1.0,
            ))
        else:
            max_count = 0
            denominator = 1.0
            conf = 0.0

        results = []

        for user in self.users:
            if exclude_user is not None and user == str(exclude_user):
                continue

            # Best semantic match across the user's interest prototypes.
            if use_matrix:
                s, e = self.user_proto_indices[user]
                semantic_raw = float(np.max(proto_sims[s:e]))
            else:
                semantic_raw = max(
                    float(np.dot(query_vector, prototype))
                    for prototype in self.profiles[user]
                )

            # Map cosine similarity from [-1, 1] to [0, 1].
            semantic_score = float(np.clip(
                (semantic_raw + 1.0) / 2.0, 0.0, 1.0
            ))

            history = self.history_size[user]
            global_activity = float(np.log1p(history) / history_denominator)

            # Behavioral relevance: community-specific activity scaled by prediction confidence,
            # falling back gracefully to defensible behavioral signal (global activity or zero).
            if community is not None and max_count > 0:
                count = self.community_counts[user].get(community, 0)
                comm_activity = float(np.log1p(count) / denominator)
                if self.fallback_behavior == "activity":
                    behavior_score = float(conf * comm_activity + (1.0 - conf) * global_activity)
                else:
                    behavior_score = float(conf * comm_activity)
            else:
                behavior_score = global_activity if self.fallback_behavior == "activity" else 0.0

            # Longer user histories move the semantic weight toward alpha_max.
            alpha = self.alpha_min + (
                self.alpha_max - self.alpha_min
            ) * history / (history + self.tau)

            # Final score calculation
            if community is None and self.fallback_behavior == "semantic":
                final_score = semantic_score
            else:
                final_score = (
                    alpha * semantic_score
                    + (1.0 - alpha) * behavior_score
                )

            results.append({
                "user_id": user,
                "score": float(final_score),
                "semantic_score": semantic_score,
                "behavior_score": behavior_score,
                "inferred_community": community,
                "history_size": history,
                "alpha": float(alpha),
            })

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]
