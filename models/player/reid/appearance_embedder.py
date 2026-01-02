import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class AppearanceFeatures:
    """Container for extracted appearance features."""

    color_histogram: np.ndarray
    dominant_colors: np.ndarray
    texture_features: np.ndarray
    embedding: Optional[np.ndarray] = None

    def to_vector(self) -> np.ndarray:
        """Concatenate all features into single vector."""
        features = [
            self.color_histogram.flatten(),
            self.dominant_colors.flatten(),
            self.texture_features.flatten(),
        ]
        if self.embedding is not None:
            features.append(self.embedding.flatten())
        return np.concatenate(features)


class AppearanceEmbedder:
    """
    Extract appearance features from player crops.

    Combines:
    - Color histograms (HSV space)
    - Dominant color extraction
    - Optional deep embeddings
    """

    def __init__(
        self,
        use_deep_features: bool = False,
        reid_model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize appearance embedder.

        Args:
            use_deep_features: Whether to use deep ReID features
            reid_model_path: Path to ReID model (required if use_deep_features)
            device: Device for deep model inference
        """
        self.use_deep_features = use_deep_features
        self.reid_extractor = None

        if use_deep_features and reid_model_path:
            from .reid_model import ReIDFeatureExtractor

            self.reid_extractor = ReIDFeatureExtractor(reid_model_path, device)

    def extract_color_histogram(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bins: Tuple[int, int, int] = (8, 12, 8),
    ) -> np.ndarray:
        """
        Extract HSV color histogram from image.

        Args:
            image: BGR image (crop of player)
            mask: Optional mask for region of interest
            bins: Number of bins for H, S, V channels

        Returns:
            Normalized histogram vector
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(bins), [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def extract_dominant_colors(
        self,
        image: np.ndarray,
        n_colors: int = 3,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract dominant colors using K-means clustering.

        Args:
            image: BGR image
            n_colors: Number of dominant colors to extract
            mask: Optional mask

        Returns:
            Array of dominant colors in LAB space (n_colors, 3)
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if mask is not None:
            pixels = lab[mask > 0].reshape(-1, 3)
        else:
            pixels = lab.reshape(-1, 3)

        if len(pixels) < n_colors:
            return np.zeros((n_colors, 3), dtype=np.float32)

        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
        kmeans.fit(pixels.astype(np.float32))

        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(-counts)

        return kmeans.cluster_centers_[sorted_indices].astype(np.float32)

    def extract_texture_features(
        self,
        image: np.ndarray,
        n_points: int = 24,
        radius: int = 3,
    ) -> np.ndarray:
        """
        Extract Local Binary Pattern (LBP) texture features.

        Args:
            image: BGR image
            n_points: Number of points for LBP
            radius: Radius for LBP

        Returns:
            LBP histogram
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                for k, angle in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
                    ni = int(i + radius * np.sin(angle))
                    nj = int(j + radius * np.cos(angle))
                    if gray[ni, nj] >= center:
                        code |= 1 << k
                lbp[i - radius, j - radius] = code

        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / hist.sum()

        return hist

    def get_jersey_region_mask(
        self,
        image: np.ndarray,
        upper_ratio: float = 0.5,
    ) -> np.ndarray:
        """
        Create mask for upper body (jersey) region.

        Args:
            image: Player crop
            upper_ratio: Fraction of height for upper body

        Returns:
            Binary mask
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        upper_h = int(h * upper_ratio)
        margin_w = int(w * 0.2)

        mask[int(h * 0.1) : upper_h, margin_w : w - margin_w] = 255

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

        return mask

    def extract(self, image: np.ndarray) -> AppearanceFeatures:
        """
        Extract full appearance features from player crop.

        Args:
            image: BGR image of player crop

        Returns:
            AppearanceFeatures object
        """
        jersey_mask = self.get_jersey_region_mask(image)

        color_hist = self.extract_color_histogram(image, jersey_mask)
        dominant_colors = self.extract_dominant_colors(image, mask=jersey_mask)
        texture_feat = self.extract_texture_features(image)

        embedding = None
        if self.use_deep_features and self.reid_extractor:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            embedding = self.reid_extractor.extract([rgb])[0]

        return AppearanceFeatures(
            color_histogram=color_hist,
            dominant_colors=dominant_colors,
            texture_features=texture_feat,
            embedding=embedding,
        )

    def compute_similarity(
        self,
        feat1: AppearanceFeatures,
        feat2: AppearanceFeatures,
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Compute similarity between two appearance feature sets.

        Args:
            feat1: First feature set
            feat2: Second feature set
            weights: Weights for different feature types

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        weights = weights or {
            "color_histogram": 0.4,
            "dominant_colors": 0.3,
            "texture": 0.1,
            "embedding": 0.2,
        }

        similarity = 0.0

        hist_sim = np.minimum(feat1.color_histogram, feat2.color_histogram).sum()
        similarity += weights["color_histogram"] * hist_sim

        color_dist = np.linalg.norm(feat1.dominant_colors - feat2.dominant_colors)
        color_sim = 1.0 / (1.0 + color_dist / 50.0)
        similarity += weights["dominant_colors"] * color_sim

        tex_sim = np.minimum(feat1.texture_features, feat2.texture_features).sum()
        similarity += weights["texture"] * tex_sim

        if feat1.embedding is not None and feat2.embedding is not None:
            emb_sim = np.dot(feat1.embedding, feat2.embedding)
            emb_sim = (emb_sim + 1) / 2
            similarity += weights["embedding"] * emb_sim

        return similarity


class TeamClassifier:
    """
    Classify players into teams based on appearance.

    Uses unsupervised clustering with:
    - Gaussian Mixture Models for soft assignment
    - Temporal smoothing for consistency
    - Special handling for goalkeepers and referees
    """

    def __init__(
        self,
        n_teams: int = 2,
        n_components: int = 3,  # 2 teams + 1 referee
        min_samples_for_clustering: int = 6,
    ):
        """
        Initialize team classifier.

        Args:
            n_teams: Number of teams (usually 2)
            n_components: Number of GMM components
            min_samples_for_clustering: Minimum samples before clustering
        """
        self.n_teams = n_teams
        self.n_components = n_components
        self.min_samples = min_samples_for_clustering

        self.gmm: Optional[GaussianMixture] = None
        self.feature_buffer: List[np.ndarray] = []
        self.team_centroids: Optional[np.ndarray] = None

        self.track_history: Dict[int, List[int]] = defaultdict(list)

    def _extract_team_features(self, features: AppearanceFeatures) -> np.ndarray:
        """Extract features relevant for team classification."""
        return np.concatenate(
            [
                features.dominant_colors.flatten(),
                features.color_histogram[:32],
            ]
        )

    def update(
        self,
        track_id: int,
        features: AppearanceFeatures,
    ) -> Optional[int]:
        """
        Update classifier with new observation and return team prediction.

        Args:
            track_id: Track ID for temporal smoothing
            features: Appearance features

        Returns:
            Team ID (0, 1, or 2 for referee) or None if not enough data
        """
        team_feat = self._extract_team_features(features)
        self.feature_buffer.append(team_feat)

        if len(self.feature_buffer) < self.min_samples:
            return None

        if self.gmm is None or len(self.feature_buffer) % 50 == 0:
            self._fit_gmm()

        if self.gmm is not None:
            prediction = self.gmm.predict([team_feat])[0]

            self.track_history[track_id].append(prediction)
            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id] = self.track_history[track_id][-10:]

            history = self.track_history[track_id]
            if len(history) >= 3:
                return max(set(history), key=history.count)
            return prediction

        return None

    def _fit_gmm(self) -> None:
        """Fit GMM to accumulated features."""
        features = np.array(self.feature_buffer)

        if len(features) > 500:
            indices = np.concatenate(
                [
                    np.arange(len(features) - 200, len(features)),
                    np.random.choice(len(features) - 200, 100, replace=False),
                ]
            )
            self.feature_buffer = [self.feature_buffer[i] for i in indices]
            features = np.array(self.feature_buffer)

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            n_init=3,
            reg_covar=1e-3,
            random_state=42,
        )
        self.gmm.fit(features)

        self.team_centroids = self.gmm.means_

    def predict_batch(
        self,
        features_list: List[AppearanceFeatures],
        track_ids: List[int],
    ) -> List[int]:
        """
        Predict teams for a batch of features.

        Args:
            features_list: List of appearance features
            track_ids: Corresponding track IDs

        Returns:
            List of team predictions
        """
        predictions = []
        for features, track_id in zip(features_list, track_ids):
            pred = self.update(track_id, features)
            predictions.append(pred if pred is not None else -1)
        return predictions

    def get_team_colors(self) -> Optional[np.ndarray]:
        """
        Get estimated team colors.

        Returns:
            Array of team colors (n_components, 3) in LAB space
        """
        if self.team_centroids is None:
            return None

        team_colors = self.team_centroids[:, :9].reshape(-1, 3, 3)

        return team_colors[:, 0, :]

    def reset(self) -> None:
        """Reset classifier state."""
        self.gmm = None
        self.feature_buffer = []
        self.team_centroids = None
        self.track_history.clear()


class AppearanceTracker:
    """
    Integrate appearance features with tracking.

    Maintains appearance gallery for each track and
    uses appearance matching for re-identification.
    """

    def __init__(
        self,
        embedder: AppearanceEmbedder,
        max_gallery_size: int = 20,
        appearance_weight: float = 0.3,
    ):
        """
        Initialize appearance tracker.

        Args:
            embedder: Appearance feature extractor
            max_gallery_size: Max features per track
            appearance_weight: Weight for appearance in matching
        """
        self.embedder = embedder
        self.max_gallery_size = max_gallery_size
        self.appearance_weight = appearance_weight
        self.galleries: Dict[int, List[AppearanceFeatures]] = defaultdict(list)

        self.team_classifier = TeamClassifier()

    def update_track(
        self,
        track_id: int,
        image: np.ndarray,
        bbox: List[float],
    ) -> Tuple[AppearanceFeatures, Optional[int]]:
        """
        Update track with new observation.

        Args:
            track_id: Track ID
            image: Full frame
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Tuple of (features, team_id)
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        features = self.embedder.extract(crop)

        gallery = self.galleries[track_id]
        gallery.append(features)
        if len(gallery) > self.max_gallery_size:
            indices = [0, len(gallery) - 1] + list(
                np.random.choice(
                    range(1, len(gallery) - 1), min(self.max_gallery_size - 2, len(gallery) - 2), replace=False
                )
            )
            self.galleries[track_id] = [gallery[i] for i in sorted(set(indices))]

        team_id = self.team_classifier.update(track_id, features)

        return features, team_id

    def compute_appearance_cost(
        self,
        track_id: int,
        detection_crop: np.ndarray,
    ) -> float:
        """
        Compute appearance cost for track-detection association.

        Args:
            track_id: Existing track ID
            detection_crop: Detection crop image

        Returns:
            Cost value (lower = better match)
        """
        if track_id not in self.galleries or len(self.galleries[track_id]) == 0:
            return 0.5

        det_features = self.embedder.extract(detection_crop)

        gallery = self.galleries[track_id]
        similarities = [self.embedder.compute_similarity(det_features, gal_feat) for gal_feat in gallery]

        max_sim = max(similarities)

        return 1.0 - max_sim

    def get_gallery_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get mean embedding for a track's gallery."""
        if track_id not in self.galleries or len(self.galleries[track_id]) == 0:
            return None

        gallery = self.galleries[track_id]

        embeddings = [f.embedding for f in gallery if f.embedding is not None]
        if embeddings:
            return np.mean(embeddings, axis=0)

        return np.mean([f.to_vector() for f in gallery], axis=0)

    def remove_track(self, track_id: int) -> None:
        """Remove track from galleries."""
        if track_id in self.galleries:
            del self.galleries[track_id]
