"""
Player identity and team assignment.

This module provides tools for assigning player identities using
ReID embeddings (when available) or color clustering as fallback,
including team classification, goalie detection, and referee identification.

Example:
    >>> from torchkick.tracking import IdentityAssigner
    >>>
    >>> assigner = IdentityAssigner(fps=30.0)
    >>> assignments = assigner.assign_roles(trajectory_store)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from torchkick.tracking.models import (
    PlayerSlot,
    TrackData,
)
from torchkick.tracking.trajectory import TrajectoryStore

if TYPE_CHECKING:
    from torchkick.models.reid import DINOv2ReIDEmbedder


class IdentityAssigner:
    """
    Assign player identities using ReID embeddings or color clustering.

    Strategy:
    1. If observations have reid_embedding set: use DINOv2 embedding clustering
    2. Otherwise: fall back to color GMM (legacy path)
    3. Spatial analysis for goalies (penalty area) and linesmen (sidelines)
    4. Outlier detection for referee

    Args:
        fps: Video frame rate.
        embedder: Optional DINOv2ReIDEmbedder for unsupervised team clustering.
            When provided and reid_embedding fields are populated in observations,
            clustering uses cosine-similarity spectral clustering instead of GMM.
            Falls back to color GMM when embeddings are absent.
        debug: Print debug information.

    Example:
        >>> assigner = IdentityAssigner(fps=30.0)
        >>> assignments = assigner.assign_roles(store)
        >>> for tid, info in assignments.items():
        ...     print(f"Track {tid}: {info['role']} Team {info['team']}")
    """

    def __init__(
        self,
        fps: float = 30.0,
        embedder: Optional["DINOv2ReIDEmbedder"] = None,
        siglip_embedder: Optional["SigLIPTeamEmbedder"] = None,
        team_centroids: Optional[np.ndarray] = None,
        debug: bool = True,
    ) -> None:
        self.fps = fps
        self.embedder = embedder
        self.siglip_embedder = siglip_embedder
        self.team_centroids = team_centroids  # [3, D] from calibrate_team_centroids()
        self.debug = debug

    def assign_roles(
        self,
        store: TrajectoryStore,
        team_labels: Optional[Dict[int, int]] = None,
    ) -> Dict[int, Dict]:
        """
        Assign roles and teams to all tracks.

        Args:
            store:       TrajectoryStore with accumulated observations.
            team_labels: Optional pre-computed team labels from GameTeamEmbedder.
                         When provided, bypasses internal embedding clustering.
                         Labels: 0=team0, 1=team1, 2=ref, -1=noise (→ team 0).

        Returns:
            Dict mapping track_id to {'role', 'team', 'player_id'}.
        """
        tracks = store.get_long_tracks(min_frames=30)

        if self.debug:
            print(f"[IdentityAssigner] Processing {len(tracks)} long tracks")

        # Compute position statistics
        track_stats = {}
        for track in tracks:
            stats = track.pitch_position_stats()
            if stats:
                track_stats[track.track_id] = stats

        if not track_stats:
            return {}

        # Find special roles by position
        goalie_candidates = self._find_goalie_candidates(track_stats, store=store, team_centroids=self.team_centroids)
        linesman_candidates = self._find_linesman_candidates(track_stats, exclude=goalie_candidates)
        remaining_ids = [
            tid for tid in track_stats.keys() if tid not in goalie_candidates and tid not in linesman_candidates
        ]

        # Cluster: use pre-computed labels if provided, else fall back to ReID/embedding clustering.
        if team_labels is not None:
            if self.debug:
                print("[IdentityAssigner] Using pre-computed team labels from GameTeamEmbedder")
            referee_id = None
            team_assignments = {}
            for tid in remaining_ids:
                label = team_labels.get(tid, 0)
                if label == 2:
                    # Mark as referee — pick the first one as representative
                    team_assignments[tid] = 2
                    if referee_id is None:
                        referee_id = tid
                else:
                    team_assignments[tid] = max(label, 0)  # -1 noise → 0
        else:
            has_reid = self._has_reid_embeddings(store, remaining_ids)
            if has_reid:
                team_assignments, referee_id = self._cluster_by_reid(store, remaining_ids)
            else:
                if self.debug:
                    print(
                        "[IdentityAssigner] No ReID embeddings — team classification unavailable. "
                        "Run with --reid-weights to enable. Defaulting all to team 0."
                    )
                team_assignments = {tid: 0 for tid in remaining_ids}
                referee_id = None

        # Assign goalies to teams
        goalie_teams = self._assign_goalie_teams(goalie_candidates, track_stats, team_assignments)

        # Build results
        results = {}
        for tid in track_stats.keys():
            track = store.get_track(tid)

            if tid in goalie_candidates:
                results[tid] = {
                    "role": "goalie",
                    "team": goalie_teams.get(tid, -1),
                    "player_id": 1,
                }
                if track:
                    track.role = "goalie"
                    track.team = goalie_teams.get(tid, -1)

            elif tid in linesman_candidates:
                results[tid] = {
                    "role": "linesman",
                    "team": -1,
                    "player_id": None,
                }
                if track:
                    track.role = "linesman"
                    track.team = -1

            elif tid == referee_id:
                results[tid] = {
                    "role": "referee",
                    "team": -1,
                    "player_id": None,
                }
                if track:
                    track.role = "referee"
                    track.team = -1

            else:
                team = team_assignments.get(tid, 0)
                results[tid] = {
                    "role": "player",
                    "team": team,
                    "player_id": None,
                }
                if track:
                    track.role = "player"
                    track.team = team

        if self.debug:
            self._print_summary(results)

        return results

    def _find_goalie_candidates(
        self,
        track_stats: Dict,
        store: Optional["TrajectoryStore"] = None,
        team_centroids: Optional[np.ndarray] = None,
    ) -> Set[int]:
        """
        Find goalkeeper candidates by position isolation.

        A track qualifies if:
        1. It is the most extreme-x player on one side of the pitch
        2. It is ≥ ``isolation_gap`` metres behind the next field player
        3. Its mean_x is outside ±20 m from centre (clearly in defensive half)

        When ``team_centroids`` is provided, tracks whose embedding has high
        cosine distance from both team centroids (different-coloured jersey)
        are accepted with a looser isolation gap (3 m instead of 5 m).
        """
        candidates: Set[int] = set()

        # Only stable, long tracks
        stable = {tid: s for tid, s in track_stats.items() if s["std_x"] < 12 and s["n_samples"] > 50}
        if len(stable) < 3:
            return candidates

        # Identify embedding outliers (different jersey → lower threshold)
        outlier_tids: Set[int] = set()
        if team_centroids is not None and store is not None:
            for tid in stable:
                track = store.get_track(tid)
                if track is None:
                    continue
                embeds = [o.reid_embedding for o in track.observations if o.reid_embedding is not None]
                if not embeds:
                    continue
                mean_emb = np.mean(embeds, axis=0).astype(np.float64)
                norm = np.linalg.norm(mean_emb)
                if norm < 1e-8:
                    continue
                mean_emb /= norm
                # Cosine distance from team0 and team1 centroids (index 0 and 1)
                dists = []
                for i in range(2):
                    c = team_centroids[i].astype(np.float64)
                    c /= np.linalg.norm(c) + 1e-8
                    dists.append(1.0 - float(np.dot(mean_emb, c)))
                if min(dists) > 0.35:
                    outlier_tids.add(tid)

        sorted_by_x = sorted(stable.items(), key=lambda kv: kv[1]["mean_x"])

        def _check_side(tid: int, stats: Dict, gap: float, side: str) -> None:
            threshold = 3.0 if tid in outlier_tids else 5.0
            if abs(stats["mean_x"]) > 20 and gap >= threshold:
                candidates.add(tid)
                if self.debug:
                    print(
                        f"[DEBUG] Track {tid} → GOALIE ({side}): "
                        f"mean_x={stats['mean_x']:.1f}m, gap={gap:.1f}m"
                        + (" [outlier jersey]" if tid in outlier_tids else "")
                    )

        if len(sorted_by_x) >= 2:
            left_tid, left_s = sorted_by_x[0]
            _, next_s = sorted_by_x[1]
            _check_side(left_tid, left_s, next_s["mean_x"] - left_s["mean_x"], "LEFT")

            right_tid, right_s = sorted_by_x[-1]
            _, prev_s = sorted_by_x[-2]
            _check_side(right_tid, right_s, right_s["mean_x"] - prev_s["mean_x"], "RIGHT")

        return candidates

    def _find_linesman_candidates(self, track_stats: Dict, exclude: Set[int]) -> Set[int]:
        """Find tracks predominantly on sidelines.

        Requires all three: near sideline (high mean_y), low lateral variance
        (stays on the line), AND high longitudinal variance (moves along the line).
        """
        candidates = set()

        for tid, stats in track_stats.items():
            if tid in exclude:
                continue

            mean_y = stats["mean_y"]
            std_y = stats["std_y"]
            std_x = stats["std_x"]

            near_sideline = abs(mean_y) > 32
            low_y_var = std_y < 5
            high_x_var = std_x > 5  # moves along touchline

            if near_sideline and low_y_var and high_x_var and stats["n_samples"] > 50:
                candidates.add(tid)
                if self.debug:
                    side = "TOP" if mean_y > 0 else "BOTTOM"
                    print(
                        f"[DEBUG] Track {tid} -> LINESMAN candidate ({side}): "
                        f"mean_y={mean_y:.1f}m, std_y={std_y:.1f}m, std_x={std_x:.1f}m"
                    )

        return candidates

    # ------------------------------------------------------------------
    # ReID-based clustering (primary path when embeddings are available)
    # ------------------------------------------------------------------

    def _has_reid_embeddings(self, store: TrajectoryStore, track_ids: List[int]) -> bool:
        """Return True if at least one observation has a reid_embedding."""
        for tid in track_ids:
            track = store.get_track(tid)
            if track and any(obs.reid_embedding is not None for obs in track.observations):
                return True
        return False

    def _has_team_probs(self, store: TrajectoryStore, track_ids: List[int]) -> bool:
        """Return True if at least one observation has team_probs stored."""
        for tid in track_ids:
            track = store.get_track(tid)
            if track and any(obs.team_probs is not None for obs in track.observations):
                return True
        return False

    def _vote_by_team_probs(
        self,
        store: TrajectoryStore,
        track_ids: List[int],
    ) -> Tuple[Dict[int, int], Optional[int]]:
        """
        Confidence-weighted majority vote from per-frame team_probs.

        For each track: ``Σ(probs × max_prob) → argmax``.
        Occluded/blurry frames have low max_prob and thus contribute
        near-zero weight, making the vote robust to transient noise.

        Returns:
            (team_assignments, referee_id)
        """
        team_assignments: Dict[int, int] = {}
        track_labels: Dict[int, int] = {}

        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                team_assignments[tid] = 0
                continue

            probs_list = [obs.team_probs for obs in track.observations if obs.team_probs is not None]
            if not probs_list:
                team_assignments[tid] = 0
                continue

            stacked = np.stack(probs_list)  # [T, 3]
            confidence = stacked.max(axis=1, keepdims=True)  # [T, 1]
            weighted = (stacked * confidence).sum(axis=0)  # [3]
            label = int(weighted.argmax())
            track_labels[tid] = label
            team_assignments[tid] = label if label != 2 else -1

        if self.debug:
            label_counts = Counter(track_labels.values())
            print(f"[DEBUG] Confidence-weighted vote: {dict(label_counts)}")

        referee_id = None
        ref_candidates = [tid for tid, l in track_labels.items() if l == 2]
        if ref_candidates:
            referee_id = self._pick_best_referee(ref_candidates, store, team_assignments)

        return team_assignments, referee_id

    def _cluster_by_reid(
        self,
        store: TrajectoryStore,
        track_ids: List[int],
    ) -> Tuple[Dict[int, int], Optional[int]]:
        """
        Cluster tracks by ReID embeddings or confidence-weighted majority vote.

        Prefers per-frame team_probs (confidence-weighted vote) when stored.
        Falls back to mean-pooled embedding clustering when probs are absent.

        Returns:
            (team_assignments, referee_id)
        """
        if not track_ids:
            return {}, None

        # Prefer confidence-weighted majority vote when team_probs are available
        if self._has_team_probs(store, track_ids):
            if self.debug:
                print("[IdentityAssigner] Using confidence-weighted majority vote (team_probs)")
            return self._vote_by_team_probs(store, track_ids)

        if self.debug:
            print("[IdentityAssigner] Falling back to mean-pool embedding clustering")

        # Mean-pool per-frame embeddings to get one vector per track
        tids_with_emb: List[int] = []
        track_embs: List[np.ndarray] = []

        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                continue
            embs = [obs.reid_embedding for obs in track.observations if obs.reid_embedding is not None]
            if embs:
                tids_with_emb.append(tid)
                track_embs.append(np.mean(embs, axis=0))

        if not tids_with_emb:
            return {tid: 0 for tid in track_ids}, None

        emb_matrix = np.stack(track_embs)  # [N, dim]

        # Get per-track cluster labels (0=home, 1=away, 2=ref)
        if self.embedder is not None:
            labels = self.embedder.cluster_teams(emb_matrix)
        elif self.siglip_embedder is not None:
            labels = self.siglip_embedder.cluster_teams(emb_matrix)
        else:
            # Fallback: cosine-normalised k-means, 3 clusters
            try:
                from sklearn.cluster import KMeans

                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
                normed = emb_matrix / norms
                km = KMeans(n_clusters=3, random_state=42, n_init=10)
                labels = km.fit_predict(normed)
                # Assign label 2 to the smallest cluster (likely referee)
                counts = np.bincount(labels, minlength=3)
                labels = np.where(labels == np.argmin(counts), 2, labels)
            except ImportError:
                labels = np.zeros(len(tids_with_emb), dtype=int)

        if self.debug:
            unique, cnts = np.unique(labels, return_counts=True)
            print(f"[DEBUG] ReID clusters: {dict(zip(unique.tolist(), cnts.tolist()))}")

        team_assignments: Dict[int, int] = {}
        referee_id: Optional[int] = None

        # Tracks without embeddings fall back to most-common non-ref team
        majority_team = (
            int(np.bincount([l for l in labels if l != 2], minlength=2).argmax()) if any(l != 2 for l in labels) else 0
        )
        for tid in track_ids:
            if tid not in tids_with_emb:
                team_assignments[tid] = majority_team

        for i, tid in enumerate(tids_with_emb):
            label = int(labels[i])
            if label == 2:
                team_assignments[tid] = -1
                if referee_id is None:
                    # Prefer the mobile, non-sideline track most likely to be referee
                    referee_id = self._pick_best_referee(
                        [tids_with_emb[j] for j, l in enumerate(labels) if l == 2],
                        store,
                        team_assignments,
                    )
            else:
                team_assignments[tid] = label

        if referee_id is not None and self.debug:
            print(f"[DEBUG] Selected referee: Track {referee_id} (ReID cluster 2)")

        return team_assignments, referee_id

    def _pick_best_referee(
        self,
        candidate_ids: List[int],
        store: TrajectoryStore,
        team_assignments: Dict[int, int],
    ) -> Optional[int]:
        """Return the most plausible referee from cluster-2 candidates."""
        best_id: Optional[int] = None
        best_mobility = -1.0

        for tid in candidate_ids:
            track = store.get_track(tid)
            if not track:
                continue
            stats = track.pitch_position_stats()
            if not stats:
                continue
            if abs(stats.get("mean_y", 0)) > 30:
                continue  # sideline — likely linesman
            mobility = stats.get("std_x", 0) + stats.get("std_y", 0)
            if mobility > best_mobility:
                best_mobility = mobility
                best_id = tid

        return best_id

    def _assign_goalie_teams(
        self,
        goalie_ids: Set[int],
        track_stats: Dict,
        team_assignments: Dict[int, int],
    ) -> Dict[int, int]:
        """Assign goalies to teams based on defensive side."""
        goalie_teams = {}

        team_0_x = []
        team_1_x = []

        for tid, team in team_assignments.items():
            if tid in track_stats:
                if team == 0:
                    team_0_x.append(track_stats[tid]["mean_x"])
                elif team == 1:
                    team_1_x.append(track_stats[tid]["mean_x"])

        team_0_avg_x = np.mean(team_0_x) if team_0_x else 0
        team_1_avg_x = np.mean(team_1_x) if team_1_x else 0

        for gid in goalie_ids:
            if gid in track_stats:
                goalie_x = track_stats[gid]["mean_x"]

                if goalie_x < 0:
                    goalie_teams[gid] = 0 if team_0_avg_x > team_1_avg_x else 1
                else:
                    goalie_teams[gid] = 0 if team_0_avg_x < team_1_avg_x else 1

                if self.debug:
                    print(f"[DEBUG] Goalie {gid} assigned to Team {goalie_teams[gid]}")

        return goalie_teams

    def _print_summary(self, results: Dict) -> None:
        """Print assignment summary."""
        role_counts: Dict[str, int] = defaultdict(int)
        team_counts: Dict[int, int] = defaultdict(int)

        for tid, info in results.items():
            role_counts[info["role"]] += 1
            if info["team"] in [0, 1]:
                team_counts[info["team"]] += 1

        print("\n[IdentityAssigner] Summary:")
        print(f"  Roles: {dict(role_counts)}")
        print(f"  Teams: Team 0: {team_counts[0]}, Team 1: {team_counts[1]}")


class PitchSlotManager:
    """
    Manage fixed 11v11 player slots on the pitch.

    Assigns tracks to persistent slots, merges fragmented tracks,
    and maintains consistent identity across the video.

    Args:
        fps: Video frame rate.
        debug: Print debug information.

    Example:
        >>> manager = PitchSlotManager(fps=30.0)
        >>> manager.initialize_from_assignments(store, assignments)
        >>> positions = manager.get_frame_positions(100)
    """

    def __init__(self, fps: float = 30.0, debug: bool = False) -> None:
        self.fps = fps
        self.debug = debug
        self.slots: Dict[str, PlayerSlot] = {}

        # Initialize 11 slots per team + referee
        for team in [0, 1]:
            for i in range(11):
                slot_key = f"T{team}_{i}"
                self.slots[slot_key] = PlayerSlot(
                    slot_id=i,
                    team=team,
                    position=(0.0, 0.0),
                    last_observed_frame=-1,
                )

        self.slots["REF"] = PlayerSlot(
            slot_id=0,
            team=-1,
            position=(0.0, 0.0),
            last_observed_frame=-1,
        )

        self.track_to_slot: Dict[int, str] = {}
        self.frame_slot_positions: Dict[int, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    def initialize_from_assignments(
        self,
        store: TrajectoryStore,
        assignments: Dict[int, Dict],
    ) -> None:
        """
        Initialize slots from identity assignments.

        Merges fragmented tracks and assigns to slots.

        Args:
            store: TrajectoryStore with all tracks.
            assignments: Role/team assignments from IdentityAssigner.
        """
        if self.debug:
            print("\n[PitchSlotManager] Initializing slots...")

        # Group by team/role
        team_tracks: Dict[int, List[Tuple[int, TrackData, Dict]]] = {0: [], 1: []}
        referee_tracks = []

        for tid, info in assignments.items():
            track = store.get_track(tid)
            if not track:
                continue

            role = info.get("role", "unknown")
            team = info.get("team", -1)

            if role == "linesman":
                continue
            if role == "referee":
                referee_tracks.append((tid, track, info))
            elif team in [0, 1]:
                team_tracks[team].append((tid, track, info))

        # Process each team
        for team in [0, 1]:
            tracks = team_tracks[team]
            merged_groups = self._merge_fragmented_tracks(tracks, store)

            merged_groups.sort(
                key=lambda g: sum(store.get_track(tid).duration_frames() for tid in g if store.get_track(tid)),
                reverse=True,
            )

            for i, group in enumerate(merged_groups[:11]):
                slot_key = f"T{team}_{i}"

                for tid in group:
                    self.track_to_slot[tid] = slot_key
                    self.slots[slot_key].assigned_track_ids.append(tid)

                # Find main track
                main_track = max(
                    (store.get_track(tid) for tid in group if store.get_track(tid)),
                    key=lambda t: t.duration_frames(),
                    default=None,
                )

                if main_track:
                    pos = main_track.mean_pitch_position()
                    if pos:
                        self.slots[slot_key].position = pos

                # Check for goalie
                for tid in group:
                    if assignments.get(tid, {}).get("role") == "goalie":
                        self.slots[slot_key].is_goalie = True
                        break

        # Assign referee
        if referee_tracks:
            tid, track, info = referee_tracks[0]
            self.track_to_slot[tid] = "REF"
            self.slots["REF"].assigned_track_ids.append(tid)

        if self.debug:
            print(f"[PitchSlotManager] Assigned {len(self.track_to_slot)} tracks to slots")

    def _merge_fragmented_tracks(
        self,
        tracks: List[Tuple[int, TrackData, Dict]],
        store: TrajectoryStore,
        max_gap_frames: int = 30,
        max_distance_px: float = 150.0,
        max_height_ratio: float = 1.5,
    ) -> List[List[int]]:
        """Merge fragmented tracks based on temporal and spatial proximity."""
        if not tracks:
            return []

        track_info = []
        for tid, track, info in tracks:
            if not track.observations:
                continue

            first_obs = track.observations[0]
            last_obs = track.observations[-1]

            first_box = first_obs.box
            last_box = last_obs.box

            track_info.append(
                {
                    "tid": tid,
                    "first_frame": first_obs.frame_idx,
                    "last_frame": last_obs.frame_idx,
                    "first_center": ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2),
                    "last_center": ((last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2),
                    "first_height": first_box[3] - first_box[1],
                    "last_height": last_box[3] - last_box[1],
                }
            )

        track_info.sort(key=lambda x: x["first_frame"])

        # Union-find
        parent = {t["tid"]: t["tid"] for t in track_info}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, curr in enumerate(track_info):
            best_match = None
            best_score = float("inf")

            for j in range(i):
                prev = track_info[j]

                gap = curr["first_frame"] - prev["last_frame"]
                if gap < 0 or gap > max_gap_frames:
                    continue

                dx = curr["first_center"][0] - prev["last_center"][0]
                dy = curr["first_center"][1] - prev["last_center"][1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > max_distance_px:
                    continue

                height_ratio = max(curr["first_height"], prev["last_height"]) / max(
                    min(curr["first_height"], prev["last_height"]), 1
                )
                if height_ratio > max_height_ratio:
                    continue

                score = gap + dist * 0.5 + (height_ratio - 1) * 50
                if score < best_score:
                    best_score = score
                    best_match = prev["tid"]

            if best_match is not None:
                union(curr["tid"], best_match)

        groups: Dict[int, List[int]] = defaultdict(list)
        for t in track_info:
            root = find(t["tid"])
            groups[root].append(t["tid"])

        return list(groups.values())

    def get_frame_positions(self, frame_idx: int) -> List[Dict]:
        """
        Get all slot positions for a frame.

        Args:
            frame_idx: Frame number.

        Returns:
            List of {slot_key, position, team, is_goalie, slot_id}.
        """
        result = []

        for slot_key, slot in self.slots.items():
            if slot.last_observed_frame >= 0:
                if frame_idx in self.frame_slot_positions and slot_key in self.frame_slot_positions[frame_idx]:
                    pos = self.frame_slot_positions[frame_idx][slot_key]
                else:
                    pos = slot.position

                result.append(
                    {
                        "slot_key": slot_key,
                        "position": pos,
                        "team": slot.team,
                        "is_goalie": slot.is_goalie,
                        "slot_id": slot.slot_id,
                    }
                )

        return result

    def build_all_frame_positions(
        self,
        store: TrajectoryStore,
        total_frames: int,
    ) -> None:
        """
        Build frame positions for all frames using smoothed track data.

        Interpolates between observations for smooth movement.

        Args:
            store: TrajectoryStore with all tracks.
            total_frames: Total number of frames in video.
        """
        if self.debug:
            print(f"\n[PitchSlotManager] Building positions for {total_frames} frames...")

        # Build lookup from smoothed data when available, otherwise raw observations
        track_frame_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}

        for tid in self.track_to_slot:
            track = store.get_track(tid)
            if not track:
                continue

            track_frame_positions[tid] = {}

            # Prefer smoothed positions if available (from Pass 2)
            if track.smoothed_positions is not None and track.smoothed_frames is not None:
                for i, frame_idx in enumerate(track.smoothed_frames):
                    pos = (track.smoothed_positions[i, 0], track.smoothed_positions[i, 1])
                    track_frame_positions[tid][int(frame_idx)] = pos
            else:
                # Fallback to raw observations
                for obs in track.observations:
                    if obs.pitch_pos is not None:
                        track_frame_positions[tid][obs.frame_idx] = obs.pitch_pos

        # For each slot, build interpolated positions across all frames
        for tid, slot_key in self.track_to_slot.items():
            if tid not in track_frame_positions:
                continue

            frame_pos = track_frame_positions[tid]
            if not frame_pos:
                continue

            frames = sorted(frame_pos.keys())
            if not frames:
                continue

            # Initialize slot position
            first_pos = frame_pos[frames[0]]
            self.slots[slot_key].position = first_pos
            self.slots[slot_key].last_observed_frame = frames[0]

            # Interpolate for all frames in range
            min_frame, max_frame = frames[0], frames[-1]

            for frame_idx in range(min_frame, max_frame + 1):
                if frame_idx in frame_pos:
                    pos = frame_pos[frame_idx]
                else:
                    # Interpolate between nearest known frames
                    prev_frames = [f for f in frames if f < frame_idx]
                    next_frames = [f for f in frames if f > frame_idx]

                    if prev_frames and next_frames:
                        prev_frame = prev_frames[-1]  # max(prev_frames)
                        next_frame = next_frames[0]  # min(next_frames)
                        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
                        prev_pos = frame_pos[prev_frame]
                        next_pos = frame_pos[next_frame]
                        pos = (
                            prev_pos[0] + t * (next_pos[0] - prev_pos[0]),
                            prev_pos[1] + t * (next_pos[1] - prev_pos[1]),
                        )
                    elif prev_frames:
                        pos = frame_pos[prev_frames[-1]]
                    elif next_frames:
                        pos = frame_pos[next_frames[0]]
                    else:
                        pos = self.slots[slot_key].position

                self.slots[slot_key].position = pos
                self.slots[slot_key].last_observed_frame = frame_idx
                self.frame_slot_positions[frame_idx][slot_key] = pos

            # Extend last known position for frames after track ends
            for frame_idx in range(max_frame + 1, total_frames):
                self.frame_slot_positions[frame_idx][slot_key] = self.slots[slot_key].position

        if self.debug:
            obs_count = sum(len(positions) for positions in self.frame_slot_positions.values())
            print(f"[PitchSlotManager] Built {obs_count} slot-frame observations")


__all__ = [
    "IdentityAssigner",
    "PitchSlotManager",
]
