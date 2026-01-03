"""
Player identity and team assignment.

This module provides tools for assigning player identities using
color clustering and spatial priors, including team classification,
goalie detection, and referee identification.

Example:
    >>> from torchkick.tracking import IdentityAssigner
    >>> 
    >>> assigner = IdentityAssigner(fps=30.0)
    >>> assignments = assigner.assign_roles(trajectory_store)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from torchkick.tracking.models import (
    PENALTY_AREA_X,
    PlayerSlot,
    TrackData,
)
from torchkick.tracking.trajectory import TrajectoryStore


class IdentityAssigner:
    """
    Assign player identities using color clustering and spatial priors.

    Strategy:
    1. Per-frame color voting to classify each observation
    2. Majority vote across track lifetime for final assignment
    3. Spatial analysis for goalies (penalty area) and linesmen (sidelines)
    4. Outlier detection for referee (distinct color from both teams)

    Args:
        fps: Video frame rate.
        debug: Print debug information.

    Example:
        >>> assigner = IdentityAssigner(fps=30.0)
        >>> assignments = assigner.assign_roles(store)
        >>> for tid, info in assignments.items():
        ...     print(f"Track {tid}: {info['role']} Team {info['team']}")
    """

    def __init__(self, fps: float = 30.0, debug: bool = True) -> None:
        self.fps = fps
        self.debug = debug
        self.gmm: Optional[GaussianMixture] = None
        self.team_a_cluster: Optional[int] = None
        self.team_b_cluster: Optional[int] = None
        self.team_means: Optional[np.ndarray] = None
        self.team_covs: Optional[np.ndarray] = None

    def assign_roles(
        self,
        store: TrajectoryStore,
    ) -> Dict[int, Dict]:
        """
        Assign roles and teams to all tracks.

        Args:
            store: TrajectoryStore with accumulated observations.

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
        goalie_candidates = self._find_goalie_candidates(track_stats)
        linesman_candidates = self._find_linesman_candidates(track_stats, exclude=goalie_candidates)
        remaining_ids = [
            tid for tid in track_stats.keys() if tid not in goalie_candidates and tid not in linesman_candidates
        ]

        # Cluster by color
        team_assignments, referee_id = self._cluster_by_color(store, remaining_ids)

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

    def _find_goalie_candidates(self, track_stats: Dict) -> Set[int]:
        """Find tracks predominantly in penalty areas."""
        candidates = set()

        for tid, stats in track_stats.items():
            mean_x = stats["mean_x"]
            std_x = stats["std_x"]

            in_left_penalty = mean_x < -PENALTY_AREA_X and std_x < 10
            in_right_penalty = mean_x > PENALTY_AREA_X and std_x < 10

            if (in_left_penalty or in_right_penalty) and stats["n_samples"] > 50:
                candidates.add(tid)
                if self.debug:
                    side = "LEFT" if in_left_penalty else "RIGHT"
                    print(
                        f"[DEBUG] Track {tid} -> GOALIE candidate ({side}): "
                        f"mean_x={mean_x:.1f}m, std_x={std_x:.1f}m"
                    )

        return candidates

    def _find_linesman_candidates(self, track_stats: Dict, exclude: Set[int]) -> Set[int]:
        """Find tracks predominantly on sidelines."""
        candidates = set()

        for tid, stats in track_stats.items():
            if tid in exclude:
                continue

            mean_y = stats["mean_y"]
            std_y = stats["std_y"]

            near_sideline = abs(mean_y) > 32
            low_y_var = std_y < 5

            if near_sideline and low_y_var and stats["n_samples"] > 50:
                candidates.add(tid)
                if self.debug:
                    side = "TOP" if mean_y > 0 else "BOTTOM"
                    print(
                        f"[DEBUG] Track {tid} -> LINESMAN candidate ({side}): "
                        f"mean_y={mean_y:.1f}m, std_y={std_y:.1f}m"
                    )

        return candidates

    def _cluster_by_color(
        self,
        store: TrajectoryStore,
        track_ids: List[int],
    ) -> Tuple[Dict[int, int], Optional[int]]:
        """
        Cluster tracks by color using GMM.

        Returns:
            (team_assignments, referee_id)
        """
        if not track_ids:
            return {}, None

        if not self._fit_color_gmm(store, track_ids):
            return {tid: 0 for tid in track_ids}, None

        team_assignments = {}
        track_outlier_scores = {}

        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                continue

            majority_team, confidence, outlier_score = self._vote_per_frame(track)
            team_assignments[tid] = majority_team
            track_outlier_scores[tid] = outlier_score

            if self.debug and outlier_score > 1.5:
                print(
                    f"[DEBUG] Track {tid}: team={majority_team}, " f"conf={confidence:.2f}, outlier={outlier_score:.2f}"
                )

        # Find referee (highest outlier score, not near sideline)
        referee_id = None
        best_outlier = 0.0

        for tid, outlier_score in track_outlier_scores.items():
            track = store.get_track(tid)
            if not track:
                continue

            stats = track.pitch_position_stats()
            if not stats:
                continue

            # Skip sideline tracks
            if abs(stats.get("mean_y", 0)) > 30:
                continue

            # Require mobility
            mobility = stats.get("std_x", 0) + stats.get("std_y", 0)
            if mobility < 2.0:
                continue

            if outlier_score > best_outlier:
                best_outlier = outlier_score
                referee_id = tid

        if referee_id is not None:
            team_assignments[referee_id] = -1
            if self.debug:
                print(f"[DEBUG] Selected referee: Track {referee_id} (outlier={best_outlier:.2f})")

        return team_assignments, referee_id

    def _fit_color_gmm(
        self,
        store: TrajectoryStore,
        track_ids: List[int],
    ) -> bool:
        """Fit 2-component GMM for team colors."""
        if not track_ids:
            return False

        all_features = []
        for tid in track_ids:
            track = store.get_track(tid)
            if not track:
                continue
            for obs in track.observations:
                if obs.color_feature is not None and np.linalg.norm(obs.color_feature) > 0:
                    all_features.append(obs.color_feature)

        if len(all_features) < 100:
            if self.debug:
                print(f"[DEBUG] Not enough color samples ({len(all_features)}) for GMM")
            return False

        X = np.array(all_features, dtype=np.float64)

        try:
            self.gmm = GaussianMixture(
                n_components=2,
                covariance_type="diag",
                random_state=42,
                n_init=10,
                reg_covar=1e-3,
            )
            self.gmm.fit(X)

            self.team_means = self.gmm.means_
            self.team_covs = self.gmm.covariances_

            labels = self.gmm.predict(X)
            cluster_counts = np.bincount(labels, minlength=2)

            sorted_clusters = np.argsort(cluster_counts)[::-1]
            self.team_a_cluster = sorted_clusters[0]
            self.team_b_cluster = sorted_clusters[1]

            if self.debug:
                print(f"[DEBUG] GMM fitted on {len(all_features)} color samples")
                print(f"[DEBUG] Team A: cluster {self.team_a_cluster} ({cluster_counts[self.team_a_cluster]} samples)")
                print(f"[DEBUG] Team B: cluster {self.team_b_cluster} ({cluster_counts[self.team_b_cluster]} samples)")

            return True

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] GMM fitting failed: {e}")
            return False

    def _vote_per_frame(
        self,
        track: TrackData,
    ) -> Tuple[int, float, float]:
        """
        Per-frame voting with outlier score.

        Returns:
            (majority_team, confidence, outlier_score)
        """
        if self.gmm is None:
            return 0, 0.0, 0.0

        votes = []
        outlier_scores = []

        for obs in track.observations:
            if obs.color_feature is None or np.linalg.norm(obs.color_feature) == 0:
                continue

            feat = obs.color_feature.reshape(1, -1)
            probs = self.gmm.predict_proba(feat)[0]
            label = np.argmax(probs)
            conf = probs[label]

            max_prob = max(probs)
            outlier = -np.log(max_prob + 1e-10)
            outlier_scores.append(outlier)

            team = 0 if label == self.team_a_cluster else 1
            votes.append((team, conf))

        if not votes:
            return 0, 0.0, 0.0

        team_scores = {0: 0.0, 1: 0.0}
        for team, conf in votes:
            team_scores[team] += conf

        majority_team = max(team_scores, key=team_scores.get)
        total = sum(team_scores.values())
        confidence = team_scores[majority_team] / total if total > 0 else 0.0

        avg_outlier = np.mean(outlier_scores) if outlier_scores else 0.0
        track.frame_team_votes = [v[0] for v in votes]

        return majority_team, confidence, avg_outlier

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
                    prev_frame = max(f for f in frames if f < frame_idx)
                    next_frame = min(f for f in frames if f > frame_idx)

                    if prev_frame is not None and next_frame is not None:
                        t = (frame_idx - prev_frame) / (next_frame - prev_frame)
                        prev_pos = frame_pos[prev_frame]
                        next_pos = frame_pos[next_frame]
                        pos = (
                            prev_pos[0] + t * (next_pos[0] - prev_pos[0]),
                            prev_pos[1] + t * (next_pos[1] - prev_pos[1]),
                        )
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
