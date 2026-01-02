import cv2
import numpy as np
from typing import List, Optional, Tuple
from pydantic import BaseModel
from utils.timing import timed


class BallState(BaseModel):
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float
    is_visible: bool
    in_play: bool
    track_id: int


class BallKalmanTrack:
    def __init__(self, position: Tuple[float, float], track_id: int):
        self.id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.confidence = 1.0

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kf.statePost = np.array([[position[0]], [position[1]], [0], [0]], np.float32)

    def predict(self) -> Tuple[float, float]:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_position()

    def update(self, position: Tuple[float, float], confidence: float) -> None:
        measurement = np.array([[position[0]], [position[1]]], np.float32)
        self.kf.correct(measurement)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = 0.8 * self.confidence + 0.2 * confidence

    def get_position(self) -> Tuple[float, float]:
        x = self.kf.statePost[0].item()
        y = self.kf.statePost[1].item()
        return (x, y)

    def get_velocity(self) -> Tuple[float, float]:
        vx = self.kf.statePost[2].item()
        vy = self.kf.statePost[3].item()
        return (vx, vy)


class BallTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        distance_threshold: float = 50.0,
        velocity_threshold: float = 100.0,
        occlusion_handler: bool = True,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.velocity_threshold = velocity_threshold
        self.occlusion_handler = occlusion_handler

        self.tracks: List[BallKalmanTrack] = []
        self.next_id = 1
        self.frame_id = 0
        self.primary_track_id: Optional[int] = None

        self.trajectory_buffer: List[Tuple[float, float]] = []
        self.trajectory_max_len = 60

        self.occlusion_frames = 0
        self.last_known_velocity = (0.0, 0.0)

    def _euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _validate_detection(
        self, detection: Tuple[float, float, float, float, float], player_boxes: List[List[float]]
    ) -> bool:
        x1, y1, x2, y2, conf = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        if w > 50 or h > 50:
            return False

        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False

        for pbox in player_boxes:
            px1, py1, px2, py2 = pbox[:4]
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                player_h = py2 - py1
                if cy < py1 + player_h * 0.3:
                    return True
                if cy > py1 + player_h * 0.9:
                    return True
                return False

        return True

    def _predict_occlusion_position(self) -> Optional[Tuple[float, float]]:
        if not self.trajectory_buffer or len(self.trajectory_buffer) < 2:
            return None

        vx, vy = self.last_known_velocity
        last_pos = self.trajectory_buffer[-1]

        gravity_effect = 0.5 * self.occlusion_frames
        predicted_y = last_pos[1] + vy * self.occlusion_frames + gravity_effect
        predicted_x = last_pos[0] + vx * self.occlusion_frames

        return (predicted_x, predicted_y)

    @timed
    def update(
        self, detections: List[Tuple[float, float, float, float, float]], player_boxes: List[List[float]] = None
    ) -> Optional[BallState]:
        self.frame_id += 1
        player_boxes = player_boxes or []

        for track in self.tracks:
            track.predict()

        valid_detections = []
        for det in detections:
            if self._validate_detection(det, player_boxes):
                x1, y1, x2, y2, conf = det
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                valid_detections.append((cx, cy, conf))

        matched_tracks = set()
        matched_dets = set()

        if self.primary_track_id is not None:
            primary_track = next((t for t in self.tracks if t.id == self.primary_track_id), None)
            if primary_track and valid_detections:
                pred_pos = primary_track.get_position()
                pred_vel = primary_track.get_velocity()

                best_det_idx = None
                best_score = float('inf')

                for i, (cx, cy, conf) in enumerate(valid_detections):
                    dist = self._euclidean_distance(pred_pos, (cx, cy))

                    expected_x = pred_pos[0] + pred_vel[0]
                    expected_y = pred_pos[1] + pred_vel[1]
                    vel_dist = self._euclidean_distance((expected_x, expected_y), (cx, cy))

                    score = dist + 0.3 * vel_dist - 10 * conf

                    if dist < self.distance_threshold * 2 and score < best_score:
                        best_score = score
                        best_det_idx = i

                if best_det_idx is not None:
                    cx, cy, conf = valid_detections[best_det_idx]
                    primary_track.update((cx, cy), conf)
                    matched_tracks.add(primary_track.id)
                    matched_dets.add(best_det_idx)
                    self.occlusion_frames = 0

        for track in self.tracks:
            if track.id in matched_tracks:
                continue

            pred_pos = track.get_position()
            best_det_idx = None
            best_dist = self.distance_threshold

            for i, (cx, cy, conf) in enumerate(valid_detections):
                if i in matched_dets:
                    continue
                dist = self._euclidean_distance(pred_pos, (cx, cy))
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = i

            if best_det_idx is not None:
                cx, cy, conf = valid_detections[best_det_idx]
                track.update((cx, cy), conf)
                matched_tracks.add(track.id)
                matched_dets.add(best_det_idx)

        for i, (cx, cy, conf) in enumerate(valid_detections):
            if i in matched_dets:
                continue

            new_track = BallKalmanTrack((cx, cy), self.next_id)
            new_track.confidence = conf
            self.tracks.append(new_track)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        if self.primary_track_id is not None:
            primary_exists = any(t.id == self.primary_track_id for t in self.tracks)
            if not primary_exists:
                self.primary_track_id = None

        if self.primary_track_id is None and self.tracks:
            best_track = max(self.tracks, key=lambda t: t.hits * t.confidence - t.time_since_update)
            if best_track.hits >= self.min_hits:
                self.primary_track_id = best_track.id

        if self.primary_track_id is not None:
            primary = next(t for t in self.tracks if t.id == self.primary_track_id)
            pos = primary.get_position()
            vel = primary.get_velocity()

            self.trajectory_buffer.append(pos)
            if len(self.trajectory_buffer) > self.trajectory_max_len:
                self.trajectory_buffer.pop(0)

            if primary.time_since_update > 0:
                self.occlusion_frames += 1
                self.last_known_velocity = vel
            else:
                self.occlusion_frames = 0

            is_visible = primary.time_since_update == 0

            speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
            in_play = speed < self.velocity_threshold

            return BallState(
                position=pos,
                velocity=vel,
                confidence=primary.confidence,
                is_visible=is_visible,
                in_play=in_play,
                track_id=primary.id,
            )

        if self.occlusion_handler and self.trajectory_buffer:
            self.occlusion_frames += 1
            predicted = self._predict_occlusion_position()
            if predicted:
                return BallState(
                    position=predicted,
                    velocity=self.last_known_velocity,
                    confidence=max(0.1, 1.0 - self.occlusion_frames * 0.05),
                    is_visible=False,
                    in_play=True,
                    track_id=-1,
                )

        return None

    def get_trajectory(self) -> List[Tuple[float, float]]:
        return list(self.trajectory_buffer)

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0
        self.primary_track_id = None
        self.trajectory_buffer = []
        self.occlusion_frames = 0
