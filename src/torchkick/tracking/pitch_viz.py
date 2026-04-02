"""
2D pitch visualization.

This module provides tools for visualizing player positions and
movement on a 2D pitch diagram, including trails, velocity vectors,
and spatial dominance heatmaps.

Example:
    >>> from torchkick.tracking import PitchVisualizer
    >>> 
    >>> viz = PitchVisualizer()
    >>> image = viz.draw_players(positions)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from torchkick.tracking.homography import (
    CENTER_CIRCLE_RADIUS,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    GOAL_WIDTH,
    HALF_LENGTH,
    HALF_WIDTH,
    PENALTY_AREA_DEPTH,
    PENALTY_AREA_WIDTH,
    PENALTY_SPOT_DISTANCE,
)

# Team colors
COLOR_TEAM_1 = (0, 0, 255)  # Red
COLOR_TEAM_2 = (255, 0, 0)  # Blue
COLOR_REFEREE = (0, 255, 255)  # Yellow
COLOR_UNKNOWN = (128, 128, 128)  # Gray
COLOR_PITCH = (34, 139, 34)  # Green
COLOR_LINE = (255, 255, 255)  # White


class PitchVisualizer:
    """
    Visualize player positions on a 2D pitch diagram.

    Supports player dots, movement trails, velocity vectors,
    and spatial dominance heatmaps.

    Args:
        width: Pitch image width in pixels.
        height: Pitch image height in pixels.
        margin: Margin around pitch in pixels.
        scale: Pixels per meter.
        use_gpu: Use GPU acceleration for heatmaps.

    Example:
        >>> viz = PitchVisualizer(width=1050, height=680)
        >>> positions = [(0.0, 0.0, 1, 0), (10.0, 5.0, 2, 1)]
        >>> image = viz.draw_players(positions)
    """

    def __init__(
        self,
        width: int = 1050,
        height: int = 680,
        margin: int = 50,
        scale: float = 10.0,
        use_gpu: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.margin = margin
        self.scale = scale

        self.pitch_color = COLOR_PITCH
        self.line_color = COLOR_LINE
        self.team1_color = COLOR_TEAM_1
        self.team2_color = COLOR_TEAM_2
        self.referee_color = COLOR_REFEREE
        self.unknown_color = COLOR_UNKNOWN

        self.base_pitch = self._draw_pitch()

        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self._init_gpu_grids()
        self._init_pitch_mask()

        # Temporal smoothing
        self.prev_inf_gpu = None
        self.heatmap_alpha = 0.1

    def _init_gpu_grids(self) -> None:
        """Initialize coordinate grids for GPU acceleration."""
        h_img = self.height + 2 * self.margin
        w_img = self.width + 2 * self.margin

        self.scale_factor = 0.25
        h_small = int(h_img * self.scale_factor)
        w_small = int(w_img * self.scale_factor)
        self.h_small = h_small
        self.w_small = w_small
        self.h_img = h_img
        self.w_img = w_img

        x_indices = torch.arange(w_small, device=self.device, dtype=torch.float32)
        y_indices = torch.arange(h_small, device=self.device, dtype=torch.float32)
        Y_pix, X_pix = torch.meshgrid(y_indices, x_indices, indexing="ij")

        self.X_meters = (X_pix / self.scale_factor - self.margin) / self.scale - HALF_LENGTH
        self.Y_meters = HALF_WIDTH - (Y_pix / self.scale_factor - self.margin) / self.scale

    def _init_pitch_mask(self) -> None:
        """Initialize pitch mask for heatmap blending."""
        tl = self._pitch_to_pixel(-HALF_LENGTH, HALF_WIDTH)
        br = self._pitch_to_pixel(HALF_LENGTH, -HALF_WIDTH)

        x_min = min(tl[0], br[0])
        x_max = max(tl[0], br[0])
        y_min = min(tl[1], br[1])
        y_max = max(tl[1], br[1])

        self.pitch_mask = np.zeros((self.h_img, self.w_img), dtype=np.uint8)
        cv2.rectangle(self.pitch_mask, (x_min, y_min), (x_max, y_max), 255, -1)

        self.overlay_buffer = np.zeros((self.h_img, self.w_img, 3), dtype=np.uint8)

    def _pitch_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert pitch coordinates to pixel coordinates."""
        px = int(self.margin + (x + HALF_LENGTH) * self.scale)
        py = int(self.margin + (HALF_WIDTH - y) * self.scale)
        return px, py

    def _draw_pitch(self) -> np.ndarray:
        """Draw base pitch with all markings."""
        total_width = self.width + 2 * self.margin
        total_height = self.height + 2 * self.margin
        img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # Green pitch
        cv2.rectangle(
            img,
            (self.margin, self.margin),
            (self.margin + self.width, self.margin + self.height),
            self.pitch_color,
            -1,
        )

        # Boundary
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH, HALF_WIDTH),
            self._pitch_to_pixel(HALF_LENGTH, -HALF_WIDTH),
            self.line_color,
            2,
        )

        # Halfway line
        cv2.line(
            img,
            self._pitch_to_pixel(0, HALF_WIDTH),
            self._pitch_to_pixel(0, -HALF_WIDTH),
            self.line_color,
            2,
        )

        # Center circle
        center = self._pitch_to_pixel(0, 0)
        radius = int(CENTER_CIRCLE_RADIUS * self.scale)
        cv2.circle(img, center, radius, self.line_color, 2)
        cv2.circle(img, center, 3, self.line_color, -1)

        # Penalty areas
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH, PENALTY_AREA_WIDTH / 2),
            self._pitch_to_pixel(-HALF_LENGTH + PENALTY_AREA_DEPTH, -PENALTY_AREA_WIDTH / 2),
            self.line_color,
            2,
        )
        cv2.rectangle(
            img,
            self._pitch_to_pixel(HALF_LENGTH - PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2),
            self._pitch_to_pixel(HALF_LENGTH, -PENALTY_AREA_WIDTH / 2),
            self.line_color,
            2,
        )

        # Goal areas
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH, GOAL_AREA_WIDTH / 2),
            self._pitch_to_pixel(-HALF_LENGTH + GOAL_AREA_DEPTH, -GOAL_AREA_WIDTH / 2),
            self.line_color,
            2,
        )
        cv2.rectangle(
            img,
            self._pitch_to_pixel(HALF_LENGTH - GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2),
            self._pitch_to_pixel(HALF_LENGTH, -GOAL_AREA_WIDTH / 2),
            self.line_color,
            2,
        )

        # Penalty spots
        left_spot = self._pitch_to_pixel(-HALF_LENGTH + PENALTY_SPOT_DISTANCE, 0)
        right_spot = self._pitch_to_pixel(HALF_LENGTH - PENALTY_SPOT_DISTANCE, 0)
        cv2.circle(img, left_spot, 3, self.line_color, -1)
        cv2.circle(img, right_spot, 3, self.line_color, -1)

        # Penalty arcs
        cv2.ellipse(img, left_spot, (radius, radius), 0, -53, 53, self.line_color, 2)
        cv2.ellipse(img, right_spot, (radius, radius), 180, -53, 53, self.line_color, 2)

        # Goals
        goal_depth = 2.44
        cv2.rectangle(
            img,
            self._pitch_to_pixel(-HALF_LENGTH - goal_depth, GOAL_WIDTH / 2),
            self._pitch_to_pixel(-HALF_LENGTH, -GOAL_WIDTH / 2),
            self.line_color,
            2,
        )
        cv2.rectangle(
            img,
            self._pitch_to_pixel(HALF_LENGTH, GOAL_WIDTH / 2),
            self._pitch_to_pixel(HALF_LENGTH + goal_depth, -GOAL_WIDTH / 2),
            self.line_color,
            2,
        )

        return img

    def draw_players(
        self,
        pitch_positions: List[Tuple[float, float, int, Optional[int]]],
    ) -> np.ndarray:
        """
        Draw players on the pitch.

        Args:
            pitch_positions: List of (x, y, track_id, team_id) tuples.
                x, y in meters, team_id: 0, 1, or None.

        Returns:
            BGR image with players drawn.
        """
        img = self.base_pitch.copy()

        for x, y, track_id, team_id in pitch_positions:
            if abs(x) > HALF_LENGTH + 5 or abs(y) > HALF_WIDTH + 5:
                continue

            px, py = self._pitch_to_pixel(x, y)

            if team_id == 0:
                color = self.team1_color
            elif team_id == 1:
                color = self.team2_color
            elif team_id == 2:
                color = self.referee_color
            else:
                color = self.unknown_color

            cv2.circle(img, (px, py), 8, color, -1)
            cv2.circle(img, (px, py), 8, (0, 0, 0), 2)
            cv2.putText(
                img,
                str(track_id),
                (px - 5, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return img

    def draw_with_trails(
        self,
        pitch_positions: List[Tuple[float, float, float, float, int, Optional[int]]],
        history: Dict[int, List[Tuple[float, float]]],
        trail_length: int = 30,
        draw_vectors: bool = True,
        draw_dominance: bool = True,
    ) -> np.ndarray:
        """
        Draw players with movement trails and velocity vectors.

        Args:
            pitch_positions: List of (x, y, vx, vy, id, team) tuples.
            history: Dict mapping track_id to position history.
            trail_length: Maximum trail points to show.
            draw_vectors: Draw velocity arrows.
            draw_dominance: Draw spatial dominance heatmap.

        Returns:
            BGR image with visualization.
        """
        img = self.base_pitch.copy()

        if draw_dominance:
            self._draw_heatmap(img, pitch_positions)

        self._draw_trails(img, history, trail_length)
        self._draw_players_with_vectors(img, pitch_positions, draw_vectors)

        return img

    def _draw_trails(
        self,
        img: np.ndarray,
        history: Dict[int, List[Tuple[float, float]]],
        trail_length: int,
    ) -> None:
        """Draw movement trails."""
        for track_id, positions in history.items():
            if len(positions) < 2:
                continue

            recent = positions[-trail_length:]
            for i in range(len(recent) - 1):
                alpha = (i + 1) / len(recent)
                x1, y1 = recent[i]
                x2, y2 = recent[i + 1]

                if abs(x1) > HALF_LENGTH + 5 or abs(y1) > HALF_WIDTH + 5:
                    continue
                if abs(x2) > HALF_LENGTH + 5 or abs(y2) > HALF_WIDTH + 5:
                    continue

                p1 = self._pitch_to_pixel(x1, y1)
                p2 = self._pitch_to_pixel(x2, y2)

                gray = int(100 + 155 * alpha)
                cv2.line(img, p1, p2, (gray, gray, gray), 2)

    def _draw_players_with_vectors(
        self,
        img: np.ndarray,
        pitch_positions: list,
        draw_vectors: bool,
    ) -> None:
        """Draw player dots and velocity vectors."""
        h_img, w_img = img.shape[:2]

        ARROW_LENGTH_M = 2.0

        for x, y, vx, vy, track_id, team_id in pitch_positions:
            if abs(x) > HALF_LENGTH + 5 or abs(y) > HALF_WIDTH + 5:
                continue

            px, py = self._pitch_to_pixel(x, y)
            if not (0 <= px < w_img and 0 <= py < h_img):
                continue

            if team_id == 0:
                color = self.team1_color
            elif team_id == 1:
                color = self.team2_color
            elif team_id == 2:
                color = self.referee_color
            else:
                color = self.unknown_color

            # Draw velocity arrow (fixed length, direction only)
            if draw_vectors and (abs(vx) > 0.1 or abs(vy) > 0.1):
                # Normalize velocity to get direction
                speed = np.sqrt(vx**2 + vy**2)
                dir_x = vx / speed
                dir_y = vy / speed

                # Fixed arrow length of 2 meters
                end_x = x + dir_x * ARROW_LENGTH_M
                end_y = y + dir_y * ARROW_LENGTH_M

                px_end, py_end = self._pitch_to_pixel(end_x, end_y)
                px_end = int(np.clip(px_end, 0, w_img - 1))
                py_end = int(np.clip(py_end, 0, h_img - 1))

                if px != px_end or py != py_end:
                    # Sleek arrow: thin line with small tip
                    cv2.arrowedLine(
                        img,
                        (px, py),
                        (px_end, py_end),
                        color,
                        thickness=1,
                        tipLength=0.4,
                        line_type=cv2.LINE_AA,
                    )

            # Player dot with outline
            cv2.circle(img, (px, py), 8, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 8, (30, 30, 30), 1, cv2.LINE_AA)

            # Player number (smaller, cleaner)
            cv2.putText(
                img,
                str(track_id),
                (px - 4, py + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_heatmap(
        self,
        img: np.ndarray,
        pitch_positions: list,
    ) -> None:
        """Draw spatial dominance heatmap."""
        t0_players = []
        t1_players = []

        for px, py, vx, vy, _, team_id in pitch_positions:
            if team_id == 0:
                t0_players.append((px, py, vx, vy))
            elif team_id == 1:
                t1_players.append((px, py, vx, vy))

        if not t0_players and not t1_players:
            return

        inf_t0, inf_t1 = self._compute_influence(t0_players, t1_players)

        if inf_t0 is None or inf_t1 is None:
            return

        # Normalize
        total = inf_t0 + inf_t1 + 1e-6
        inf_t0 /= total
        inf_t1 /= total

        self.overlay_buffer[:, :, 0] = (inf_t1 * 255).astype(np.uint8)
        self.overlay_buffer[:, :, 1] = 0
        self.overlay_buffer[:, :, 2] = (inf_t0 * 255).astype(np.uint8)
        self.overlay_buffer[self.pitch_mask == 0] = 0

        cv2.addWeighted(img, 0.6, self.overlay_buffer, 0.4, 0, dst=img)

    def _compute_influence(
        self,
        t0_players: List[tuple],
        t1_players: List[tuple],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute spatial influence maps for both teams."""
        zeros = torch.zeros((self.h_small, self.w_small), device=self.device)

        inf_t0 = self._compute_team_influence(t0_players) if t0_players else zeros
        inf_t1 = self._compute_team_influence(t1_players) if t1_players else zeros

        stacked = torch.stack([inf_t0, inf_t1], dim=0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(stacked, size=(self.h_img, self.w_img), mode="nearest").squeeze(0)

        if self.prev_inf_gpu is not None:
            resized = self.prev_inf_gpu * (1 - self.heatmap_alpha) + resized * self.heatmap_alpha

        self.prev_inf_gpu = resized
        result = resized.cpu().numpy()
        return result[0], result[1]

    def _compute_team_influence(self, players: List[tuple]) -> torch.Tensor:
        """Compute influence tensor for one team."""
        data = torch.tensor(players, device=self.device, dtype=torch.float32)
        px, py = data[:, 0], data[:, 1]
        vx, vy = data[:, 2], data[:, 3]

        speed = torch.sqrt(vx**2 + vy**2)
        clamped_speed = torch.clamp(speed, 0, 8.0)

        mu_x = px + vx * 0.5
        mu_y = py + vy * 0.5

        sigma_x = 4.0 + clamped_speed * 0.5
        sigma_y = 4.0 / (1 + clamped_speed * 0.1)

        angle = torch.atan2(vy, vx)
        cos_a = torch.cos(angle).view(-1, 1, 1)
        sin_a = torch.sin(angle).view(-1, 1, 1)

        mu_x = mu_x.view(-1, 1, 1)
        mu_y = mu_y.view(-1, 1, 1)
        sigma_x = sigma_x.view(-1, 1, 1)
        sigma_y = sigma_y.view(-1, 1, 1)

        X = self.X_meters.unsqueeze(0)
        Y = self.Y_meters.unsqueeze(0)

        dx = X - mu_x
        dy = Y - mu_y

        dx_rot = dx * cos_a + dy * sin_a
        dy_rot = -dx * sin_a + dy * cos_a

        gaussian = torch.exp(-(dx_rot**2 / (2 * sigma_x**2) + dy_rot**2 / (2 * sigma_y**2)))

        return gaussian.sum(dim=0)


__all__ = [
    "COLOR_TEAM_1",
    "COLOR_TEAM_2",
    "COLOR_REFEREE",
    "PitchVisualizer",
]
