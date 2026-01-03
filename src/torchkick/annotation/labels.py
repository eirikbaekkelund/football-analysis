"""
Label definitions for football/soccer annotation.

This module provides predefined label schemas for CVAT annotation projects
covering players, referees, goalkeepers, pitch lines, and re-identification.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Football player and team labels
FOOTBALL_LABELS: List[Dict[str, Any]] = [
    {
        "name": "player",
        "color": "#ff0000",
        "attributes": [
            {
                "name": "team",
                "mutable": True,
                "input_type": "select",
                "values": ["team_a", "team_b", "unknown"],
            },
            {
                "name": "jersey_number",
                "mutable": True,
                "input_type": "number",
                "values": ["0", "99"],
            },
            {
                "name": "role",
                "mutable": False,
                "input_type": "select",
                "values": ["outfield", "goalkeeper"],
            },
        ],
    },
    {
        "name": "referee",
        "color": "#ffff00",
        "attributes": [
            {
                "name": "role",
                "mutable": False,
                "input_type": "select",
                "values": ["main", "assistant", "fourth"],
            },
        ],
    },
    {
        "name": "goalkeeper",
        "color": "#00ff00",
        "attributes": [
            {
                "name": "team",
                "mutable": True,
                "input_type": "select",
                "values": ["team_a", "team_b"],
            },
            {
                "name": "jersey_number",
                "mutable": True,
                "input_type": "number",
                "values": ["0", "99"],
            },
        ],
    },
]

# Pitch line labels for calibration annotation
PITCH_LINE_LABELS: List[Dict[str, Any]] = [
    {"name": "Side line top", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Side line bottom", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Side line left", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Side line right", "color": "#00ff00", "type": "polyline", "attributes": []},
    {"name": "Middle line", "color": "#ffff00", "type": "polyline", "attributes": []},
    {"name": "Big rect. left main", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. left top", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. left bottom", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. right main", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. right top", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Big rect. right bottom", "color": "#ff00ff", "type": "polyline", "attributes": []},
    {"name": "Small rect. left main", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. left top", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. left bottom", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. right main", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. right top", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Small rect. right bottom", "color": "#00ffff", "type": "polyline", "attributes": []},
    {"name": "Circle central", "color": "#ffffff", "type": "ellipse", "attributes": []},
    {"name": "Circle left", "color": "#ffffff", "type": "ellipse", "attributes": []},
    {"name": "Circle right", "color": "#ffffff", "type": "ellipse", "attributes": []},
    {"name": "Goal left crossbar", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal left post left", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal left post right", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal right crossbar", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal right post left", "color": "#ff0000", "type": "polyline", "attributes": []},
    {"name": "Goal right post right", "color": "#ff0000", "type": "polyline", "attributes": []},
]

# Re-identification labels for person tracking
REID_LABELS: List[Dict[str, Any]] = [
    {
        "name": "person",
        "color": "#ff6600",
        "attributes": [
            {
                "name": "person_id",
                "mutable": False,
                "input_type": "text",
                "values": [""],
            },
            {
                "name": "category",
                "mutable": False,
                "input_type": "select",
                "values": ["player", "referee", "staff", "other"],
            },
            {
                "name": "team",
                "mutable": True,
                "input_type": "select",
                "values": ["team_a", "team_b", "neutral", "unknown"],
            },
            {
                "name": "jersey_color",
                "mutable": True,
                "input_type": "text",
                "values": [""],
            },
            {
                "name": "shorts_color",
                "mutable": True,
                "input_type": "text",
                "values": [""],
            },
        ],
    },
]

# Line index to name mapping for calibration
LINE_INDEX_TO_NAME: Dict[int, str] = {
    1: "Big rect. left bottom",
    2: "Big rect. left main",
    3: "Big rect. left top",
    4: "Big rect. right bottom",
    5: "Big rect. right main",
    6: "Big rect. right top",
    7: "Goal left crossbar",
    8: "Goal left post left",
    9: "Goal left post right",
    10: "Goal right crossbar",
    11: "Goal right post left",
    12: "Goal right post right",
    13: "Middle line",
    14: "Side line bottom",
    15: "Side line left",
    16: "Side line right",
    17: "Side line top",
    18: "Small rect. left bottom",
    19: "Small rect. left main",
    20: "Small rect. left top",
    21: "Small rect. right bottom",
    22: "Small rect. right main",
    23: "Small rect. right top",
}

# YOLO class ID mapping
YOLO_LABEL_MAP: Dict[str, int] = {
    "player": 0,
    "referee": 1,
    "goalkeeper": 2,
    "ball": 3,
}

__all__ = [
    "FOOTBALL_LABELS",
    "PITCH_LINE_LABELS",
    "REID_LABELS",
    "LINE_INDEX_TO_NAME",
    "YOLO_LABEL_MAP",
]
