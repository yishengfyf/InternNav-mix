from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_MP3D_LANDMARKS = [
    "chair",
    "door",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "window",
    "sofa",
    "bed",
    "curtain",
    "chest_of_drawers",
    "plant",
    "sink",
    "stairs",
    "toilet",
    "stool",
    "towel",
    "mirror",
    "tv_monitor",
    "shower",
    "column",
    "bathtub",
    "counter",
    "fireplace",
    "lighting",
    "railing",
    "shelving",
    "blinds",
    "gym_equipment",
    "seating",
    "board_panel",
    "furniture",
    "appliances",
    "clothes",
    "objects",
]

_ROOM_TERMS = [
    "bathroom",
    "bedroom",
    "closet",
    "corridor",
    "dining area",
    "dining room",
    "entryway",
    "hall",
    "hallway",
    "kitchen",
    "living area",
    "living room",
    "office",
    "patio",
    "balcony",
    "room",
]

_ALIASES = {
    "appliance": "appliances",
    "appliances": "appliances",
    "armchair": "chair",
    "arch": "door",
    "arched doorway": "door",
    "arched entry": "entryway",
    "archway": "door",
    "balcony": "balcony",
    "bath": "bathtub",
    "bath room": "bathroom",
    "bathroom": "bathroom",
    "bathtub": "bathtub",
    "bed": "bed",
    "bed room": "bedroom",
    "bedroom": "bedroom",
    "billiard room": "room",
    "billiard table": "table",
    "blind": "blinds",
    "blinds": "blinds",
    "book shelf": "shelving",
    "bookshelf": "shelving",
    "cabinet": "cabinet",
    "chair": "chair",
    "chairs": "chair",
    "closet": "closet",
    "column": "column",
    "corridor": "corridor",
    "couch": "sofa",
    "couches": "sofa",
    "coutch": "sofa",
    "counter": "counter",
    "curtain": "curtain",
    "curtains": "curtain",
    "cushion": "cushion",
    "dinning room": "dining room",
    "dining area": "dining area",
    "dining room": "dining room",
    "door": "door",
    "doors": "door",
    "doorway": "door",
    "drawer": "chest_of_drawers",
    "drawers": "chest_of_drawers",
    "dresser": "chest_of_drawers",
    "entrance": "entryway",
    "entryway": "entryway",
    "fireplace": "fireplace",
    "foyer": "entryway",
    "gym equipment": "gym_equipment",
    "hall": "hall",
    "hallway": "hallway",
    "island counter": "counter",
    "kitchen": "kitchen",
    "lamp": "lighting",
    "light": "lighting",
    "lighting": "lighting",
    "living area": "living area",
    "living room": "living room",
    "mirror": "mirror",
    "office": "office",
    "painting": "picture",
    "patio": "patio",
    "picture": "picture",
    "plant": "plant",
    "plants": "plant",
    "pool table": "table",
    "railing": "railing",
    "room": "room",
    "seat": "seating",
    "seating": "seating",
    "shelf": "shelving",
    "shelves": "shelving",
    "shelving": "shelving",
    "shower": "shower",
    "sink": "sink",
    "sofa": "sofa",
    "stair": "stairs",
    "staircase": "stairs",
    "stairs": "stairs",
    "stool": "stool",
    "table": "table",
    "tables": "table",
    "television": "tv_monitor",
    "toilet": "toilet",
    "towel": "towel",
    "tv": "tv_monitor",
    "window": "window",
    "windows": "window",
}

_ROOM_TERM_SET = set(_ROOM_TERMS)
_OBJECT_TERM_SET = set(_MP3D_LANDMARKS)

_QUERY_LABELS = {
    "board_panel": "board panel",
    "chest_of_drawers": "chest of drawers",
    "gym_equipment": "gym equipment",
    "tv_monitor": "television",
}

_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "there is a {} in the scene.",
    "there is the {} in the scene.",
]


class VLMapSemanticShadow:
    """CLIP-based instruction landmark shadow logger for Habitat VLN eval."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.enabled = bool(self.config.get("semantic_match_enable", False))
        self.shadow_only = bool(self.config.get("semantic_match_shadow_only", True))
        self.backend = str(self.config.get("semantic_match_backend", "auto")).lower()
        self.device_name = str(self.config.get("semantic_match_device", "cpu"))
        self.clip_model_name = str(self.config.get("semantic_match_clip_model", "ViT-B/32"))
        self.longclip_model_path = str(
            self.config.get("semantic_match_model_path")
            or self.config.get("semantic_match_longclip_model_path")
            or "checkpoints/clip-long/longclip-B.pt"
        )
        self.score_threshold = float(self.config.get("semantic_match_score_threshold", 0.20))
        threshold_values = [self.score_threshold]
        threshold_values.extend(
            self._parse_float_list(
                self.config.get(
                    "semantic_match_score_thresholds",
                    [0.25, 0.27, 0.29, 0.31],
                )
            )
        )
        self.score_thresholds = sorted({round(float(value), 6) for value in threshold_values})
        self.relative_z_threshold = float(self.config.get("semantic_match_relative_z_threshold", 0.5))
        self.margin_threshold = float(self.config.get("semantic_match_margin_threshold", 0.01))
        self.top_k = max(1, int(self.config.get("semantic_match_top_k", 3)))
        self.max_terms = max(1, int(self.config.get("semantic_match_max_terms", 8)))
        self.use_templates = bool(self.config.get("semantic_match_use_templates", True))
        self.save_rgb = bool(self.config.get("semantic_match_save_rgb", False))
        self.verbose = bool(self.config.get("verbose", True))
        self.strict = bool(self.config.get("semantic_match_strict_import", False))
        self.debug_dir: Optional[str] = None

        self._clip = None
        self._torch = None
        self._tokenize = None
        self._model = None
        self._preprocess = None
        self._device = None
        self._active_backend: Optional[str] = None
        self._disabled_reason: Optional[str] = None
        self._init_error_logged = False
        self._episode_meta: Dict[str, Any] = {}
        self._landmarks: List[Dict[str, Any]] = []
        self._text_features = None
        self._rolling_max: Dict[str, float] = {}
        self._first_seen_step: Dict[str, Optional[int]] = {}
        self._rank1_counts: Dict[str, int] = {}
        self._rank1_first_step: Dict[str, Optional[int]] = {}
        self._rank1_confident_counts: Dict[str, int] = {}
        self._rank1_confident_first_step: Dict[str, Optional[int]] = {}
        self._relative_counts: Dict[str, int] = {}
        self._relative_first_step: Dict[str, Optional[int]] = {}
        self._top_sequence: List[str] = []
        self._top_score_values: List[float] = []
        self._top_margin_values: List[float] = []
        self._event_count = 0

    def set_debug_dir(self, debug_dir: Optional[str]) -> None:
        self.debug_dir = debug_dir

    def reset_episode(
        self,
        *,
        instruction: str,
        scene_id: str,
        episode_id: int,
        episode_index: int,
        episode_count: int,
    ) -> Dict[str, Any]:
        self._episode_meta = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "episode_count": episode_count,
            "instruction": instruction,
        }
        self._landmarks = self._extract_landmarks(instruction) if self.enabled else []
        self._text_features = None
        self._rolling_max = {item["term"]: float("-inf") for item in self._landmarks}
        self._first_seen_step = {item["term"]: None for item in self._landmarks}
        self._rank1_counts = {item["term"]: 0 for item in self._landmarks}
        self._rank1_first_step = {item["term"]: None for item in self._landmarks}
        self._rank1_confident_counts = {item["term"]: 0 for item in self._landmarks}
        self._rank1_confident_first_step = {item["term"]: None for item in self._landmarks}
        self._relative_counts = {item["term"]: 0 for item in self._landmarks}
        self._relative_first_step = {item["term"]: None for item in self._landmarks}
        self._top_sequence = []
        self._top_score_values = []
        self._top_margin_values = []
        self._event_count = 0

        event = {
            "event_type": "episode_start",
            **self._episode_meta,
            "enabled": bool(self.enabled),
            "shadow_only": bool(self.shadow_only),
            "landmarks": self._landmarks,
            "landmark_terms": [item["term"] for item in self._landmarks],
            "disabled_reason": self._disabled_reason,
        }
        self._write_event(event)
        if self.verbose and self.enabled:
            terms = [item["term"] for item in self._landmarks]
            print(f"[VLMapSemantic] episode landmarks: {terms}")
        return event

    def match_observation(
        self,
        rgb: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        base_event = {
            "event_type": "semantic_match",
            **self._episode_meta,
            **context,
            "enabled": bool(self.enabled),
            "shadow_only": bool(self.shadow_only),
            "landmarks": self._landmarks,
            "landmark_terms": [item["term"] for item in self._landmarks],
        }

        if not self.enabled:
            base_event["status"] = "disabled"
            return base_event
        if not self._landmarks:
            base_event["status"] = "no_landmarks"
            self._write_event(base_event)
            return base_event
        if not self._ensure_model():
            base_event["status"] = "model_unavailable"
            base_event["disabled_reason"] = self._disabled_reason
            self._write_event(base_event)
            return base_event
        if not self._ensure_text_features():
            base_event["status"] = "text_feature_unavailable"
            base_event["disabled_reason"] = self._disabled_reason
            self._write_event(base_event)
            return base_event

        try:
            image_features = self._encode_image(rgb)
            scores_arr = (image_features @ self._text_features.T).reshape(-1).detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover - depends on optional CLIP runtime
            if self.strict:
                raise
            self._disabled_reason = f"CLIP image scoring failed: {exc}"
            base_event["status"] = "score_error"
            base_event["disabled_reason"] = self._disabled_reason
            self._write_event(base_event)
            return base_event

        order = np.argsort(scores_arr)[::-1]
        second_score = float(scores_arr[order[1]]) if len(order) > 1 else None
        top_idx = int(order[0])
        top_score = float(scores_arr[top_idx])
        top_term = self._landmarks[top_idx]["term"]
        top_margin_to_second = None if second_score is None else float(top_score - second_score)
        score_mean = float(np.mean(scores_arr))
        score_std = float(np.std(scores_arr))
        rank1_confident = top_margin_to_second is None or top_margin_to_second >= self.margin_threshold
        threshold_hits = []
        threshold_hits_by_threshold: Dict[str, List[str]] = {
            self._threshold_key(threshold): [] for threshold in self.score_thresholds
        }
        relative_hits = []
        score_items = []
        eval_step = context.get("step_id")
        try:
            eval_step_int = int(eval_step) if eval_step is not None else None
        except (TypeError, ValueError):
            eval_step_int = None

        self._rank1_counts[top_term] = self._rank1_counts.get(top_term, 0) + 1
        if self._rank1_first_step.get(top_term) is None:
            self._rank1_first_step[top_term] = eval_step_int
        if rank1_confident:
            self._rank1_confident_counts[top_term] = self._rank1_confident_counts.get(top_term, 0) + 1
            if self._rank1_confident_first_step.get(top_term) is None:
                self._rank1_confident_first_step[top_term] = eval_step_int
        self._top_sequence.append(top_term)
        self._top_score_values.append(top_score)
        if top_margin_to_second is not None:
            self._top_margin_values.append(top_margin_to_second)

        for rank, idx in enumerate(order, start=1):
            idx = int(idx)
            landmark = self._landmarks[idx]
            term = landmark["term"]
            score = float(scores_arr[idx])
            score_z = float((score - score_mean) / score_std) if score_std > 1e-12 else 0.0
            if score > self._rolling_max.get(term, float("-inf")):
                self._rolling_max[term] = score
            seen = score >= self.score_threshold
            if seen:
                threshold_hits.append(term)
                if self._first_seen_step.get(term) is None:
                    self._first_seen_step[term] = eval_step_int
            for threshold in self.score_thresholds:
                if score >= threshold:
                    threshold_hits_by_threshold[self._threshold_key(threshold)].append(term)
            relative_seen = bool(score_z >= self.relative_z_threshold or (len(order) == 1 and rank == 1))
            if relative_seen:
                relative_hits.append(term)
                self._relative_counts[term] = self._relative_counts.get(term, 0) + 1
                if self._relative_first_step.get(term) is None:
                    self._relative_first_step[term] = eval_step_int
            item = {
                "term": term,
                "term_type": landmark.get("term_type", self._term_type(term)),
                "query": landmark["query"],
                "matched_phrase": landmark["matched_phrase"],
                "score": score,
                "score_z": score_z,
                "rank": int(rank),
                "is_rank1": bool(rank == 1),
                "seen": bool(seen),
                "relative_seen": bool(relative_seen),
                "rolling_max": float(self._rolling_max.get(term, score)),
                "margin_to_top": float(top_score - score),
            }
            if rank == 1 and second_score is not None:
                item["top_margin_to_second"] = top_margin_to_second
                item["rank1_confident"] = bool(rank1_confident)
            score_items.append(item)

        top_terms = [self._landmarks[int(idx)]["term"] for idx in order[: self.top_k]]
        event = {
            **base_event,
            "status": "ok",
            "score_threshold": float(self.score_threshold),
            "score_thresholds": self.score_thresholds,
            "clip_model": self.clip_model_name,
            "backend": self._active_backend,
            "device": str(self._device),
            "top_match": top_term,
            "top_score": top_score,
            "rank1_term": top_term,
            "rank1_score": top_score,
            "score_mean": score_mean,
            "score_std": score_std,
            "top_margin_to_second": top_margin_to_second,
            "rank1_margin_to_second": top_margin_to_second,
            "margin_threshold": float(self.margin_threshold),
            "rank1_confident": bool(rank1_confident),
            "relative_z_threshold": float(self.relative_z_threshold),
            "top_terms": top_terms,
            "threshold_hits": threshold_hits,
            "threshold_hits_by_threshold": threshold_hits_by_threshold,
            "relative_hits": relative_hits,
            "scores": score_items,
            "rolling_max_by_term": {
                key: None if value == float("-inf") else float(value)
                for key, value in self._rolling_max.items()
            },
            "first_seen_step_by_term": self._first_seen_step,
        }
        if self.save_rgb:
            image_path = self._save_rgb(rgb, context)
            if image_path:
                event["rgb_debug_path"] = image_path
        self._event_count += 1
        self._write_event(event)
        if self.verbose:
            print(
                "[VLMapSemantic] "
                f"step={context.get('step_id')} top={top_term}:{top_score:.3f} "
                f"margin={0.0 if top_margin_to_second is None else top_margin_to_second:.3f} "
                f"rank1_conf={rank1_confident} hits={threshold_hits}"
            )
        return event

    def finish_episode(self, *, metrics: Optional[Dict[str, Any]] = None, steps: Optional[int] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        metrics = dict(metrics or {})
        terms = [item["term"] for item in self._landmarks]
        term_type_by_term = {
            item["term"]: item.get("term_type", self._term_type(item["term"]))
            for item in self._landmarks
        }
        room_terms = [term for term in terms if term_type_by_term.get(term) == "room"]
        object_terms = [term for term in terms if term_type_by_term.get(term) == "object"]
        max_score_by_term = {
            key: None if value == float("-inf") else float(value)
            for key, value in self._rolling_max.items()
        }
        seen_terms = [
            term
            for term, score in max_score_by_term.items()
            if score is not None and score >= self.score_threshold
        ]
        seen_by_threshold = {}
        coverage_by_threshold = {}
        room_coverage_by_threshold = {}
        object_coverage_by_threshold = {}
        for threshold in self.score_thresholds:
            key = self._threshold_key(threshold)
            seen_at_threshold = [
                term
                for term, score in max_score_by_term.items()
                if score is not None and score >= threshold
            ]
            seen_by_threshold[key] = seen_at_threshold
            coverage_by_threshold[key] = self._coverage(terms, seen_at_threshold)
            room_coverage_by_threshold[key] = self._coverage(room_terms, seen_at_threshold)
            object_coverage_by_threshold[key] = self._coverage(object_terms, seen_at_threshold)

        rank1_terms = [term for term, count in self._rank1_counts.items() if count > 0]
        rank1_confident_terms = [
            term for term, count in self._rank1_confident_counts.items() if count > 0
        ]
        relative_terms = [term for term, count in self._relative_counts.items() if count > 0]
        first_seen_values = [
            step for step in self._first_seen_step.values() if step is not None
        ]
        rank1_first_seen_values = [
            step for step in self._rank1_first_step.values() if step is not None
        ]
        rank1_confident_first_seen_values = [
            step for step in self._rank1_confident_first_step.values() if step is not None
        ]
        relative_first_seen_values = [
            step for step in self._relative_first_step.values() if step is not None
        ]
        top1_stability = self._top1_stability()
        top1_entropy = self._top1_entropy()
        summary = {
            "event_type": "episode_summary",
            **self._episode_meta,
            "enabled": bool(self.enabled),
            "shadow_only": bool(self.shadow_only),
            "steps": steps,
            "success": metrics.get("success"),
            "spl": metrics.get("spl"),
            "os": metrics.get("oracle_success", metrics.get("os")),
            "ne": metrics.get("distance_to_goal", metrics.get("ne")),
            "landmark_terms": terms,
            "landmark_type_by_term": term_type_by_term,
            "room_terms": room_terms,
            "object_terms": object_terms,
            "landmark_count": len(terms),
            "seen_terms": seen_terms,
            "seen_count": len(seen_terms),
            "coverage": (len(seen_terms) / len(terms)) if terms else 0.0,
            "seen_by_threshold": seen_by_threshold,
            "coverage_by_threshold": coverage_by_threshold,
            "room_coverage_by_threshold": room_coverage_by_threshold,
            "object_coverage_by_threshold": object_coverage_by_threshold,
            "first_seen_step": min(first_seen_values) if first_seen_values else None,
            "max_score_by_term": max_score_by_term,
            "first_seen_step_by_term": self._first_seen_step,
            "rank1_terms": rank1_terms,
            "rank1_count": len(rank1_terms),
            "rank1_coverage": self._coverage(terms, rank1_terms),
            "rank1_room_coverage": self._coverage(room_terms, rank1_terms),
            "rank1_object_coverage": self._coverage(object_terms, rank1_terms),
            "rank1_first_seen_step": (
                min(rank1_first_seen_values) if rank1_first_seen_values else None
            ),
            "rank1_first_seen_step_by_term": self._rank1_first_step,
            "rank1_counts_by_term": self._rank1_counts,
            "rank1_confident_terms": rank1_confident_terms,
            "rank1_confident_count": len(rank1_confident_terms),
            "rank1_confident_coverage": self._coverage(terms, rank1_confident_terms),
            "rank1_confident_room_coverage": self._coverage(room_terms, rank1_confident_terms),
            "rank1_confident_object_coverage": self._coverage(object_terms, rank1_confident_terms),
            "rank1_confident_first_seen_step": (
                min(rank1_confident_first_seen_values)
                if rank1_confident_first_seen_values
                else None
            ),
            "rank1_confident_counts_by_term": self._rank1_confident_counts,
            "relative_terms": relative_terms,
            "relative_count": len(relative_terms),
            "relative_coverage": self._coverage(terms, relative_terms),
            "relative_room_coverage": self._coverage(room_terms, relative_terms),
            "relative_object_coverage": self._coverage(object_terms, relative_terms),
            "relative_first_seen_step": (
                min(relative_first_seen_values) if relative_first_seen_values else None
            ),
            "relative_counts_by_term": self._relative_counts,
            "semantic_top_sequence": self._top_sequence,
            "mean_top_score": self._mean(self._top_score_values),
            "max_top_score": max(self._top_score_values) if self._top_score_values else None,
            "mean_top_margin": self._mean(self._top_margin_values),
            "max_top_margin": max(self._top_margin_values) if self._top_margin_values else None,
            "top1_stability": top1_stability,
            "top1_diversity": len(set(self._top_sequence)),
            "top1_entropy": top1_entropy,
            "top1_transition_count": self._top1_transition_count(),
            "semantic_event_count": int(self._event_count),
            "disabled_reason": self._disabled_reason,
        }
        for threshold_key, coverage in coverage_by_threshold.items():
            summary[f"coverage_at_{threshold_key.replace('.', '_')}"] = coverage
        self._write_summary(summary)
        return summary

    def _extract_landmarks(self, instruction: str) -> List[Dict[str, Any]]:
        text = f" {instruction.lower()} "
        candidates: List[Tuple[int, str, str]] = []
        alias_items = dict(_ALIASES)
        for term in _MP3D_LANDMARKS:
            alias_items.setdefault(term.replace("_", " "), term)
        for term in _ROOM_TERMS:
            alias_items.setdefault(term, term)

        for phrase, canonical in alias_items.items():
            pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
            match = re.search(pattern, text)
            if match:
                candidates.append((match.start(), phrase, canonical))

        candidates.sort(key=lambda item: (item[0], -len(item[1])))
        specific_rooms = set(_ROOM_TERMS) - {"room"}
        has_specific_room = any(canonical in specific_rooms for _, _, canonical in candidates)
        seen = set()
        landmarks = []
        for _, phrase, canonical in candidates:
            if canonical == "room" and has_specific_room:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            query = _QUERY_LABELS.get(canonical, canonical.replace("_", " "))
            landmarks.append(
                {
                    "term": canonical,
                    "term_type": self._term_type(canonical),
                    "query": query,
                    "matched_phrase": phrase,
                }
            )
            if len(landmarks) >= self.max_terms:
                break
        return landmarks

    def _term_type(self, term: str) -> str:
        if term in _ROOM_TERM_SET:
            return "room"
        if term in _OBJECT_TERM_SET:
            return "object"
        return "other"

    def _parse_float_list(self, raw_value: Any) -> List[float]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            values = [item.strip() for item in raw_value.split(",")]
        elif isinstance(raw_value, (list, tuple, set)):
            values = list(raw_value)
        else:
            values = [raw_value]
        parsed = []
        for value in values:
            if value in ("", None):
                continue
            try:
                parsed.append(float(value))
            except (TypeError, ValueError):
                continue
        return parsed

    def _threshold_key(self, threshold: float) -> str:
        return f"{float(threshold):.2f}"

    def _coverage(self, terms: List[str], seen_terms: List[str]) -> float:
        if not terms:
            return 0.0
        return len(set(terms) & set(seen_terms)) / len(set(terms))

    def _mean(self, values: List[float]) -> Optional[float]:
        return float(sum(values) / len(values)) if values else None

    def _top1_stability(self) -> Optional[float]:
        if not self._top_sequence:
            return None
        counts: Dict[str, int] = {}
        for term in self._top_sequence:
            counts[term] = counts.get(term, 0) + 1
        return max(counts.values()) / len(self._top_sequence)

    def _top1_entropy(self) -> Optional[float]:
        if not self._top_sequence:
            return None
        counts: Dict[str, int] = {}
        for term in self._top_sequence:
            counts[term] = counts.get(term, 0) + 1
        total = float(len(self._top_sequence))
        entropy = 0.0
        for count in counts.values():
            prob = count / total
            entropy -= prob * math.log(prob + 1e-12)
        return float(entropy)

    def _top1_transition_count(self) -> int:
        if len(self._top_sequence) < 2:
            return 0
        return sum(
            1
            for previous, current in zip(self._top_sequence[:-1], self._top_sequence[1:])
            if previous != current
        )

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._disabled_reason:
            self._write_init_error_once()
            return False

        backend_errors = []
        if self.backend in ("auto", "longclip", "clip_long", "clip-long"):
            loaded, error = self._try_load_longclip()
            if loaded:
                return True
            backend_errors.append(error)
            if self.backend in ("longclip", "clip_long", "clip-long"):
                self._disabled_reason = error
                self._write_init_error_once()
                return False

        if self.backend in ("auto", "openai_clip", "clip"):
            loaded, error = self._try_load_openai_clip()
            if loaded:
                return True
            backend_errors.append(error)

        self._disabled_reason = "; ".join(str(item) for item in backend_errors if item) or (
            f"unsupported semantic_match_backend={self.backend}"
        )
        self._write_init_error_once()
        return False

    def _resolve_longclip_model_path(self) -> Optional[str]:
        raw_path = Path(os.path.expanduser(self.longclip_model_path))
        candidates = [raw_path]
        if not raw_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            candidates.append(project_root / raw_path)
            candidates.append(
                project_root
                / "internnav"
                / "model"
                / "basemodel"
                / "LongCLIP"
                / "checkpoints"
                / raw_path.name
            )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _try_load_longclip(self) -> tuple[bool, str]:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            if self.strict:
                raise
            return False, f"torch import failed for LongCLIP: {exc}"

        if self.device_name.startswith("cuda") and torch.cuda.is_available():
            device = self.device_name
        else:
            device = "cpu"

        model_path = self._resolve_longclip_model_path()
        if model_path is None:
            return False, f"LongCLIP checkpoint not found: {self.longclip_model_path}"

        try:
            from internnav.model.basemodel.LongCLIP.model import longclip  # type: ignore

            model, preprocess = longclip.load(model_path, device=device)
            model.eval()
        except Exception as exc:  # pragma: no cover - optional dependency/weights
            if self.strict:
                raise
            return False, f"LongCLIP load failed: {exc}"

        self._clip = longclip
        self._torch = torch
        self._tokenize = longclip.tokenize
        self._model = model
        self._preprocess = preprocess
        self._device = device
        self._active_backend = "longclip"
        if self.verbose:
            print(f"[VLMapSemantic] loaded LongCLIP from {model_path} on {device}")
        return True, ""

    def _try_load_openai_clip(self) -> tuple[bool, str]:
        try:
            import clip  # type: ignore
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            if self.strict:
                raise
            return False, f"OpenAI CLIP import failed: {exc}"

        if self.device_name.startswith("cuda") and torch.cuda.is_available():
            device = self.device_name
        else:
            device = "cpu"
        try:
            model, preprocess = clip.load(self.clip_model_name, device=device, jit=False)
            model.eval()
        except Exception as exc:  # pragma: no cover - optional dependency/weights
            if self.strict:
                raise
            return False, f"OpenAI CLIP load failed: {exc}"

        self._clip = clip
        self._torch = torch
        self._tokenize = clip.tokenize
        self._model = model
        self._preprocess = preprocess
        self._device = device
        self._active_backend = "openai_clip"
        if self.verbose:
            print(f"[VLMapSemantic] loaded OpenAI CLIP {self.clip_model_name} on {device}")
        return True, ""

    def _ensure_text_features(self) -> bool:
        if self._text_features is not None:
            return True
        if not self._landmarks:
            return False
        if self._model is None or self._tokenize is None or self._torch is None:
            return False
        queries = [item["query"] for item in self._landmarks]
        prompts = []
        prompt_groups = []
        for query in queries:
            if self.use_templates:
                group = [template.format(query) for template in _TEMPLATES]
            else:
                group = [query]
            prompt_groups.append(len(group))
            prompts.extend(group)
        try:
            tokens = self._tokenize(prompts).to(self._device)
            with self._torch.no_grad():
                feats = self._model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks = []
            offset = 0
            for group_size in prompt_groups:
                chunk = feats[offset : offset + group_size].mean(dim=0, keepdim=True)
                chunk = chunk / chunk.norm(dim=-1, keepdim=True)
                chunks.append(chunk)
                offset += group_size
            self._text_features = self._torch.cat(chunks, dim=0)
        except Exception as exc:  # pragma: no cover - optional dependency runtime
            if self.strict:
                raise
            self._disabled_reason = f"CLIP text encoding failed: {exc}"
            self._write_init_error_once()
            return False
        return True

    def _encode_image(self, rgb: Any):
        if self._torch is None or self._preprocess is None or self._model is None:
            raise RuntimeError("CLIP model is not loaded")
        from PIL import Image

        if isinstance(rgb, Image.Image):
            image = rgb.convert("RGB")
        else:
            arr = np.asarray(rgb)
            if arr.ndim != 3:
                raise ValueError(f"Expected RGB image with 3 dims, got shape {arr.shape}")
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_image(tensor).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    def _write_event(self, event: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        with open(os.path.join(self.debug_dir, "semantic_events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        with open(os.path.join(self.debug_dir, "semantic_episode_summary.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(self._jsonable(summary), ensure_ascii=False) + "\n")

    def _write_init_error_once(self) -> None:
        if self._init_error_logged:
            return
        self._init_error_logged = True
        event = {
            "event_type": "semantic_init_error",
            **self._episode_meta,
            "enabled": bool(self.enabled),
            "disabled_reason": self._disabled_reason,
        }
        self._write_event(event)
        if self.verbose:
            print(f"[VLMapSemantic] disabled: {self._disabled_reason}")

    def _save_rgb(self, rgb: Any, context: Dict[str, Any]) -> Optional[str]:
        if not self.debug_dir:
            return None
        try:
            from PIL import Image

            if isinstance(rgb, Image.Image):
                image = rgb.convert("RGB")
            else:
                arr = np.asarray(rgb)
                if arr.ndim != 3:
                    return None
                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            rgb_dir = os.path.join(self.debug_dir, "semantic_rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            scene_id = str(self._episode_meta.get("scene_id", "scene")).replace(os.sep, "_")
            episode_id = self._episode_meta.get("episode_id", "unknown")
            step_id = context.get("step_id", 0)
            path = os.path.join(rgb_dir, f"{scene_id}_ep{episode_id}_step{int(step_id):05d}.jpg")
            image.save(path)
            return path
        except Exception:
            return None

    def _jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        return value
