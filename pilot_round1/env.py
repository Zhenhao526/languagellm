"""A visual reference world with no text or class labels in its observations.

The integer metadata returned by ``sample`` is for reward calculation and
offline analysis only. Agents should receive only their designated images.
Target and candidate images are independently rendered, including when they
depict the same class. All randomness comes from an explicit NumPy generator.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


class VisualWorld:
    """Sample four-way visual reference tasks from four geometric classes.

    ``sample(B)`` returns float32 images in [0, 1] with these shapes:
    target_images [B, 1, 32, 32], candidate_images [B, 4, 1, 32, 32].
    Metadata arrays use int64: target_ids [B], candidate_ids [B, 4], and
    correct_indices [B]. Every candidate set contains each class exactly once.

    Class names are a human-facing description of the renderer, not inputs to
    the agents. There is no association between class and candidate position.
    """

    CLASS_NAMES = ("circle", "square", "triangle", "cross")
    NUM_CLASSES = 4
    IMAGE_SIZE = 32

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[: self.IMAGE_SIZE, : self.IMAGE_SIZE]
        self._xx = xx.astype(np.float32)[None, :, :]
        self._yy = yy.astype(np.float32)[None, :, :]

    @staticmethod
    def _count(value: int, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
        return int(value)

    def get_rng_state(self) -> dict[str, Any]:
        """Return an independent state snapshot for exact sequence replay."""
        return deepcopy(self.rng.bit_generator.state)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore a state produced by this generator type."""
        self.rng.bit_generator.state = deepcopy(state)

    def render_class(
        self, class_id: int, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Render n independent observations of one class as [n, 1, 32, 32].

        An optional external generator allows analysis probes to avoid changing
        the training world's random stream. All classes share the same nuisance
        distributions: position, scale, rotation, brightness, and pixel noise.
        """
        n = self._count(n, "n")
        class_id = self._count(class_id, "class_id")
        if class_id >= self.NUM_CLASSES:
            raise ValueError(f"class_id must be in [0, {self.NUM_CLASSES})")
        rng = self.rng if rng is None else rng
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if n == 0:
            return np.empty((0, 1, self.IMAGE_SIZE, self.IMAGE_SIZE), np.float32)

        # Independent continuous nuisance variables prevent identical pixels
        # from serving as an instance-matching shortcut across viewpoints.
        cx = rng.uniform(13.0, 18.0, (n, 1, 1)).astype(np.float32)
        cy = rng.uniform(13.0, 18.0, (n, 1, 1)).astype(np.float32)
        radius = rng.uniform(7.0, 9.5, (n, 1, 1)).astype(np.float32)
        angle = rng.uniform(-0.12, 0.12, (n, 1, 1)).astype(np.float32)
        dx, dy = self._xx - cx, self._yy - cy
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        x = (cos_angle * dx + sin_angle * dy) / radius
        y = (-sin_angle * dx + cos_angle * dy) / radius

        if class_id == 0:
            mask = x * x + y * y <= 1.0
        elif class_id == 1:
            mask = np.maximum(np.abs(x), np.abs(y)) <= 1.0
        elif class_id == 2:
            mask = (y >= -1.0) & (y <= 1.0) & (np.abs(x) <= (y + 1.0) / 2.0)
        else:
            mask = ((np.abs(x) <= 0.35) & (np.abs(y) <= 1.0)) | (
                (np.abs(y) <= 0.35) & (np.abs(x) <= 1.0)
            )

        background = rng.uniform(0.02, 0.12, (n, 1, 1)).astype(np.float32)
        foreground = rng.uniform(0.70, 0.98, (n, 1, 1)).astype(np.float32)
        images = np.where(mask, foreground, background).astype(np.float32)
        images += rng.normal(0.0, 0.012, images.shape).astype(np.float32)
        np.clip(images, 0.0, 1.0, out=images)
        return images[:, None, :, :]

    def _render_ids(self, ids: np.ndarray) -> np.ndarray:
        flat_ids = ids.reshape(-1)
        images = np.empty(
            (len(flat_ids), 1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32
        )
        for class_id in range(self.NUM_CLASSES):
            locations = np.flatnonzero(flat_ids == class_id)
            images[locations] = self.render_class(class_id, len(locations))
        return images.reshape(*ids.shape, 1, self.IMAGE_SIZE, self.IMAGE_SIZE)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Draw independent targets with replacement and shuffled candidates."""
        batch_size = self._count(batch_size, "batch_size")
        target_ids = self.rng.integers(
            self.NUM_CLASSES, size=batch_size, dtype=np.int64
        )
        candidate_ids = np.tile(
            np.arange(self.NUM_CLASSES, dtype=np.int64), (batch_size, 1)
        )
        # Generator.permuted independently shuffles each row, rather than
        # applying one common permutation to every member of the batch.
        candidate_ids = self.rng.permuted(candidate_ids, axis=1)
        correct_indices = np.argmax(
            candidate_ids == target_ids[:, None], axis=1
        ).astype(np.int64)
        return {
            "target_images": self._render_ids(target_ids),
            "candidate_images": self._render_ids(candidate_ids),
            "target_ids": target_ids,
            "candidate_ids": candidate_ids,
            "correct_indices": correct_indices,
        }
