"""Deterministic resource-specific views built from the frozen visual input."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "redesign_v0.28"))
import world
sys.path.insert(0, str(PROJECT / "redesign_v0.20"))
import temporal_model


VIEWS = ("full", "food_only", "water_only")


def _validate(projected, worlds):
    if not isinstance(projected, torch.Tensor) or projected.ndim != 2 or projected.shape[1] != 64 or projected.dtype != torch.float32 or projected.device.type != "cpu" or projected.requires_grad or not bool(torch.isfinite(projected).all()):
        raise ValueError("projected must be frozen CPU float32[12,64]")
    n = len(worlds["map_id"])
    if set(worlds) != {"map_id", "photo_ids", "positions", "shown"} or worlds["photo_ids"].shape != (n, 2) or worlds["positions"].shape != (n, 2) or worlds["shown"].shape != (n,):
        raise ValueError("world table has invalid shape")
    if not np.issubdtype(worlds["photo_ids"].dtype, np.integer) or not np.issubdtype(worlds["positions"].dtype, np.integer) or not np.isin(worlds["shown"], (0, 1)).all():
        raise ValueError("world table has invalid dtype/domain")


def frames(projected, worlds, view="full"):
    """Return legal two-frame inputs with an optional resource mask.

    ``food_only`` and ``water_only`` erase the other resource slot and its
    existence bit in both frames. The original full view is delegated to the
    inherited world implementation and is therefore exactly reproducible.
    """
    _validate(projected, worlds)
    if view not in VIEWS:
        raise ValueError("unknown view")
    if view == "full":
        return world.frames(projected, worlds)
    visible_kind = 0 if view == "food_only" else 1
    n = len(worlds["map_id"])
    first_slots = projected.new_zeros(n, 6, 64)
    first_exists = projected.new_zeros(n, 6)
    rows = torch.arange(n)
    sites = torch.from_numpy(worlds["positions"][:, visible_kind])
    ids = torch.from_numpy(worlds["photo_ids"][:, visible_kind])
    first_slots[rows, sites] = projected[ids]
    first_exists[rows, sites] = 1.
    second_slots = projected.new_zeros(n, 6, 64)
    second_exists = projected.new_zeros(n, 6)
    shown = worlds["shown"] == visible_kind
    if bool(shown.any()):
        selected = torch.from_numpy(np.flatnonzero(shown))
        selected_sites = torch.from_numpy(worlds["positions"][shown, visible_kind])
        selected_ids = torch.from_numpy(worlds["photo_ids"][shown, visible_kind])
        second_slots[selected, selected_sites] = projected[selected_ids]
        second_exists[selected, selected_sites] = 1.
    first = torch.cat((first_slots.flatten(1), first_exists), dim=1)
    second = torch.cat((second_slots.flatten(1), second_exists), dim=1)
    bits = projected.new_ones(n, 2)
    bits[:, 1] = 0
    return torch.stack((first, second), dim=1), bits


@torch.no_grad()
def encode(agent, projected, worlds, view="full"):
    inputs, bits = frames(projected, worlds, view)
    h = temporal_model.observe_sequence(agent, inputs, bits, "full")
    if h.shape != (len(worlds["map_id"]), 96) or not bool(torch.isfinite(h).all()):
        raise ValueError("encoded view has invalid shape or values")
    return h
