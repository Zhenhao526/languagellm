"""Frozen v0.33 team schedules and condition-paired world fixtures."""
from pathlib import Path
import hashlib
import json
import numpy as np


ROOT = Path(__file__).resolve().parent
DESIGN_FILE = ROOT / "support_design.json"
DESIGN_SHA256 = "00758a782127cbda10a48e7e4a1d5a2fd4e39616ee8c25f8d03500c9c8b3db8c"
_design_bytes = DESIGN_FILE.read_bytes()
if hashlib.sha256(_design_bytes).hexdigest() != DESIGN_SHA256:
    raise ValueError("prespecified v0.33 design changed")
_DESIGN = json.loads(_design_bytes)
CONDITIONS = tuple(_DESIGN["conditions"])
NAMESPACE = int(_DESIGN["sampling"]["namespace"])
BASE_BATCH = int(_DESIGN["sampling"]["base_worlds_per_team_slot"])
MAPS = np.asarray([(i, j) for i in range(6) for j in range(6) if i != j], np.int64)
INDEX = {tuple(x): i for i, x in enumerate(MAPS)}
PANELS = [tuple(x) for x in _DESIGN["panels"]]
TRAIN = tuple(tuple(x) for x in _DESIGN["training12_pairs"])
TARGET = tuple(tuple(x) for x in _DESIGN["target12_pairs"])


def seed_value(seed, panel, agent, kind):
    return int(np.random.SeedSequence([NAMESPACE, int(seed), int(panel), int(agent), int(kind)]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def _mapped(panel, pairs):
    q = PANELS[panel - 1]
    return np.sort(np.asarray([INDEX[(q[i], q[j])] for i, j in pairs], np.int64))


def groups(panel, condition):
    if panel not in (1, 2, 3) or condition not in CONDITIONS:
        raise ValueError("unknown panel or condition")
    train = _mapped(panel, TRAIN)
    target = _mapped(panel, TARGET)
    return dict(train12=train, target12=target,
                held18=np.setdiff1d(np.arange(30, dtype=np.int64), train),
                common30=np.arange(30, dtype=np.int64))


def team_slots(condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    if condition == "single_full":
        return tuple((int(r), int(s)) for r, s in _DESIGN["population"]["single_teams"])
    return tuple(tuple(int(x) for x in team) for team in _DESIGN["population"]["dual_teams"])


def fixture(seed, panel, slot, step, condition, train_table):
    """Return 120 base worlds expanded to both resource masks.

    The condition is deliberately excluded from random streams. A slot's
    worlds and four uniform draws are therefore identical across all three
    communication conditions.
    """
    if condition not in CONDITIONS or slot not in range(4) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError("invalid fixture identity")
    if len(train_table["map_id"]) != 720:
        raise ValueError("requires frozen 720-row source training table")
    photo = np.asarray(train_table["photo_ids"])
    food = np.unique(photo[:, 0]); water = np.unique(photo[:, 1])
    expected = np.asarray([(f, w) for f in food for w in water], np.int64)
    if photo.shape != (720, 2) or len(food) != 6 or len(water) != 2 or not np.array_equal(photo[:12], expected):
        raise ValueError("requires six-by-two source photos")
    rng = lambda stream: np.random.default_rng(np.random.SeedSequence([NAMESPACE, int(seed), int(panel), int(slot), int(step), int(stream)]))
    maps = np.tile(groups(panel, condition)["train12"], 10)[rng(0).permutation(BASE_BATCH)]
    photos = rng(1).random((BASE_BATCH, 2))
    base = maps * 12 + (photos[:, 0] * 6).astype(np.int64) * 2 + (photos[:, 1] * 2).astype(np.int64)
    indices = np.concatenate((base, base + 360))
    uniforms = rng(2).random((2 * BASE_BATCH, 4)).astype(np.float32)
    return dict(indices=indices, uniforms=uniforms)


def subset(worlds, indices):
    return {key: value[indices] for key, value in worlds.items()}
