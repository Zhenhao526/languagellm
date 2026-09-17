"""Fixed, non-social Minecraft calibration arenas; never supplies policy actions.

MineRL 1.0 ignores several legacy XML handlers. These missions therefore require
the separately audited calibration Java initializer, and the runner checks reset
state before allowing actions. Geometry is an experimenter-built local arena in
a native Minecraft survival world, not a reconstruction of natural foraging.
"""
from __future__ import annotations

import math
import random

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers as H
from minerl.herobraine.hero.handler import Handler

SCENARIOS = ("movement", "wood", "food_selected", "food_select")
SCENE_VERSION = "local-survival-arena-v1"
DEFAULT_STEPS = {"movement": 1200, "wood": 2400,
                 "food_selected": 800, "food_select": 800}


class CorrectStartingFood(Handler):
    """Correct the legacy Python handler's StartingHealth/StartingFood typo."""
    def __init__(self, food: int, saturation: float = 0.0):
        self.food = food
        self.saturation = saturation

    def to_string(self):
        return "calibration_starting_food"

    def xml_template(self):
        return '<StartingFood food="{{food}}" foodSaturation="{{saturation}}"/>'


class LiteralDrawingDecorator(H.DrawingDecorator):
    """Only program-generated primitives; avoid Jinja escaping nested XML."""
    def xml_template(self):
        return "<DrawingDecorator>" + self.to_draw + "</DrawingDecorator>"


def _cuboid(x1, y1, z1, x2, y2, z2, block):
    return (f'<DrawCuboid x1="{x1}" y1="{y1}" z1="{z1}" '
            f'x2="{x2}" y2="{y2}" z2="{z2}" type="{block}"/>')


def _block(x, y, z, block):
    return f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{block}"/>'


def scene_configuration(scenario: str, seed: int, food: int = 8) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if not 1 <= food <= 20:
        raise ValueError("food must be between 1 and 20")
    if scenario.startswith("food") and food == 20:
        raise ValueError("Food calibration requires food < 20; full hunger blocks native eating")
    rng = random.Random(seed)
    heading_index = rng.randrange(4)
    dx, dz, yaw = ((0, 1, 0), (-1, 0, 90),
                   (0, -1, 180), (1, 0, -90))[heading_index]
    # Fixed three-block distance gives a visible and reachable trunk in every
    # prespecified seed. Variation is orientation and food slot, not seed search.
    primary_tree = {"x": 3 * dx, "y": 64, "z": 3 * dz}
    food_slot = 0 if scenario == "food_selected" else rng.randrange(1, 9)
    inventory = {}
    if scenario.startswith("food"):
        inventory[food_slot] = {"type": "bread", "quantity": 4}
        if food_slot != 0:
            inventory[0] = {"type": "dirt", "quantity": 1}
    return {
        "version": SCENE_VERSION, "scenario": scenario, "seed": seed,
        "name": f"Calib_{scenario}_seed_{seed}",
        "placement": {"x": .5, "y": 64.0, "z": .5,
                      "yaw": float(yaw), "pitch": 2.3 if scenario == "wood" else 0.0},
        "arena_bounds": {"x": [-20, 20], "z": [-20, 20], "floor_y": 63},
        "primary_tree": primary_tree if scenario == "wood" else None,
        "food_item": "bread", "food_slot": food_slot if inventory else None,
        "initial_selected_slot": 0, "inventory": inventory,
        "food": food if scenario.startswith("food") else 20,
        "health": 20.0, "saturation": 0.0,
        "resolution": [640, 360], "gui_scale": 1,
        "gamma": 2.0, "fov": 70.0, "cursor_size": 16,
        "world": {"mode": "Survival", "difficulty": "HARD",
                  "time": 6000, "daylight_cycle": False, "weather": "clear",
                  "mob_spawning": False,
                  "generation": "native generator plus bounded experimenter-built arena",
                  "native_breaking_eating_and_hunger_rules": True,
                  "requires_calibration_java_initializer": True},
        "limits": [
            "Artificial flat safe arena; not natural food search or resource ecology.",
            "Wood is hand-placed natural Minecraft log/leaf blocks; no dropped items are preloaded.",
            "Food is supplied at reset; this does not test food acquisition.",
            "The name of the scenario is never given to VPT as an instruction.",
            "No world-knowledge, navigation-goal, or causal-understanding claim follows from movement alone.",
        ],
    }


class CalibrationSurvival(HumanSurvival):
    def __init__(self, scenario: str, seed: int, food: int = 8):
        self.configuration = scene_configuration(scenario, seed, food)
        super().__init__(name=self.configuration["name"], resolution=(640, 360),
                         guiscale_range=[1, 1], gamma_range=[2., 2.],
                         fov_range=[70., 70.], cursor_size_range=[16, 16])

    def create_agent_start(self):
        c = self.configuration
        return super().create_agent_start() + [
            H.AgentStartPlacement(**c["placement"]),
            H.InventoryAgentStart(c["inventory"]),
            H.StartingHealthAgentStart(max_health=c["health"], health=c["health"]),
            CorrectStartingFood(c["food"], c["saturation"]),
            H.DoneOnDeath(),
        ]

    def create_server_world_generators(self):
        # The Java calibration initializer consumes local drawings. MineRL 1.0
        # retains its native generator; this legacy declaration alone does not
        # prove a flat world was actually created.
        return [H.FlatWorldGenerator(force_reset=True, generatorString="")]

    def create_server_initial_conditions(self):
        return [H.TimeInitialCondition(False, 6000),
                H.WeatherInitialCondition("clear"), H.SpawningInitialCondition(False)]

    def create_server_decorators(self):
        drawings = [
            _cuboid(-20, 64, -20, 20, 85, 20, "air"),
            _cuboid(-20, 62, -20, 20, 62, 20, "stone"),
            _cuboid(-20, 63, -20, 20, 63, 20, "grass_block"),
        ]
        if self.configuration["scenario"] == "wood":
            p = self.configuration["primary_tree"]
            # One reachable tree and two peripheral trees. Leaves precede logs
            # so that trunk cells are never overwritten by the canopy.
            trees = [(p["x"], p["z"]), (8, 7), (-8, -7)]
            for x, z in trees:
                drawings += [_cuboid(x - 2, 67, z - 2, x + 2, 68, z + 2, "oak_leaves"),
                             _cuboid(x - 1, 69, z - 1, x + 1, 69, z + 1, "oak_leaves")]
                drawings += [_block(x, y, z, "oak_log") for y in range(64, 69)]
        elif self.configuration["scenario"] == "movement":
            drawings += [_cuboid(7, 64, 6, 9, 64, 8, "dirt"),
                         _cuboid(-9, 64, -8, -7, 64, -6, "dirt")]
        return [LiteralDrawingDecorator("\n".join(drawings))]


def script_action(config: dict, step: int, state: dict) -> dict:
    """Explicit positive control. This function is never called for VPT."""
    scenario = config["scenario"]
    if scenario == "movement":
        if 20 <= step < 60 or 100 <= step < 140:
            return {"forward": 1}
        if 80 <= step < 90:
            return {"camera": [0., 9.]}
        return {}
    if scenario.startswith("food"):
        if scenario == "food_select" and step == 10:
            return {f"hotbar.{config['food_slot'] + 1}": 1}
        return {"use": 1} if 20 <= step < 220 else {}
    if scenario == "wood":
        if step < 20 or step >= 320:
            return {}
        loc = state.get("location_stats", {})
        p = config["primary_tree"]
        # This scripted instrument explicitly uses scoring coordinates to aim;
        # it is not a model action or evidence of model competence.
        x, y, z = (float(loc.get(k, config["placement"][v]))
                   for k, v in (("xpos", "x"), ("ypos", "y"), ("zpos", "z")))
        # Break upper then bottom log. 110 environment steps per target leaves
        # ample room for native hand-breaking; no mining-speed multiplier.
        target_y = 65.5 if step < 130 else 64.5
        dx, dz = p["x"] + .5 - x, p["z"] + .5 - z
        distance = max(math.hypot(dx, dz), .05)
        yaw = math.degrees(math.atan2(-dx, dz))
        pitch = math.degrees(math.atan2(y + 1.62 - target_y, distance))
        yaw_error = (yaw - float(loc.get("yaw", 0)) + 180.) % 360. - 180.
        pitch_error = pitch - float(loc.get("pitch", 0))
        action = {"attack": 1,
                  "camera": [max(-30., min(30., pitch_error)),
                             max(-30., min(30., yaw_error))]}
        if step >= 240 and distance > .7:
            action["forward"] = 1
        return action
    raise ValueError(scenario)
