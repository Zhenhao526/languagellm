"""Cases for the heldout joint-combination semantic-transfer probe."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_new_receiver_compositional_holdout_study import design as holdout_design

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "triadic_new_receiver_compositional_holdout_study/results/holdout_001"
SEEDS = tuple(range(66701, 66709))
SCHEDULES = ("static", "rematched")
LIVES = (True, False)
CONDITIONS = tuple(
    f"{schedule}_new_receiver_compositional_seen_joint_only_PL_{'live' if live else 'silent'}"
    for schedule in SCHEDULES for live in LIVES
)
AXES = ("kind", "length")
CHUNK_SIZE = 2048
SHAM_ROWS_PER_SENDER = 128


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_prepared():
    return json.loads((SOURCE_RUN / "prepared.json").read_text(encoding="utf8"))


def make_cases(static):
    spec = static["heldout_spec"]
    needs = [tuple(map(int, row)) for row in spec["needs"]]
    support = set(needs); lookup = {row: index for index, row in enumerate(needs)}
    nl, no = len(spec["layouts"]), len(spec["private_sites"])
    receiver = []; donor = []; sender = []; axis = []; background = []
    for row in needs:
        receiver_need_index = lookup[row]
        for actor in range(3):
            for axis_name in AXES:
                neighbors = [value for name, value in dataset.flips(row[actor]) if name == axis_name]
                for neighbor_value in neighbors:
                    changed = list(row); changed[actor] = int(neighbor_value); changed = tuple(changed)
                    if changed not in support:
                        continue
                    donor_need_index = lookup[changed]
                    for layout_index in range(nl):
                        for owner_index in range(no):
                            receiver.append((receiver_need_index * nl + layout_index) * no + owner_index)
                            donor.append((donor_need_index * nl + layout_index) * no + owner_index)
                            sender.append(actor); axis.append(axis_name); background.append(layout_index * no + owner_index)
    require(len(receiver) > 0, "Heldout neighbor set is empty")
    groups = defaultdict(list)
    for index, (actor, axis_name, bg) in enumerate(zip(sender, axis, background)):
        groups[(int(actor), str(axis_name), int(bg))].append(index)
    require(all(len(indices) >= 2 for indices in groups.values()), "Placebo stratum has one case")
    mapping = list(range(len(receiver)))
    for indices in groups.values():
        for position, index in enumerate(indices):
            mapping[index] = indices[(position + 1) % len(indices)]
    require(all(index != mapped for index, mapped in enumerate(mapping)), "Placebo identity")
    return dict(schema="compositional_holdout_semantic_transfer_cases_v1", partition=spec["partition"],
                needs=spec["needs"], layouts=spec["layouts"], private_sites=spec["private_sites"],
                world_count=spec["world_count"], background_count=nl * no,
                receiver_state_indices=receiver, donor_state_indices=donor, sender=sender, axis=axis,
                background_indices=background, placebo_donor_case_indices=mapping,
                case_count=len(receiver), strata_count=len(groups),
                donor_rule="same heldout orbit support, same layout and owner, one sender kind/length neighbor",
                placebo_rule="cyclic donor packet within sender×axis×background")


def make_prepared():
    static = source_prepared(); cases = make_cases(static)
    return dict(schema="compositional_holdout_semantic_transfer_prepared_v1", seeds=list(SEEDS),
                schedules=list(SCHEDULES), lives=["live", "silent"], conditions=list(CONDITIONS),
                heldout_spec=static["heldout_spec"], cases=cases, source_run=str(SOURCE_RUN),
                scientific_question="Does the new receiver's message directionally transfer heldout joint-combination content?",
                primary="A versus mean(B,C) aligned-minus-placebo plan transfer on heldout neighbors",
                controls="B/C are frozen full-combination parent policies; silent is closed-channel",
                posthoc_relative_to="triadic_new_receiver_compositional_holdout_study/results/holdout_001",
                claim_boundary="Heldout behavioral transfer is not evidence for lexical meaning, compositional syntax or language origin.")


if __name__ == "__main__":
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
