"""Deterministic, partially observed collection world for the v0.2 pilot.

The global state and witness are experimenter-only.  Agents receive observe()
and action_menu(); item identifiers in these APIs are private to each agent.
All actions are validated against the same pre-step state and resolve together.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import random

AGENTS = ("A", "B", "C")
LOCATIONS = ("营地", "林地", "河滩")
RULES = """你们三人共同在12个物理动作步内完成营地任务板的两项交付需求，三人能力相同。
【目标与计数】交付物的种类、长短、干湿必须同时与同一项需求完全一致，且在该项指定地点交付。可搬运不等于符合需求。quantity是本段累计需要的物品件数；每件物品算一个单位，两人共同搬一件仍只算一个单位。数量为2时需要两件不同物品。
【观察与地图】营地、林地、河滩两两有道路，正常通行一步到达。你只直接观察所在地的地面物品、同地伙伴的可见状态、自己的携带物和动作结果。任务板只在营地可读，正文是本段固定需求，不显示异地交付进度，再次查看不会增加需求。伙伴广播不是环境保证的事实。物品句柄只对自己有效，伙伴对同一物品使用不同句柄；动作编号也可能随观察改变。
【动作与时间】每人每步只能选一个当前菜单动作，三人的动作按本步开始时的物理状态同时结算。不能让自己的本步动作依赖伙伴本步尚未完成的移动、拿取、加工或交接。拿取、移动、加工、放下、交付分别占一步；广播不执行物理动作。菜单表示可以尝试，不保证满足需求或合作成功。
【携带】每人有一个普通物品位和一个斧具位，可以同时携带一件普通物品和一把斧具。所有纤维无论长短、干湿都无需加工，可由一人拿取并占一个普通位。短木材加工后可单人拿取。长木材加工后仍需两人共同搬运，占双方各自的普通位；开始共同搬运时双方普通位必须空闲。共同负载期间，只能等待或对该负载共同搬运、共同卸下、共同交付，不能独立移动、使用或交接其他物品；已携带的斧具仍保留。
【加工】本人持斧且与未加工木材同地才能加工。加工只使该木材成为已加工状态，不改变种类、长短、干湿，不自动拿起木材，木材仍在原地，斧具不消耗。没有改变资源长短或干湿的操作。
【共同动作】两人在本步开始时同地，分别选同一物品、互为搭档的同一种共同动作；共同搬运还须选相同目的地。任一项不匹配，共同动作不发生，失败结果不透露伙伴实际选择。共同搬运成功时两人和物品一起移动一步，到达后仍共同携带。共同卸下把物品放在当地、释放双方普通位，但不计交付。共同交付成功才释放负载并计入需求；当地地面上已加工的长木材也可由两个普通位空闲的人直接共同交付。
【交接与争抢】未处于共同负载时，可把自己携带的普通物品或斧具交给同地伙伴；接收者对应携带位须空闲，且本步选择等待，双方各占本步动作。多个合法交接同时指向同一接收者时全部失败；多人同时拿取同一件地面物品时也全部失败。
【交付结果】纤维和短木材须由自己携带后选单人交付。拿到物品、运到指定地点或放在指定地点，都不自动完成交付。只有“交付被接收”才计入完成量；接收后物品退出可操作物品，释放相应携带位。拒收不改变物品原来的携带或地面状态。若本步开始时某项需求仍有缺额，本步全部匹配交付均可接收，但计分最多达到所需总数；之后动作步不再接收该项超额物品。同一物品不能重复计分。
【道路变化】道路可能局部关闭，关闭路段双向不可通过；变化不一定能从另一端看见。已知关闭道路列表只表示本人已直接得知的情况，空列表不保证所有路都开放。尝试受阻道路消耗一步，位置不变，菜单仍可能列出该通行尝试。物理变化在动作结算后发生，之后观察才反映本人可见的变化。
【反馈与结束】你只获得自己可见的动作结果，不直接获得伙伴未公开的反馈或未执行动作。失败消耗一步，不给正确目标或计划。全部需求完成则提前结束，否则第12步后结束；结束时三人获得实际完成的目标单位比例。以完成单位最多为先，同完成度下越早完成越好。"""


def _rng(*parts):
    data = "|".join(map(str, parts)).encode()
    return random.Random(int.from_bytes(hashlib.sha256(data).digest()[:8], "big"))


class World:
    """World(seed, variant=0): 3 equal agents, nine objects, two goals.

    Variant changes material attributes, fiber quantity, destinations, initial
    axe position and the occurrence of a single road closure.  Seed changes
    role/location assignments, private handles and presentation.
    There is always a replay-verified solution within max_steps (default 12).
    """

    def __init__(self, seed: int, variant: int = 0, max_steps: int = 12):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(variant, int) or variant < 0:
            raise ValueError("variant must be a nonnegative integer")
        if max_steps < 9:
            raise ValueError("the verified family requires at least 9 steps")
        self.seed, self.variant, self.max_steps = seed, variant, max_steps
        self.t = 0
        rng = _rng(seed, variant, "world")
        roles = list(AGENTS)
        rng.shuffle(roles)
        self.roles = dict(zip(("collector", "wood", "tool"), roles))
        wood_site = rng.choice(("林地", "河滩"))
        fiber_site = next(x for x in ("林地", "河滩") if x != wood_site)
        self.layout = {"wood_site": wood_site, "fiber_site": fiber_site}
        self.positions = dict(zip(roles, ("营地", wood_site, fiber_site)))
        self.inventory = {a: {"tool": None, "cargo": None} for a in AGENTS}
        self.known_closed = {a: [] for a in AGENTS}
        self.event = {
            "kind": "road_closure", "enabled": variant % 3 != 0,
            "at_step": 2 + rng.randrange(3), "edge": ["营地", wood_site],
            "observer_site": "营地", "occurred": False,
        }
        wood_length = "长" if variant % 2 == 0 else "短"
        wood_condition = "干" if (variant // 2) % 2 == 0 else "湿"
        fiber_length = "长" if (variant // 4) % 2 == 0 else "短"
        fiber_condition = "干" if (variant // 8) % 2 == 0 else "湿"
        fiber_quantity = 2 if variant % 3 == 1 else 1
        wood_dest = "营地" if (variant // 3) % 2 == 0 else fiber_site
        fiber_dest = "营地" if (variant // 6) % 2 == 0 else wood_site
        self.goals = [
            {"kind": "木材", "length": wood_length, "condition": wood_condition,
             "quantity": 1, "destination": wood_dest, "delivered": 0},
            {"kind": "纤维", "length": fiber_length, "condition": fiber_condition,
             "quantity": fiber_quantity, "destination": fiber_dest, "delivered": 0},
        ]
        self.items = {}
        def add(kind, length, condition, site, target=False):
            oid = f"o{len(self.items):02d}"
            self.items[oid] = {
                "kind": kind, "length": length, "condition": condition,
                "location": site,
                "processed": kind != "木材", "carriers": [], "delivered": False,
                "target_for_witness": target,
            }
            return oid

        # All four wood combinations occur once, independently of the demand;
        # local attribute frequency cannot identify the hidden target.
        for length in ("长", "短"):
            for condition in ("干", "湿"):
                add("木材", length, condition, wood_site,
                    length == wood_length and condition == wood_condition)
        # Two fiber classes occur twice each.  Neither frequency nor ordering
        # reveals which class the board requests, or whether it requests 1 or 2.
        fiber_classes = sorted({(fiber_length, fiber_condition),
                                ("短" if fiber_length == "长" else "长",
                                 "湿" if fiber_condition == "干" else "干")})
        for length, condition in fiber_classes:
            for _ in range(2):
                add("纤维", length, condition, fiber_site,
                    length == fiber_length and condition == fiber_condition)
        axe_site = (fiber_site, "营地", wood_site)[(variant // 2) % 3]
        add("斧具", None, None, axe_site)
        self.scenario_parameters = {
            "family": "river_valley_v0.2", "wood_attributes": [wood_length, wood_condition],
            "fiber_attributes": [fiber_length, fiber_condition], "wood_quantity": 1,
            "fiber_quantity": fiber_quantity, "wood_destination": wood_dest,
            "fiber_destination": fiber_dest, "wood_site": wood_site, "fiber_site": fiber_site,
            "axe_initial_site": axe_site, "initial_positions": deepcopy(self.positions),
            "local_event": deepcopy(self.event),
            "split": "engineering; no confirmatory train/test split assigned",
            "factorial_note": "variant selects a deterministic family, not a fully crossed factorial design; wood quantity is fixed at 1",
        }
        self.handles = {}
        for i, a in enumerate(AGENTS):
            labels = list(range(1 + i * 100, 100 + i * 100))
            _rng(seed, variant, a, "handles").shuffle(labels)
            self.handles[a] = {oid: f"物{labels[j]}" for j, oid in enumerate(self.items)}
        self.last_feedback = {a: None for a in AGENTS}
        self.witness = []
        self.witness = self._make_witness()

    @property
    def done(self):
        return self.status != "running"

    @property
    def status(self):
        if all(g["delivered"] >= g["quantity"] for g in self.goals):
            return "success"
        return "timeout" if self.t >= self.max_steps else "running"

    @property
    def score(self):
        return sum(g["delivered"] for g in self.goals) / sum(g["quantity"] for g in self.goals)

    def state_dict(self):
        """Experimenter-only state; never insert it in an agent prompt."""
        return deepcopy(self.__dict__)

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.__dict__.update(deepcopy(state))
        return obj

    def _check_agent(self, agent):
        if agent not in AGENTS:
            raise ValueError(f"unknown agent {agent!r}")

    def _item_view(self, agent, oid):
        x = self.items[oid]
        view = {"handle": self.handles[agent][oid], "kind": x["kind"]}
        if x["kind"] != "斧具":
            view.update(length=x["length"], condition=x["condition"])
        if x["kind"] == "木材":
            view["processed"] = x["processed"]
        if x["carriers"]:
            view["carried_by"] = sorted(x["carriers"])
        return self._shuffle_fields(view, agent, oid)

    def _shuffle_fields(self, value, agent, salt):
        """Presentation order is agent-specific; semantics are unchanged."""
        if isinstance(value, dict):
            keys = list(value)
            _rng(self.seed, self.variant, agent, self.t, salt, "fields").shuffle(keys)
            return {k: self._shuffle_fields(value[k], agent, f"{salt}/{k}") for k in keys}
        if isinstance(value, list):
            return [self._shuffle_fields(v, agent, f"{salt}/{i}") for i, v in enumerate(value)]
        return value

    def _closed(self, src, dest):
        return self.event["occurred"] and set((src, dest)) == set(self.event["edge"])

    def observe(self, agent):
        self._check_agent(agent)
        here = self.positions[agent]
        ground = [oid for oid, x in self.items.items()
                  if x["location"] == here and not x["delivered"] and not x["carriers"]]
        _rng(self.seed, self.variant, agent, self.t, "objects").shuffle(ground)
        inv = self.inventory[agent]
        partners = []
        for other in AGENTS:
            if other != agent and self.positions[other] == here:
                visible = [self._item_view(agent, oid) for oid, x in self.items.items()
                           if other in x["carriers"] and not x["delivered"]]
                partners.append({"agent": other, "visible_carried_items": visible})
        obs = {
            "agent": agent, "step": self.t, "steps_remaining": self.max_steps - self.t,
            "location": here, "map": list(LOCATIONS),
            "ground_items": [self._item_view(agent, oid) for oid in ground],
            "carried_items": [self._item_view(agent, oid) for oid in inv.values() if oid],
            "nearby_agents": partners,
            "known_closed_roads": deepcopy(self.known_closed[agent]),
            "last_result": deepcopy(self.last_feedback[agent]),
        }
        if here == "营地":
            # The posted requirements are static.  Remote deliveries do not
            # magically update this local board during the episode.
            obs["task_board"] = [{k: deepcopy(v) for k, v in g.items() if k != "delivered"}
                                 for g in self.goals]
            if self.event["occurred"] and self.event["edge"] not in obs["known_closed_roads"]:
                obs["known_closed_roads"].append(deepcopy(self.event["edge"]))
        return self._shuffle_fields(obs, agent, "observation")

    def full_information_observe(self, agent):
        """Independent ability control ONLY; complete current physical state.

        Uses this observer's handles.  It contains neither future events nor
        object target flags, seeds, variant identifiers or the witness plan.
        """
        self._check_agent(agent)
        current = self.observe(agent)
        sites = []
        for site in LOCATIONS:
            objects = []
            for oid, x in self.items.items():
                if x["delivered"]:
                    continue
                item_site = self.positions[x["carriers"][0]] if x["carriers"] else x["location"]
                if item_site == site:
                    objects.append(self._item_view(agent, oid))
            _rng(self.seed, self.variant, agent, self.t, site, "full_objects").shuffle(objects)
            sites.append({"location": site, "items": objects,
                          "agents": [a for a in AGENTS if self.positions[a] == site]})
        _rng(self.seed, self.variant, agent, self.t, "full_sites").shuffle(sites)
        current["all_current_sites"] = sites
        current["task_board"] = deepcopy(self.goals)
        current["all_current_closed_roads"] = [deepcopy(self.event["edge"])] if self.event["occurred"] else []
        return self._shuffle_fields(current, agent, "full_observation")

    def _oid(self, agent, handle):
        return next((oid for oid, h in self.handles[agent].items() if h == handle), None)

    def _desc_item(self, agent, oid):
        x = self.items[oid]
        return f"{self.handles[agent][oid]}（{x['condition'] or ''}{x['length'] or ''}{x['kind']}）"

    def action_menu(self, agent):
        self._check_agent(agent)
        if self.done:
            return [{"id": 0, "description": "任务已结束；等待", "action": {"kind": "wait"}}]
        here = self.positions[agent]
        inv = self.inventory[agent]
        cargo = inv["cargo"]
        joint = cargo is not None and len(self.items[cargo]["carriers"]) == 2
        entries = [("等待", {"kind": "wait"})]

        def add(description, kind, oid=None, **kw):
            action = {"kind": kind, **kw}
            if oid is not None:
                action["item"] = self.handles[agent][oid]
            entries.append((description, action))

        if not joint:
            for dest in LOCATIONS:
                if dest != here:
                    # Do not filter by current hidden road condition.
                    add(f"前往{dest}", "move", destination=dest)
        if joint:
            candidates = [cargo]
        else:
            candidates = [oid for oid, x in self.items.items()
                          if x["location"] == here and not x["carriers"] and not x["delivered"]]
        for oid in candidates:
            x = self.items[oid]
            desc = self._desc_item(agent, oid)
            heavy = x["kind"] == "木材" and x["length"] == "长"
            if not x["carriers"] and not joint:
                if x["kind"] == "斧具" and inv["tool"] is None:
                    add(f"拿取{desc}", "pickup", oid)
                elif x["kind"] != "斧具" and not heavy and x["processed"] and cargo is None:
                    add(f"拿取{desc}", "pickup", oid)
                if x["kind"] == "木材" and not x["processed"] and inv["tool"]:
                    add(f"加工{desc}", "cut", oid)
            if heavy and x["processed"] and (cargo is None or joint):
                for other in AGENTS:
                    if other == agent or self.positions[other] != here:
                        continue
                    if joint and other not in x["carriers"]:
                        continue
                    if not joint and self.inventory[other]["cargo"] is not None:
                        continue
                    for dest in LOCATIONS:
                        if dest != here:
                            add(f"与{other}共同搬运{desc}到{dest}", "carry_together", oid,
                                partner=other, destination=dest)
                    add(f"与{other}共同交付{desc}", "deliver_together", oid, partner=other)
                    if joint:
                        add(f"与{other}共同卸下{desc}", "drop_together", oid, partner=other)
        if not joint:
            for slot, oid in inv.items():
                if oid is None:
                    continue
                desc = self._desc_item(agent, oid)
                add(f"放下{desc}", "drop", oid)
                if slot == "cargo":
                    add(f"交付{desc}", "deliver", oid)
                for other in AGENTS:
                    if other != agent and self.positions[other] == here and self.inventory[other][slot] is None:
                        add(f"把{desc}交给{other}（对方须等待）", "give", oid, partner=other)
        rest = entries[1:]
        _rng(self.seed, self.variant, agent, self.t, "menu").shuffle(rest)
        entries = entries[:1] + rest
        return [{"id": i, "description": desc, "action": action}
                for i, (desc, action) in enumerate(entries)]

    def step(self, actions):
        if self.done:
            raise RuntimeError("episode already finished")
        if set(actions) != set(AGENTS):
            raise ValueError("one action is required for each of A, B, C")
        decoded, feedback = {}, {}
        for a in AGENTS:
            menu = self.action_menu(a)
            value = actions[a]
            if isinstance(value, int) and not isinstance(value, bool):
                value = menu[value]["action"] if 0 <= value < len(menu) else None
            if isinstance(value, dict) and "action" in value:
                value = value["action"]
            if value not in [entry["action"] for entry in menu]:
                decoded[a] = {"kind": "invalid"}
            else:
                decoded[a] = deepcopy(value)
                if "item" in value:
                    decoded[a]["item"] = self._oid(a, value["item"])
            feedback[a] = {"action_succeeded": False, "result": "动作没有发生。"}
        before = self.state_dict()
        reserved_agents, reserved_items = set(), set()

        def finish(agent, result, success=True):
            feedback[agent] = {"action_succeeded": success, "result": result}
            reserved_agents.add(agent)

        # Reserve bilateral actions before independent actions.  A match means
        # both agents submitted this exact reciprocal action, from the snapshot.
        for a in AGENTS:
            if a in reserved_agents:
                continue
            act = decoded[a]
            if act["kind"] not in {"carry_together", "deliver_together", "drop_together"}:
                continue
            b, oid = act["partner"], act["item"]
            expected = dict(act, partner=a)
            if decoded[b] != expected or b in reserved_agents or oid in reserved_items:
                finish(a, "共同动作没有发生。", False)
                continue
            x = self.items[oid]
            if act["kind"] == "carry_together":
                dest = act["destination"]
                if self._closed(before["positions"][a], dest):
                    for who in (a, b):
                        self._remember_closure(who)
                        finish(who, "通道受阻，位置未变。", False)
                    continue
                x["location"] = None
                x["carriers"] = sorted((a, b))
                for who in (a, b):
                    self.positions[who] = dest
                    self.inventory[who]["cargo"] = oid
                    finish(who, "共同搬运成功。")
            elif act["kind"] == "drop_together":
                x["carriers"] = []
                x["location"] = before["positions"][a]
                for who in (a, b):
                    self.inventory[who]["cargo"] = None
                    finish(who, "共同卸下成功。")
            else:
                accepted = self._accept(oid, before["positions"][a], before["goals"])
                for who in (a, b):
                    if accepted:
                        self.inventory[who]["cargo"] = None
                    finish(who, "交付被接收。" if accepted else "交付未被接收。", accepted)
            reserved_items.add(oid)

        # Concurrent pickup attempts for one item all fail, rather than choosing
        # a winner by Python dictionary iteration order.
        pickup_counts = {}
        give_counts = {}
        for act in decoded.values():
            if act["kind"] == "pickup":
                pickup_counts[act["item"]] = pickup_counts.get(act["item"], 0) + 1
            elif act["kind"] == "give":
                give_counts[act["partner"]] = give_counts.get(act["partner"], 0) + 1

        # Transfers require the receiving actor to wait, preventing an item from
        # being handed over and used/moved by that actor within the same tick.
        for a in AGENTS:
            if a in reserved_agents or decoded[a]["kind"] != "give":
                continue
            act = decoded[a]
            b, oid = act["partner"], act["item"]
            if (give_counts[b] != 1 or decoded[b]["kind"] != "wait"
                    or b in reserved_agents or oid in reserved_items):
                finish(a, "交接没有发生。", False)
                continue
            slot = "tool" if self.items[oid]["kind"] == "斧具" else "cargo"
            self.inventory[a][slot] = None
            self.inventory[b][slot] = oid
            self.items[oid]["carriers"] = [b]
            finish(a, "交接成功。")
            finish(b, "收到了同地伙伴交接的物品。")
            reserved_items.add(oid)

        for a in AGENTS:
            if a in reserved_agents:
                continue
            act = decoded[a]
            kind, oid = act["kind"], act.get("item")
            if kind == "wait":
                finish(a, "等待。")
            elif kind == "move":
                if self._closed(before["positions"][a], act["destination"]):
                    self._remember_closure(a)
                    finish(a, "通道受阻，位置未变。", False)
                else:
                    self.positions[a] = act["destination"]
                    finish(a, "移动成功。")
            elif kind == "pickup" and pickup_counts[oid] == 1 and oid not in reserved_items:
                slot = "tool" if self.items[oid]["kind"] == "斧具" else "cargo"
                self.inventory[a][slot] = oid
                self.items[oid]["location"] = None
                self.items[oid]["carriers"] = [a]
                reserved_items.add(oid)
                finish(a, "拿取成功。")
            elif kind == "cut" and oid not in reserved_items:
                self.items[oid]["processed"] = True
                reserved_items.add(oid)
                finish(a, "加工成功。")
            elif kind == "drop" and oid not in reserved_items:
                slot = "tool" if self.items[oid]["kind"] == "斧具" else "cargo"
                self.inventory[a][slot] = None
                self.items[oid]["location"] = before["positions"][a]
                self.items[oid]["carriers"] = []
                reserved_items.add(oid)
                finish(a, "放下成功。")
            elif kind == "deliver" and oid not in reserved_items:
                accepted = self._accept(oid, before["positions"][a], before["goals"])
                if accepted:
                    self.inventory[a]["cargo"] = None
                reserved_items.add(oid)
                finish(a, "交付被接收。" if accepted else "交付未被接收。", accepted)
        self.t += 1
        if self.event["enabled"] and not self.event["occurred"] and self.t == self.event["at_step"]:
            self.event["occurred"] = True
        for a in AGENTS:
            if self.event["occurred"] and self.positions[a] == self.event["observer_site"]:
                self._remember_closure(a)
            feedback[a]["step"] = self.t
            feedback[a]["done"] = self.done
            if self.done:
                feedback[a]["team_score"] = self.score
                feedback[a]["status"] = self.status
        self.last_feedback = deepcopy(feedback)
        return deepcopy(feedback)

    def _remember_closure(self, a):
        if self.event["edge"] not in self.known_closed[a]:
            self.known_closed[a].append(deepcopy(self.event["edge"]))

    def _accept(self, oid, location, goals_at_step_start):
        x = self.items[oid]
        if x["delivered"] or not x["processed"]:
            return False
        for index, g in enumerate(self.goals):
            start = goals_at_step_start[index]
            if (g["destination"] == location and start["delivered"] < start["quantity"]
                    and all(g[k] == x[k] for k in ("kind", "length", "condition"))):
                g["delivered"] = min(g["quantity"], g["delivered"] + 1)
                x["delivered"] = True
                x["location"] = location
                x["carriers"] = []
                return True
        return False

    def _next_location(self, source, destination):
        if source == destination:
            return source
        queue = deque([(source, [])])
        seen = {source}
        while queue:
            here, path = queue.popleft()
            for dest in LOCATIONS:
                if dest in seen or self._closed(here, dest):
                    continue
                nxt = path + [dest]
                if dest == destination:
                    return nxt[0]
                seen.add(dest)
                queue.append((dest, nxt))
        raise RuntimeError("unreachable destination")

    def _make_witness(self):
        """A state-aware omniscient plan, used only to certify feasibility."""
        clone = self.from_state_dict(self.state_dict())
        collector, worker, helper = (self.roles[x] for x in ("collector", "tool", "wood"))
        axe = next(oid for oid, x in clone.items.items() if x["kind"] == "斧具")
        wood = next(oid for oid, x in clone.items.items() if x["kind"] == "木材" and x["target_for_witness"])
        if clone.items[axe]["location"] == clone.layout["wood_site"]:
            worker, helper = helper, worker
        plan = []

        def action(a, kind, oid=None, **kwargs):
            out = {"kind": kind, **kwargs}
            if oid is not None:
                out["item"] = clone.handles[a][oid]
            return out

        def move(a, dest):
            return action(a, "move", destination=clone._next_location(clone.positions[a], dest))

        while not clone.done:
            acts = {a: {"kind": "wait"} for a in AGENTS}
            fiber_goal = clone.goals[1]
            if fiber_goal["delivered"] < fiber_goal["quantity"]:
                cargo = clone.inventory[collector]["cargo"]
                if cargo:
                    if clone.positions[collector] == fiber_goal["destination"]:
                        acts[collector] = action(collector, "deliver", cargo)
                    else:
                        acts[collector] = move(collector, fiber_goal["destination"])
                else:
                    fiber = next(oid for oid, x in clone.items.items()
                                 if x["kind"] == "纤维" and x["target_for_witness"] and not x["delivered"])
                    if clone.positions[collector] == clone.items[fiber]["location"]:
                        acts[collector] = action(collector, "pickup", fiber)
                    else:
                        acts[collector] = move(collector, clone.items[fiber]["location"])
            x, goal = clone.items[wood], clone.goals[0]
            if not x["delivered"]:
                if not x["processed"]:
                    if clone.inventory[worker]["tool"]:
                        acts[worker] = (action(worker, "cut", wood) if clone.positions[worker] == x["location"]
                                        else move(worker, x["location"]))
                    else:
                        acts[worker] = (action(worker, "pickup", axe)
                                        if clone.positions[worker] == clone.items[axe]["location"]
                                        else move(worker, clone.items[axe]["location"]))
                    if clone.positions[helper] != x["location"]:
                        acts[helper] = move(helper, x["location"])
                elif x["length"] == "短":
                    if clone.inventory[worker]["cargo"] == wood:
                        acts[worker] = (action(worker, "deliver", wood) if clone.positions[worker] == goal["destination"]
                                        else move(worker, goal["destination"]))
                    else:
                        acts[worker] = (action(worker, "pickup", wood) if clone.positions[worker] == x["location"]
                                        else move(worker, x["location"]))
                else:
                    where = clone.positions[worker] if x["carriers"] else x["location"]
                    for a in (worker, helper):
                        if clone.positions[a] != where:
                            acts[a] = move(a, where)
                    if all(clone.positions[a] == where for a in (worker, helper)):
                        for a, b in ((worker, helper), (helper, worker)):
                            if where == goal["destination"]:
                                acts[a] = action(a, "deliver_together", wood, partner=b)
                            else:
                                acts[a] = action(a, "carry_together", wood, partner=b,
                                                 destination=clone._next_location(where, goal["destination"]))
            plan.append(deepcopy(acts))
            result = clone.step(acts)
            if any(not f["action_succeeded"] for f in result.values()):
                raise AssertionError(f"invalid witness seed={self.seed} variant={self.variant}: {acts}, {result}")
        if clone.status != "success":
            raise AssertionError(f"no witness within {self.max_steps} steps: seed={self.seed}, variant={self.variant}")
        return plan


if __name__ == "__main__":
    w = World(17, variant=1)
    print(json.dumps({"initial_observations": {a: w.observe(a) for a in AGENTS},
                      "witness_steps": len(w.witness)}, ensure_ascii=False, indent=2))
    for joint in w.witness:
        w.step(joint)
    print(json.dumps({"status": w.status, "score": w.score, "steps": w.t}, ensure_ascii=False))
