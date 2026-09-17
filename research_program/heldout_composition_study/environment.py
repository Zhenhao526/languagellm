"""Thin local wrapper around the frozen staged dual-token environment."""
from research_program.action_dependent_signaling_study.environment import rollout, oracle_team_return

__all__ = ["rollout", "oracle_team_return"]
