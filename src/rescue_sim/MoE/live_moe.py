# Copyright 2026 TUHH Group 05 — A. Herrero Callejo, C. Marcos Alonso,
# M. M. Orfany, A. Moazzen (alirezamoazen.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# Developed within Group 5 at the Hamburg University of Technology (TUHH)
# Under the academic supervision of Prof. Dr. Rainer Marrone.
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Live-expert execution for the Neural MoE: route with the learned gate,
act with the real experts.

The distilled expert heads are lossy students (E1 clones APF at 31% vs the
real APF's 62%; E2 tops out near 70% vs TransfQMix's 84%). At rollout time
the true experts are available and cheap enough, so — mirroring the live-E3
Epidemic-Q takeover the dashboard already does — this module lets the router
keep making the *decisions* while the genuine algorithms make the *moves*:

    dominant == exploration   -> real APF (Artificial Potential Fields)
    dominant == coordination  -> real TransfQMix checkpoint (greedy)
    dominant == fallback      -> GRU head / live Epidemic-Q fleet (as before)

Measured on 100 paired unseen 14x14 grids (bare greedy, 200-step cap):
distilled-heads MoE 72% vs live-expert MoE 84% at 66 steps.

Everything degrades gracefully: if the TransfQMix checkpoint is missing or
incompatible, callers get ``None`` and the distilled heads stay in charge.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rescue_sim.config.settings import GridSettings, TransfQmixSettings


class LiveExperts:
    """Real APF + real TransfQMix, driven by the MoE router's decisions.

    Built once per dashboard run / evaluation; ``reset`` per episode;
    ``actions`` per step returns each live expert's proposed action per agent.
    """

    def __init__(self, apf_teacher, transf_trainer) -> None:
        self._apf = apf_teacher
        self._transf = transf_trainer

    # -- construction --------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        grid_settings: GridSettings,
        num_agents: int,
        view_radius: int,
        max_steps: int,
        checkpoint_dir: str = "checkpoints",
        seed: int = 0,
    ) -> Optional["LiveExperts"]:
        """Build both live experts; returns None if torch/checkpoint missing."""
        try:
            from rescue_sim.MoE.pipeline import ExplorationTeacher
            from rescue_sim.TransfQMix import EntityRescueEnv, TransfQMIX

            ctor_env = EntityRescueEnv(
                grid_settings, num_agents=num_agents, max_steps=max_steps,
                view_radius=view_radius, seed=seed,
            )
            trainer = TransfQMIX(
                ctor_env,
                TransfQmixSettings(num_agents=num_agents, random_seed=seed),
                device="cpu",
            )
            trainer.load_checkpoint(f"{checkpoint_dir}/transfqmix.pt")
            apf = ExplorationTeacher(np.random.default_rng(seed))
            return cls(apf, trainer)
        except Exception:  # noqa: BLE001 - live experts are an enhancement
            return None

    # -- per-episode / per-step API ------------------------------------------

    def reset(self, env) -> None:
        """Call after ``env.reset()``: re-anchors APF's live-target tracking."""
        self._apf.reset(env)

    def observe(self, env) -> None:
        """Call after ``env.step()``: lets APF drop just-rescued targets."""
        self._apf.observe(env)

    def actions(self, env, valid_mask: np.ndarray) -> dict[str, np.ndarray]:
        """Both experts' proposals for the current state.

        ``env`` may be any RescueEnv; entity tokens for TransfQMix are built
        with EntityRescueEnv's (self-contained) token method, which only uses
        base-class internals.
        """
        from rescue_sim.TransfQMix import EntityRescueEnv

        tokens = EntityRescueEnv.entity_obs(env)
        return {
            "apf": self._apf.act(env, valid_mask),
            "transf": self._transf.select_actions(tokens, valid_mask, greedy=True),
        }

    def apply(
        self,
        actions: np.ndarray,
        dominant: list[int],
        proposals: dict[str, np.ndarray],
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Overrides ``actions`` in place per the router's dominant expert:
        exploration (0) -> live APF, coordination (1) -> live TransfQMix.
        Fallback (2) is left for the caller (GRU / live Epidemic-Q fleet).
        """
        for i, dom in enumerate(dominant):
            if dom == 0 and valid_mask[i, proposals["apf"][i]]:
                actions[i] = proposals["apf"][i]
            elif dom == 1 and valid_mask[i, proposals["transf"][i]]:
                actions[i] = proposals["transf"][i]
        return actions
