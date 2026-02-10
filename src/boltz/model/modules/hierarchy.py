"""Hierarchical conformational guidance scheduling.

Implements time-varying coupling hierarchy orthogonal to CryoBoltz's
multi-scale guidance (global -> local within each structure).

This module manages the INTER-structure coupling dimension:
  Time: T -----> 0
  Level 0: All conformations coupled (early, high noise)
  Level 1: Group-based coupling (middle)
  Level 2: Independent sampling (late, low noise)

Meanwhile, WITHIN each structure, CryoBoltz's multi-scale guidance
(point cloud -> density map) operates independently.
"""

import torch
import numpy as np
from typing import Optional


class HierarchicalCouplingSchedule:
    """Manages time-varying coupling hierarchy for multi-conformation sampling.

    The hierarchy defines how conformations are grouped and coupled at
    different stages of the diffusion process.

    Parameters
    ----------
    n_conformations : int
        Total number of conformations being jointly sampled.
    num_steps : int
        Total number of diffusion steps.
    level_transitions : list[int]
        Step indices (1-indexed) where hierarchy transitions occur.
        E.g., [80, 140] means:
          steps 1-80: Level 0 (all coupled)
          steps 81-140: Level 1 (group coupling)
          steps 141-end: Level 2 (independent)
    groups : list[list[int]], optional
        Conformation groupings for Level 1.
        E.g., [[0, 1], [2, 3]] means conformations 0,1 are in group A
        and 2,3 are in group B. If None, each conformation is its own group.
    strength_schedule : str
        How coupling strength decays: 'cosine', 'linear', or 'constant'.
    base_strength : float
        Maximum coupling strength at Level 0.
    """

    def __init__(
        self,
        n_conformations,
        num_steps,
        level_transitions=None,
        groups=None,
        strength_schedule='cosine',
        base_strength=1.0,
    ):
        self.n_conformations = n_conformations
        self.num_steps = num_steps
        self.base_strength = base_strength
        self.strength_schedule = strength_schedule

        # Default transitions: equal thirds
        if level_transitions is None:
            t1 = num_steps // 3
            t2 = 2 * num_steps // 3
            level_transitions = [t1, t2]
        self.level_transitions = level_transitions

        # Default groups: each conformation is its own group at Level 1
        if groups is None:
            groups = [[i] for i in range(n_conformations)]
        self.groups = groups

        # Precompute the schedule
        self._build_schedule()

    def _build_schedule(self):
        """Build the full coupling schedule over all timesteps."""
        self.coupling_strengths = torch.zeros(self.num_steps)
        self.active_levels = []

        for step in range(self.num_steps):
            step_1indexed = step + 1
            # Determine current level
            if step_1indexed <= self.level_transitions[0]:
                level = 0
            elif len(self.level_transitions) > 1 and step_1indexed <= self.level_transitions[1]:
                level = 1
            else:
                level = 2

            self.active_levels.append(level)

            # Compute strength based on schedule
            if level == 2:
                self.coupling_strengths[step] = 0.0
            else:
                # Determine the range for this level
                if level == 0:
                    start = 0
                    end = self.level_transitions[0] - 1
                else:  # level == 1
                    start = self.level_transitions[0]
                    end = self.level_transitions[1] - 1 if len(self.level_transitions) > 1 else self.num_steps - 1

                # Normalized position within this level [0, 1]
                level_len = max(end - start, 1)
                t = (step - start) / level_len

                if self.strength_schedule == 'cosine':
                    strength = self.base_strength * 0.5 * (1 + np.cos(np.pi * t))
                elif self.strength_schedule == 'linear':
                    strength = self.base_strength * (1 - t)
                else:  # constant
                    strength = self.base_strength

                # Level 1 has reduced base strength
                if level == 1:
                    strength *= 0.5

                self.coupling_strengths[step] = strength

    def get_coupling_info(self, step):
        """Get coupling information for a given step.

        Parameters
        ----------
        step : int
            0-indexed diffusion step.

        Returns
        -------
        dict with keys:
            'level': int - current hierarchy level (0, 1, or 2)
            'strength': float - coupling strength multiplier
            'active_groups': list[list[int]] - which conformations are coupled
        """
        level = self.active_levels[step]
        strength = self.coupling_strengths[step].item()

        if level == 0:
            # All conformations coupled
            active_groups = [list(range(self.n_conformations))]
        elif level == 1:
            # Group-based coupling
            active_groups = self.groups
        else:
            # Independent
            active_groups = [[i] for i in range(self.n_conformations)]

        return {
            'level': level,
            'strength': strength,
            'active_groups': active_groups,
        }

    def get_active_partners(self, step, conf_idx):
        """Get which conformations are coupled to conf_idx at this step.

        Parameters
        ----------
        step : int
            0-indexed diffusion step.
        conf_idx : int
            Index of the query conformation.

        Returns
        -------
        list[int]
            Indices of conformations coupled to conf_idx (excluding itself).
        float
            Coupling strength at this step.
        """
        info = self.get_coupling_info(step)
        strength = info['strength']

        if strength == 0:
            return [], 0.0

        # Find the group containing conf_idx
        for group in info['active_groups']:
            if conf_idx in group:
                partners = [j for j in group if j != conf_idx]
                return partners, strength

        return [], 0.0


def build_default_groups(n_conformations, strategy='pairs'):
    """Build default conformation groupings.

    Parameters
    ----------
    n_conformations : int
        Number of conformations.
    strategy : str
        Grouping strategy:
        - 'pairs': pair consecutive conformations
        - 'halves': split into two halves
        - 'singles': each conformation is its own group

    Returns
    -------
    list[list[int]]
        Groupings.
    """
    if strategy == 'pairs':
        groups = []
        for i in range(0, n_conformations, 2):
            if i + 1 < n_conformations:
                groups.append([i, i + 1])
            else:
                groups.append([i])
        return groups
    elif strategy == 'halves':
        mid = n_conformations // 2
        return [list(range(mid)), list(range(mid, n_conformations))]
    elif strategy == 'singles':
        return [[i] for i in range(n_conformations)]
    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")
