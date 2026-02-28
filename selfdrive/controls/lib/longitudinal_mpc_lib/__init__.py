"""
Legacy Longitudinal MPC Library

DEPRECATED: This module is part of the classical MPC-based longitudinal control system.
It is no longer the default in openpilot as of Phase 2: Unified Neural Execution.

The default longitudinal controller now uses End-to-End (E2E) neural model outputs
directly from modelV2.action.desiredAcceleration.

This module is retained for:
- Backward compatibility when ForceClassicalMPC parameter is enabled
- Reference and comparison purposes
- Gradual transition support

New development should focus on improving the E2E model rather than this MPC implementation.
"""

# Deprecation notice is documented in the module docstring
# Runtime warnings are emitted when LongitudinalMpc class is instantiated (see long_mpc.py)

from .long_mpc import LongitudinalMpc, LongitudinalPlanSource, T_IDXS

__all__ = ['LongitudinalMpc', 'LongitudinalPlanSource', 'T_IDXS']
