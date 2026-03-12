# Navigation module for E2E Phase 4
"""
E2E Phase 4: Navigation Conditioning

This module provides navigation embeddings that condition the E2E driving model
to follow routes and make navigation-aware decisions.

Architecture:
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Navigation     │────▶│  nav_embeddingd  │────▶│   modeld     │
│  (route, instr) │     │  (embedding gen) │     │  (policy net)│
└─────────────────┘     └──────────────────┘     └──────────────┘
                               │                        │
                               │ 64-dim embedding       │ nav_embeddings input
                               │ @ 20Hz                 │ concatenated with features

Embedding Contents (64 dimensions):
- [0]: Distance to next maneuver (normalized 0-1)
- [1-10]: Maneuver type (one-hot: turn-left, turn-right, merge, exit, fork, etc.)
- [11-13]: Route curvature at 100m, 500m, 1km
- [14]: Speed limit difference
- [15]: Lane position preference
- [16]: Route progress
- [17-20]: Road type (one-hot: highway, urban, residential, rural)
- [21-64]: Route geometry encoding (relative positions)

Status: ✅ Infrastructure Complete
- ✅ nav_embeddingd process generates embeddings
- ✅ modeld.py receives and forwards embeddings to policy network
- ✅ modelV2.navEmbeddings published for logging
- ⏳ Model training required for navigation-aware behavior

Integration:
The navigation embeddings are automatically fed into the policy network when:
1. nav_embeddingd process is running (started by manager when onroad)
2. Model is trained to accept nav_embeddings input (shape: 1x64)

Until the model is trained with nav_embeddings, the network will receive
zero embeddings and behave as before. Once trained, the model will make
navigation-aware decisions (e.g., lane changes for exits, slowing for turns).
"""

from openpilot.selfdrive.nav.nav_embedding import create_route_embedding, EMBEDDING_DIM

__all__ = ['create_route_embedding', 'EMBEDDING_DIM']
