from __future__ import annotations

from reference_groups.io import replay as replay_io
from reference_groups.model_project import replay as replay_model_project


GROUP_REGISTRY: dict[str, object] = {
    "model_project": replay_model_project,
    "io": replay_io,
}
