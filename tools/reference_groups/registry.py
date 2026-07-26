from __future__ import annotations

from reference_groups.fit_compile import replay as replay_fit_compile
from reference_groups.io import replay as replay_io
from reference_groups.model_project import replay as replay_model_project
from reference_groups.physics import replay as replay_physics


GROUP_REGISTRY: dict[str, object] = {
    "model_project": replay_model_project,
    "io": replay_io,
    "physics": replay_physics,
    "fit_compile": replay_fit_compile,
}
