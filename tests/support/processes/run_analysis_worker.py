from __future__ import annotations

import xrr_fitter.api as api


def start(
    project: api.XrrProject,
    dataset_id: str,
    candidate_id: str,
) -> api.OperationJob:
    return api.start_mcmc_job(
        project,
        dataset_id,
        candidate_id,
        api.McmcConfig(walkers=4, burn_in=0, production_steps=4),
    )
