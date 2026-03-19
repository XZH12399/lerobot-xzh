#!/usr/bin/env bash
set -euo pipefail

export POLICY_TYPE=${POLICY_TYPE:-pi0_residual_every_step}
export RUN_PREFIX=${RUN_PREFIX:-pi0_residual_every_step_alpha15}
export JOB_NAME=${JOB_NAME:-pi0_residual_every_step_alpha15_pusht_sanity}
export BASE_ALPHA_OVERRIDE=${BASE_ALPHA_OVERRIDE:-1.5}

exec ./run_pi0_residual_pusht_sanity.sh $@
