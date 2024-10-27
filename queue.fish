#!/usr/bin/fish
source .venv/bin/activate.fish
export HSA_OVERRIDE_GFX_VERSION=10.3.0
rq worker --with-scheduler