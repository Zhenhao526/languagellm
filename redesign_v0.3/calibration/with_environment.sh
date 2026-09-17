#!/bin/bash
set -euo pipefail
calibration_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME="$calibration_dir/runtime/zulu8.96.0.205-ca-jdk8.0.504-macosx_x64/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
export GRADLE_USER_HOME="$calibration_dir/runtime/gradle-cache"
export PYTHONPATH="$calibration_dir/vendor/minerl${PYTHONPATH:+:$PYTHONPATH}"
exec "$calibration_dir/.venv-env/bin/python" "$@"
