#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-run}"
SIGNING_IDENTITY="${AGENTTRAINER_SIGNING_IDENTITY:-1E3F98C6916444AFAAD362575B0DA874E73D992E}"
if ! security find-identity -v -p codesigning | /usr/bin/grep "$SIGNING_IDENTITY" | /usr/bin/grep -vq REVOKED; then
  echo 'The configured Apple Development certificate is unavailable. No ad hoc fallback is permitted.' >&2
  exit 1
fi
case "$MODE" in run|--debug|--logs|--telemetry|--verify|--build|--release|--test) ;; *) echo 'Usage: build_and_run.sh [--build|--release|--test|--debug|--logs|--telemetry|--verify]' >&2; exit 2 ;; esac
if [[ "$MODE" != --build && "$MODE" != --release && "$MODE" != --test ]]; then
  pkill -x AgentTrainer >/dev/null 2>&1 || true
fi
ACTION=build
CONFIGURATION=Debug
[[ "$MODE" == --release ]] && CONFIGURATION=Release
[[ "$MODE" == --test ]] && ACTION=test
TEST_OPTIONS=(-quiet)
if [[ "$MODE" == --test ]]; then
  TEST_OPTIONS=(-test-timeouts-enabled YES -default-test-execution-time-allowance 60 -maximum-test-execution-time-allowance 1200)
fi
xcodebuild -project AgentTrainer.xcodeproj -scheme AgentTrainer -configuration "$CONFIGURATION" \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath build \
  CODE_SIGN_IDENTITY="$SIGNING_IDENTITY" DEVELOPMENT_TEAM=V5S6J7KC33 CODE_SIGN_STYLE=Manual "${TEST_OPTIONS[@]}" "$ACTION"
APP_BUNDLE="$ROOT_DIR/build/Build/Products/$CONFIGURATION/AgentTrainer.app"
codesign --verify --deep --strict "$APP_BUNDLE"
if [[ "$MODE" == --build || "$MODE" == --release || "$MODE" == --test ]]; then exit 0; fi
if [[ "$MODE" == --debug ]]; then
  lldb -- "$APP_BUNDLE/Contents/MacOS/AgentTrainer"
  exit
fi
/usr/bin/open -n "$APP_BUNDLE"
case "$MODE" in
  --verify) sleep 2; pgrep -x AgentTrainer >/dev/null ;;
  --logs) /usr/bin/log stream --info --style compact --predicate 'process == "AgentTrainer"' ;;
  --telemetry) /usr/bin/log stream --info --style compact --predicate 'subsystem == "com.agenttrainer.AgentTrainer"' ;;
esac
