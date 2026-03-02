set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

kit_dir    := ".codex-zephyr-mlsys"
launch_hf  := "./scripts/launch_mlsys_hf.sh"
zephyr_kit := ".sygaldry/zephyr"
repoctl    := zephyr_kit + "/bin/repoctl"

tts_env   := "qwen3-tts"
tts_venv  := "/workspace/audri/.venv-mlsys/" + tts_env
tts_model := "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
activate  := "unset PYTHONPATH && source " + tts_venv + "/bin/activate"

# ── zephyr infra ─────────────────────────────────────────────────────────────

# Verify the zephyr/mlsys kit is intact and policy-compliant
zephyr-check:
  test -x "{{repoctl}}" || (echo "missing repoctl: {{repoctl}}" && exit 1)
  test -f "{{kit_dir}}/runtime.yaml" || (echo "missing runtime config" && exit 1)
  test -f "{{zephyr_kit}}/infra.yaml" || (echo "missing infra config" && exit 1)
  ./scripts/zephyr_policy_check.sh
  echo "zephyr kit ok"

# Show resolved container/image config
zephyr-config:
  "{{repoctl}}" config show

# Run the spack policy checker
zephyr-policy-check:
  ./scripts/zephyr_policy_check.sh

# Build a named env (internal; prefer build-env / build-env-fast)
zephyr-build env no_validate="0": zephyr-check
  build_no_validate="{{no_validate}}"; if [[ "${build_no_validate}" == "1" ]]; then "{{launch_hf}}" "{{env}}" --no-validate; else "{{launch_hf}}" "{{env}}"; fi

# Layer packages into a named env (internal; prefer layer)
zephyr-layer env packages:
  ./scripts/zephyr_layer.sh "{{env}}" "{{packages}}"

# ── audri ─────────────────────────────────────────────────────────────────────

# Build the derived audri-tts container image (libsndfile1 + sox added to base)
image-build *args:
  "{{repoctl}}" image build {{args}}

# Build the qwen3-tts Python venv (runs full validation)
build-env:
  just zephyr-build env="{{tts_env}}"

# Build the qwen3-tts venv, skip validation (faster iteration)
build-env-fast:
  just zephyr-build env="{{tts_env}}" no_validate=1

# Layer extra packages into the qwen3-tts venv
# usage: just layer "pkg1 pkg2"
layer packages:
  just zephyr-layer env="{{tts_env}}" packages="{{packages}}"

# GPU info — name, compute capability, VRAM (no venv needed)
gpu-check:
  "{{repoctl}}" run -- python3 -c 'import torch; n=torch.cuda.device_count(); [print(f"[{i}] "+torch.cuda.get_device_properties(i).name+" CC "+str(torch.cuda.get_device_properties(i).major)+"."+str(torch.cuda.get_device_properties(i).minor)+" "+str(torch.cuda.get_device_properties(i).total_memory//1024**2)+"MB") for i in range(n)]; print("cuda:", torch.cuda.is_available(), "devices:", n)'

# Run Milestone 0 smoke test — model load + WAV synthesis → outputs/smoke_test.wav
smoke model=tts_model:
  "{{repoctl}}" run -- bash -c "{{activate}} && \
    python3 /workspace/audri/scripts/smoke_test_tts.py --model {{model}}"

# Run Milestone 1 streaming demo (headless — saves WAVs to outputs/)
demo model=tts_model:
  "{{repoctl}}" run -- bash -c "{{activate}} && \
    AUDRI_HEADLESS=1 python3 /workspace/audri/scripts/speak_demo.py --model {{model}}"

# Run an arbitrary command inside the container with the qwen3-tts venv active
# usage: just run "python3 -c 'import transformers; print(transformers.__version__)'"
run cmd:
  "{{repoctl}}" run -- bash -c "{{activate}} && {{cmd}}"
