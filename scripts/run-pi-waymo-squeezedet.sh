#!/usr/bin/env bash
# Flink + OpenVINO SqueezeDet baseline on the Raspberry Pi (aarch64).
#
# This is the Flink counterpart of the XXXX-2 "UDF" object-detection case on the WAYMO dataset:
# it reads the same "timestamp,base64jpeg" CSV, decodes+resizes to 624x192, runs SqueezeDet
# (KITTI arch, 192x624 input, random weights — benchmark-only) through OpenVINO via the JNI
# bridge, discards detections, and writes latency/throughput stats.
# Use it to SUBSTITUTE the XXXX-2-UDF row in the WAYMO framework comparison.
#
# Fairness parity with the XXXX-2 side (queries/waymo/systems.test):
#   * PARALLELISM=2  <-> worker.query_engine.number_of_worker_threads: 2
#   * STREAMS=0, THREADS=1 <-> XXXX-2 compiles OpenVINO with num_streams(0),
#     inference_num_threads(1) (its defaults; see OpenVinoRuntimeBackend.cpp). The native
#     bridge passes 0 through as an EXPLICIT property value — rebuild the JNI lib
#     (scripts/build-native-pi.sh) after updating native/src/OpenVinoEngine.cpp.
#
# Prereqs on the Pi (see build-native-pi.sh):
#   * target/native-libs/libflink_openvino_bridge_jni.so   (built on the Pi)
#   * target/flink-openvino-baseline-1.0-SNAPSHOT.jar       (copied; arch-independent)
#   * cp.txt + the ~/.m2 dependency jars it references       (copied)
#   * the OpenVINO pip wheel (provides libopenvino + the ARM CPU plugin at runtime)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

OV_DIR="${OV_DIR:-/home/user/navi/models-benchmarking/venv/lib/python3.13/site-packages/openvino}"
MODEL="${MODEL:-/home/user/navi/models-benchmarking/models/squeezedet/squeezedet-192x624.xml}"
CSV="${CSV:-/home/user/navi/datasets/waymo/camera_5.csv}"
STATS="${STATS:-/home/user/navi/results/flink_openvino_stats.csv}"

DEVICE="${DEVICE:-CPU}"
PARALLELISM="${PARALLELISM:-2}"   # Flink parallelism == number of inference lanes (XXXX-2: 2 worker threads)
STREAMS="${STREAMS:-0}"           # OpenVINO num_streams, explicit 0 like XXXX-2 (negative = leave unset)
THREADS="${THREADS:-1}"           # OpenVINO inference_num_threads, like XXXX-2
WIDTH="${WIDTH:-624}"
HEIGHT="${HEIGHT:-192}"
BASE64_COL="${BASE64_COL:-1}"     # WAYMO CSV is "timestamp,base64jpeg"...
HAS_HEADER="${HAS_HEADER:-false}" # ...with no header row

# Comparison-key columns written into the stats row so it slots into the XXXX-2/TorchServe table
# (substituting the XXXX-2-UDF case). Override to match the labels you compare against.
QUERY_NAME="${QUERY_NAME:-flink_waymo_squeezedet}"
PARAM_NAME="${PARAM_NAME:-baseline}"
PARAM_VALUE="${PARAM_VALUE:-flink}"
REPETITION="${REPETITION:-0}"

# Fixed ingestion rate (tuples/s) to match the XXXX-2/TorchServe TCP source MOCK_TUPLE_RATE (WAYMO=10),
# so all three systems see the same offered load. Set RATE=0 for an unbounded (backpressure) source.
RATE="${RATE:-10}"

NATIVE_DIR="$PROJECT_DIR/target/native-libs"
JAR="$PROJECT_DIR/target/flink-openvino-baseline-1.0-SNAPSHOT.jar"

[[ -f "$NATIVE_DIR/libflink_openvino_bridge_jni.so" ]] || { echo "Missing native lib; run scripts/build-native-pi.sh"; exit 1; }
[[ -f "$JAR" ]] || { echo "Missing app jar: $JAR"; exit 1; }
[[ -f "$PROJECT_DIR/cp.txt" ]] || { echo "Missing classpath file: $PROJECT_DIR/cp.txt"; exit 1; }

# Convert ONNX -> OpenVINO IR on first use (needs ovc from the venv).
if [[ ! -f "$MODEL" ]]; then
  ONNX="${MODEL%.*}.onnx"
  [[ -f "$ONNX" ]] || { echo "Missing model (no XML and no ONNX): $MODEL / $ONNX"; exit 1; }
  echo "Converting $ONNX -> $MODEL with ovc"
  VENV_ROOT="${OV_DIR%%/lib/*}"   # .../venv/lib/pythonX/site-packages/openvino -> .../venv
  ( source "$VENV_ROOT/bin/activate" 2>/dev/null || true
    ovc "$ONNX" --output_model "$MODEL" --compress_to_fp16=False )
fi

mkdir -p "$(dirname "$STATS")"
cd "$PROJECT_DIR"
export LD_LIBRARY_PATH="$OV_DIR/libs:$NATIVE_DIR:${LD_LIBRARY_PATH:-}"

echo "Flink OpenVINO SqueezeDet: model=$MODEL device=$DEVICE parallelism=$PARALLELISM streams=$STREAMS threads=$THREADS ${WIDTH}x${HEIGHT}"
java \
  -Djava.library.path="$NATIVE_DIR" \
  -cp "$JAR:$(cat "$PROJECT_DIR/cp.txt")" \
  benchmark.openvino.OpenVinoCsvObjectDetectionJob \
  "$MODEL" "$CSV" "$DEVICE" "$PARALLELISM" "$STREAMS" "$THREADS" \
  "$BASE64_COL" "$HAS_HEADER" "$WIDTH" "$HEIGHT" false "$STATS" \
  "$QUERY_NAME" "$PARAM_NAME" "$PARAM_VALUE" "$REPETITION" "$RATE"

echo "Wrote stats: $STATS"
