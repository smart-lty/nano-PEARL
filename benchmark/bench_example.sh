# Evaluation parameters
DRAFT_TP=1
TARGET_TP=2
GPU_MEMORY_UTILIZATION=0.95
TEMPERATURE=0.0
MAX_TOKENS=200
IGNORE_EOS=true
NUM_SAMPLES=100
INPUT_LEN=1024
BATCH_SIZE=1
RUN_AR_BENCHMARK=true
SEED=0
VERBOSE=false

# Dataset options
DATASET="all"  # Options: HumanEval, CNNDM, AIME, GSM8K, all

draft_model=$1
target_model=$2
dataset=${3:-$DATASET}
mode=${4:-"benchmark"}

export TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error

if [ "$mode" = "benchmark" ]; then
python benchmark/eval_benchmark.py \
    --draft-model "$draft_model" \
    --target-model "$target_model" \
    --draft-tp $DRAFT_TP \
    --target-tp $TARGET_TP \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    $([ "$IGNORE_EOS" = true ] && echo "--ignore-eos") \
    --dataset "$dataset" \
    --max-samples $NUM_SAMPLES \
    --bs $BATCH_SIZE \
    $([ "$RUN_AR_BENCHMARK" = true ] && echo "--run-ar-benchmark") \
    --seed $SEED \
    $([ "$VERBOSE" = true ] && echo "--verbose")
fi

if [ "$mode" = "random" ]; then
    python benchmark/eval_random.py \
        --draft-model "$draft_model" \
        --target-model "$target_model" \
        --draft-tp $DRAFT_TP \
        --target-tp $TARGET_TP \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --temperature $TEMPERATURE \
        --max-tokens $MAX_TOKENS \
        $([ "$IGNORE_EOS" = true ] && echo "--ignore-eos") \
        --num-samples $NUM_SAMPLES \
        --input-len $INPUT_LEN \
        --bs $BATCH_SIZE \
        $([ "$RUN_AR_BENCHMARK" = true ] && echo "--run-ar-benchmark") \
        --seed $SEED \
        $([ "$VERBOSE" = true ] && echo "--verbose")
fi

