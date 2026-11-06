#!/usr/bin/env bash
set -euo pipefail
# Run on a prepared Linux CUDA environment with the pinned local model.
if [[ $# != 1 ]]; then
    echo "Usage: bash examples/local_qwen_experiment.sh /absolute/run-directory" >&2
    exit 2
fi
p="$(cd "$1" && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -x "$p/venv/bin/python"
test -s "$p/qwen/model.safetensors"
for name in prompts evidence runs queries logs projection; do
    if [[ -e "$p/$name" ]]; then
        echo "Use a fresh run directory. Found $p/$name" >&2
        exit 2
    fi
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python="$p/venv/bin/python"
rev=7ae557604adf67be50417f59c2c2f167def9a775
mkdir -p "$p/runs"
fit() {
    local name=$1 seed=$2 weight=$3 lr=$4 steps=$5 interval=$6 student=$7
    shift 7
    "$python" examples/local_qwen_measure.py -- --threads 4 fit \
        --train "${QWEN_TRAIN:-$p/evidence/train.jsonl}" --validation "$p/evidence/validation.jsonl" \
        --teacher "$p/qwen" --revision "$rev" --student-config "$student" \
        --output "$p/runs/$name" --init random --trainable all --steps "$steps" \
        --eval-every "$interval" --save-every "$interval" --learning-rate "$lr" \
        --gradient-accumulation-steps 4 --topk-weight "$weight" --max-seq-length 128 \
        --device cuda --seed "$seed" "$@" > "$p/logs/$name.json" 2> "$p/logs/$name.log"
}

measure() { "$python" examples/local_qwen_measure.py -- --threads 4 "$@"; }
mkdir -p "$p/evidence" "$p/logs" "$p/v020"
git archive c7df0fe531c9b04ec08c19db26bce69c722eeea8 | tar -xf - -C "$p/v020"
"$python" examples/local_qwen_prompts.py --output "$p/prompts"
collect_args=(collect --prompts "$p/prompts/train.jsonl" --model "$p/qwen" --revision "$rev"
    --output "$p/evidence/train.jsonl" --device cuda --max-new-tokens 24 --top-k 20 --seed 42)
status=0
"$python" examples/local_qwen_measure.py --interrupt-after 48 -- --threads 4 "${collect_args[@]}" \
    > "$p/logs/collect-train-partial.json" 2> "$p/logs/collect-train-partial.log" || status=$?
test "$status" = 1
grep -q 'intentional interruption' "$p/logs/collect-train-partial.log"
test "$(wc -l < "$p/evidence/train.jsonl")" = 48
cp "$p/evidence/train.jsonl" "$p/evidence/train-prefix.jsonl"
cp "$p/evidence/train.jsonl.collection.json" "$p/evidence/train-prefix.journal.json"
measure "${collect_args[@]}" --resume > "$p/logs/collect-train.json" 2> "$p/logs/collect-train.log"
head -n 48 "$p/evidence/train.jsonl" | cmp - "$p/evidence/train-prefix.jsonl"
measure collect --prompts "$p/prompts/validation.jsonl" --model "$p/qwen" --revision "$rev" \
    --output "$p/evidence/validation.jsonl" --device cuda --max-new-tokens 24 --top-k 20 --seed 42 \
    > "$p/logs/collect-validation.json" 2> "$p/logs/collect-validation.log"

student=examples/qwen-student
PYTHONPATH="$p/v020/src" fit engine-old 17 1 .001 64 64 "$student"
fit engine-new 17 1 .001 64 64 "$student"
for seed in 17 42; do
    for weight in 0 1; do
        fit "paired-$seed-$weight" "$seed" "$weight" .001 384 96 "$student" \
            --selection-metric continuation_cross_entropy
    done
done
mkdir -p "$p/student128"
"$python" - "$student/config.json" "$p/student128/config.json" <<'PYTHON'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
config.update(hidden_size=128, intermediate_size=256)
Path(sys.argv[2]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PYTHON
fit lowlr-17 17 1 .0003 384 96 "$student" --selection-metric continuation_cross_entropy
fit wide-17 17 1 .001 384 96 "$p/student128" --selection-metric continuation_cross_entropy
fit resume-17 17 1 .001 192 96 "$student" --selection-metric continuation_cross_entropy
cp "$p/logs/resume-17.log" "$p/logs/resume-17-first.log"
cp "$p/logs/resume-17.json" "$p/logs/resume-17-first.json"
fit resume-17 17 1 .001 384 96 "$student" --selection-metric continuation_cross_entropy --resume

"$python" - "$p" > "$p/resume-check.json" <<'PYTHON'
import json
import sys
from pathlib import Path

import torch
from rephix.io import file_sha256

root = Path(sys.argv[1])
full, resumed = (root / "runs" / name for name in ("paired-17-1", "resume-17"))

def identical(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    return left == right

left, right = [torch.load(run/"training_state.pt", map_location="cpu", weights_only=True)
               for run in (full, resumed)]
fields = ("model", "optimizer", "order", "sample_index", "python_rng", "torch_rng", "cuda_rng",
          "completed_steps", "best_step", "best_validation", "history")
matches = {field: identical(left[field], right[field]) for field in fields}
checkpoints = [run / json.loads((run/"metrics.json").read_text())["best_checkpoint_relative"]
               for run in (full,resumed)]
weight_hashes = [file_sha256(path/"model.safetensors") for path in checkpoints]
assert all(matches.values()), matches
assert weight_hashes[0] == weight_hashes[1], weight_hashes
print(json.dumps({"identical": matches, "best_checkpoint_sha256": weight_hashes[0]}, indent=2))
PYTHON

mkdir -p "$p/queries"
"$python" - "$p" > "$p/queries/selector.txt" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
names=("paired-17-0","paired-17-1","lowlr-17","wide-17")
print(min(names,key=lambda n: json.loads((p/"runs"/n/"metrics.json").read_text())["best_validation"]["continuation_cross_entropy"]))
PY
selector="$(cat "$p/queries/selector.txt")"
for strategy in random entropy; do
    scoring=()
    if [[ "$strategy" == entropy ]]; then scoring=(--model "$p/runs/$selector"); fi
    measure select-queries --pool "$p/prompts/pool.jsonl" \
        --exclude "$p/prompts/train.jsonl" --exclude "$p/prompts/validation.jsonl" \
        --exclude "$p/prompts/test.jsonl" --strategy "$strategy" --budget 10 \
        --seed 92 --device cuda --max-seq-length 128 "${scoring[@]}" \
        --output "$p/queries/$strategy-prompts.jsonl" > "$p/logs/select-$strategy.json" 2> "$p/logs/select-$strategy.log"
    measure collect --prompts "$p/queries/$strategy-prompts.jsonl" \
        --model "$p/qwen" --revision "$rev" --output "$p/queries/$strategy.jsonl" \
        --device cuda --max-new-tokens 24 --top-k 20 --seed 42 \
        > "$p/logs/collect-$strategy.json" 2> "$p/logs/collect-$strategy.log"
done
"$python" - "$p" <<'PY'
from dataclasses import replace
from pathlib import Path
import sys
from transformers import AutoTokenizer
from rephix.evidence import assert_disjoint, read_observations, write_observations
from rephix.hf import _load_tokenizer
p=Path(sys.argv[1])
train=read_observations(p/"evidence/train.jsonl")
val=read_observations(p/"prompts/validation.jsonl")
test=read_observations(p/"prompts/test.jsonl")
tokenizer=_load_tokenizer(AutoTokenizer,p/"qwen",local_files_only=True)
strategies=("random","entropy")
arms=[read_observations(p/"queries"/f"{s}.jsonl") for s in strategies]
lengths=[min(len(row.completion_token_ids) for row in pair) for pair in zip(*arms, strict=True)]
for strategy, rows in zip(strategies,arms,strict=True):
    assert_disjoint(train,val,test,rows)
    capped=[]
    for row, cap in zip(rows,lengths,strict=True):
        tokens=row.completion_token_ids[:cap]
        capped.append(replace(row, completion_token_ids=tokens, top_logprobs=row.top_logprobs[:cap],
                              output=tokenizer.decode(tokens,skip_special_tokens=False)))
    write_observations(p/"queries"/f"{strategy}-train.jsonl", train+capped)
print("Per-index supervised continuation caps", lengths, "total", sum(lengths))
PY
for strategy in random entropy; do
    QWEN_TRAIN="$p/queries/$strategy-train.jsonl" fit "query-$strategy" 17 1 .001 384 96 examples/qwen-student \
        --selection-metric continuation_cross_entropy
done
fit engine-new-recheck 17 1 .001 64 64 examples/qwen-student
PYTHONPATH="$p/v020/src" fit engine-old-recheck 17 1 .001 64 64 examples/qwen-student

cp "$p/logs/lowlr-17.log" "$p/logs/lowlr-17-first.log"
cp "$p/logs/lowlr-17.json" "$p/logs/lowlr-17-first.json"
fit lowlr-17 17 1 .0003 768 96 examples/qwen-student --selection-metric continuation_cross_entropy --resume
"$python" - "$p" <<'PY'
import json, sys
from pathlib import Path
from rephix.evidence import assert_disjoint, read_observations
from rephix.io import file_sha256, write_json
p=Path(sys.argv[1])
names=("paired-17-0","paired-17-1","paired-42-0","paired-42-1","lowlr-17","wide-17")
score="continuation_cross_entropy"
models={n: json.loads((p/"runs"/n/"metrics.json").read_text()) for n in names}
winner=min(names,key=lambda n: (models[n]["best_validation"][score], n))
best=models[winner]
assert_disjoint(*(read_observations(p/"prompts"/f"{s}.jsonl") for s in ("train","validation","test","pool")))
write_json(p/"selection-lock.json", {
    "validation_rule": "Minimum continuation cross entropy over base-corpus configurations, ties by run name",
    "scores": {name: value["best_validation"][score] for name,value in models.items()},
    "winner": winner,
    "best_step": best["best_step"],
    "checkpoint_safetensors_sha256": file_sha256(Path(best["best_checkpoint"])/"model.safetensors"),
    "validation_sha256": best["validation_sha256"],
    "reserved_test_prompts_sha256": file_sha256(p/"prompts/test.jsonl"),
})
PY
winner="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$p/selection-lock.json")"
measure collect --prompts "$p/prompts/test.jsonl" --model "$p/qwen" --revision "$rev" \
    --output "$p/evidence/test.jsonl" --device cuda --max-new-tokens 24 --top-k 20 --seed 42 \
    > "$p/logs/collect-test.json" 2> "$p/logs/collect-test.log"
measure evaluate --data "$p/evidence/test.jsonl" --model "$p/runs/$winner" \
    --output "$p/test.json" --device cuda --max-seq-length 128 --max-new-tokens 24 \
    --generate --generation-details > "$p/logs/evaluate-test.json" 2> "$p/logs/evaluate-test.log"
measure recover-projection --model "$p/qwen" --revision "$rev" \
    --prompts examples/projection-prompts.jsonl --output "$p/projection" \
    --max-seq-length 64 --max-samples 4096 --vocab-limit 32 --max-memory-mb 512 \
    --device cuda --seed 42 > "$p/logs/projection.json" 2> "$p/logs/projection.log"
