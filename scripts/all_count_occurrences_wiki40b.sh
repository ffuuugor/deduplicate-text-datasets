set -uoe pipefail

ROOT="/data/igor/dedup_data/"
DATASET="wiki40b"
PARTS="2"

mkdir -p "${ROOT}/queries/counts"

for i in {0..1}
do
    ./target/debug/dedup_dataset count-occurrences-multi \
    --data-file "${ROOT}/tokenized/${DATASET}_${i}_of_${PARTS}.train" \
    --query-file "${ROOT}/queries/query" &> "${ROOT}/queries/counts/${i}_of_${PARTS}.cnt"
done