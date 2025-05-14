set -o pipefail
set -u

ROOT="/data/igor/dedup_data"
DATASET="wiki40b"
PARTS="2"

for i in {0..1}
do
  echo "Processing ${i}"

  mkdir -p "${ROOT}/caches/cache${i}"
  
  cargo run self-similar \
  --data-file "${ROOT}/tokenized/${DATASET}_${i}_of_${PARTS}.train" \
  --length-threshold 200 \
  --cache-dir "${ROOT}/caches/cache${i}" \
  --num-threads 80
done
