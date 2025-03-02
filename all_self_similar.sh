set -o pipefail
set -u

for i in {2..19}
do
  echo "Processing ${i}"

  mkdir -p /home/igor/rds/ephemeral/datasets/caches_new/cache${i}
  
  cargo run self-similar \
  --data-file /home/igor/rds/ephemeral/datasets/tokenized/slimpajama_${i}_of_20.train \
  --length-threshold 200 \
  --cache-dir /home/igor/rds/ephemeral/datasets/caches_new/cache${i} \
  --num-threads 80
done
