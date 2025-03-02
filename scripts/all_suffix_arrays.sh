set -uoe pipefail

ROOT="/home/igor/rds/ephemeral/datasets"

for i in {11..19}
do
  echo "Processing ${i}"
  
  python scripts/make_suffix_array.py \
  --input-path "${ROOT}/tokenized/slimpajama_${i}_of_20.train" \
  --tmp-path "${ROOT}/tmps/tmp${i}" \
  --total-jobs-mult 8 &> "logs/suffix_array_${i}_of_20.log"

  tail -n 1 "logs/suffix_array_${i}_of_20.log" > "merge_scripts/merge_${i}_of_20.sh"

  mkdir -p "${ROOT}/tmps/tmp${i}"
  
  bash "merge_scripts/merge_${i}_of_20.sh" &> "logs/merge_${i}_of_20.log"
    
  
  rm ${ROOT}/tokenized/slimpajama_${i}_of_20.train.part.*

  cat ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00* > "${ROOT}/tmps/tmp${i}/out.table.bin"
  mv "${ROOT}/tmps/tmp${i}/out.table.bin" "${ROOT}/tokenized/slimpajama_${i}_of_20.train.table.bin"
  rm ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00*
done
