set -uoe pipefail
ulimit -Sn 1000000

ROOT="/data/igor/dedup_data"

mkdir -p "logs"
mkdir -p "merge_scripts"

for i in {0..1}
do
  echo "Processing ${i}"
  mkdir -p "${ROOT}/tmps/tmp${i}"
  
  python py_src/dedupe/make_suffix_array.py \
  --input-path "${ROOT}/tokenized/wiki40b_${i}_of_2.train" \
  --tmp-path "${ROOT}/tmps/tmp${i}" \
  --total-jobs-mult 8 &> "logs/suffix_array_${i}_of_2.log"

  tail -n 1 "logs/suffix_array_${i}_of_2.log" > "merge_scripts/merge_${i}_of_2.sh"

  bash "merge_scripts/merge_${i}_of_2.sh" &> "logs/merge_${i}_of_2.log"
    
  
  rm ${ROOT}/tokenized/wiki40b_${i}_of_2.train.part.*

  cat ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00* > "${ROOT}/tmps/tmp${i}/out.table.bin"
  mv "${ROOT}/tmps/tmp${i}/out.table.bin" "${ROOT}/tokenized/wiki40b_${i}_of_2.train.table.bin"
  rm ${ROOT}/tmps/tmp${i}/out.table.bin.table.bin.00*
done
