#!/bin/bash
# 시드 실험: A(v2 가중치) 시드 1~3, B(기준) 시드 2~3 (B 시드 1은 기존 e0509_2f85_sac_nostab.zip 사용)
# A/B를 번갈아 실행해 중간에 멈춰도 두 조건의 시드 수가 비슷하게 남도록 함. 메모리 때문에 순차 실행
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p seed_logs
for job in "A 1" "B 2" "A 2" "B 3" "A 3"; do
    set -- $job
    script=$([ "$1" = A ] && echo Doosan_E0509_train.py || echo Doosan_E0509_train_baseline.py)
    echo "[$(date '+%H:%M')] 시작: $1 시드 $2"
    $PY -u $script --seed $2 > seed_logs/${1}_s$2.log 2>&1
    echo "[$(date '+%H:%M')] 종료: $1 시드 $2 (exit $?)"
done
echo "[$(date '+%H:%M')] 전체 완료"
