#!/bin/bash
# 시드 실험: A(안정성 페널티) 시드 1~3, B(기준) 시드 1~3을 순차 학습 (메모리 때문에 동시 실행 불가)
# A/B를 번갈아 실행해 중간에 멈춰도 두 조건의 시드 수가 비슷하게 남도록 함
# 진행 상황: logs/seed_runs.log, 실행별 로그: logs/seed_runs/
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs/seed_runs
for job in "A 1" "B 1" "A 2" "B 2" "A 3" "B 3"; do
    set -- $job
    flag=$([ "$1" = B ] && echo --baseline)
    echo "[$(date '+%H:%M')] 시작: $1 시드 $2" | tee -a logs/seed_runs.log
    $PY -u train.py $flag --seed $2 > logs/seed_runs/${1}_s$2.log 2>&1
    echo "[$(date '+%H:%M')] 종료: $1 시드 $2 (exit $?)" | tee -a logs/seed_runs.log
done
echo "[$(date '+%H:%M')] 전체 완료" | tee -a logs/seed_runs.log
