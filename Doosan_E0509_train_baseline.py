import time
import argparse

# Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from Doosan_E0509_train import E0509Env

# ==============================================================================
# 비교 실험 B: 안정성 페널티(토크/베이스 반력/스무딩/jerk)를 뺀 기준 모델
#   - 보상: 거리 + 바닥 충돌 + 도달 보너스만 사용
#   - 성공 조건(5mm + 2cm/s 정착), 환경, 하이퍼파라미터, 스텝 수, 시드는 제안 모델(A)과 동일
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="환경과 SAC(신경망 초기값/탐험/리플레이 샘플링)를 함께 고정")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    args = parser.parse_args()

    NUM_CPU = 14
    TOTAL_TIMESTEPS = args.timesteps
    MODEL_NAME = f"e0509_2f85_sac_nostab_s{args.seed}"

    print("-" * 60)
    print(f"🚀 [E0509 Baseline - 안정성 페널티 없음] 총 {TOTAL_TIMESTEPS:,} 스텝, 환경 {NUM_CPU}개, 시드 {args.seed}")
    print("-" * 60)

    env = make_vec_env(
        E0509Env,
        n_envs=NUM_CPU,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'render': False, 'stability': False}
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        buffer_size=1000000,
        batch_size=512,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        seed=args.seed,
        tensorboard_log="./e0509_2f85_nostab_logs/"
    )

    # 약 50만 스텝마다 체크포인트 저장 (save_freq는 VecEnv 호출 단위)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // NUM_CPU, 1),
        save_path=f"./checkpoints_e0509_2f85_nostab_s{args.seed}/",
        name_prefix=MODEL_NAME
    )

    try:
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, tb_log_name=f"SAC_s{args.seed}")
        end_time = time.time()
        print(f"✅ 학습 완료! 소요 시간: {(end_time - start_time)/3600:.2f}시간")
    except KeyboardInterrupt:
        print("\n🛑 학습 중단. 현재까지의 모델을 저장합니다.")

    model.save(MODEL_NAME)
    print(f"💾 최종 모델 저장 완료: {MODEL_NAME}.zip")

    env.close()
