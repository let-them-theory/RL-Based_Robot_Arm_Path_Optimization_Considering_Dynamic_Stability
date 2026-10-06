import argparse
import os
import time

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from e0509_env import E0509Env

# ==============================================================================
# SAC 학습 (두 조건을 하나의 스크립트로)
#   - 기본: A = 안정성 페널티 포함 보상 → models/stability_s{seed}.zip
#   - --baseline: B = 안정성 페널티 제외 (거리 + 바닥 접촉 + 도달만) → models/baseline_s{seed}.zip
#   - 두 조건은 보상만 다르고 환경, 성공 조건, 하이퍼파라미터, 학습량은 동일
#   - 체크포인트: checkpoints/<이름>/ (약 50만 스텝마다), 텐서보드: logs/<조건>/s{seed}
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0509 도달 SAC 학습")
    parser.add_argument("--baseline", action="store_true", help="안정성 페널티 없이 학습 (비교 기준 B)")
    parser.add_argument("--seed", type=int, default=1, help="환경과 SAC(신경망 초기값/탐험/리플레이 샘플링)를 함께 고정")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=14, help="병렬 환경 수 (환경당 메모리 약 620MB)")
    args = parser.parse_args()

    condition = "baseline" if args.baseline else "stability"
    name = f"{condition}_s{args.seed}"
    os.makedirs("models", exist_ok=True)

    print("-" * 60)
    print(f"🚀 [E0509 {'B: 기준 (안정성 페널티 없음)' if args.baseline else 'A: 안정성 페널티'}] "
          f"총 {args.timesteps:,} 스텝, 환경 {args.n_envs}개, 시드 {args.seed}")
    print("-" * 60)

    # make_vec_env가 각 환경을 Monitor로 감싸므로 VecMonitor는 사용하지 않음
    env = make_vec_env(
        E0509Env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render": False, "stability": not args.baseline},
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        buffer_size=1000000,
        batch_size=512,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        seed=args.seed,
        tensorboard_log=f"./logs/{condition}/",
    )

    # 약 50만 스텝마다 체크포인트 저장 (save_freq는 VecEnv 호출 단위)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // args.n_envs, 1),
        save_path=f"./checkpoints/{name}/",
        name_prefix=name,
    )

    try:
        start_time = time.time()
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, tb_log_name=f"s{args.seed}")
        print(f"✅ 학습 완료! 소요 시간: {(time.time() - start_time) / 3600:.2f}시간")
    except KeyboardInterrupt:
        print("\n🛑 학습 중단. 현재까지의 모델을 저장합니다.")

    model.save(f"models/{name}")
    print(f"💾 최종 모델 저장 완료: models/{name}.zip")

    env.close()
