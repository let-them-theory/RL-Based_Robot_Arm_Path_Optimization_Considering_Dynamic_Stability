import argparse
import multiprocessing as mp
import time
import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from Doosan_E0509_train import E0509Env

# ==============================================================================
# A/B 모델 나란히 보기: 모델마다 GUI 창 1개(별도 프로세스), 같은 목표/초기자세에서 동시에 출발
#   - 화면 표시: 경과 시간, 목표까지 거리, TCP 속도, 동적 베이스 반력(제어 주기 평균), 성공 여부
#   - 성공 후 hold 시간 동안 계속 실행해 도달 후 움직임도 확인
#   - PyBullet 화면 글자는 한글 미지원 → 영어 표시
# ==============================================================================


def worker(label, model_path, seeds, hold, speed, barrier, render=True):
    env = E0509Env(render=render)
    model = SAC.load(model_path, device="cpu")
    hold_steps = int(round(hold / env.control_dt))
    color = [0.1, 0.3, 0.9] if label.startswith("A") else [0.85, 0.1, 0.1]
    text_ids = {}

    def show(key, text, pos, size, rgb=color):
        kwargs = {"replaceItemUniqueId": text_ids[key]} if key in text_ids else {}
        text_ids[key] = p.addUserDebugText(text, pos, textColorRGB=rgb, textSize=size, **kwargs)

    n_success = 0
    for ep_idx, seed in enumerate(seeds):
        barrier.wait()  # 두 창이 같은 목표에서 동시에 출발
        obs, _ = env.reset(seed=seed)
        show("title", label, [0.0, -0.7, 1.25], 2.0)
        show("result", f"target #{ep_idx + 1} (seed {seed})", [0.0, -0.7, 0.95], 1.4, [0.2, 0.2, 0.2])
        reached_at = None
        k = 0
        while True:
            t0 = time.time()
            P_before = env.prev_momentum[0].copy()
            obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
            k += 1
            force = np.linalg.norm(env.prev_momentum[0] - P_before) / env.control_dt
            show("status", f"t={k * env.control_dt:5.2f}s  dist={info['distance'] * 1000:6.1f}mm  "
                           f"v={info['ee_speed']:.3f}m/s  F_dyn={force:5.1f}N",
                 [0.0, -0.7, 1.1], 1.3, [0.1, 0.1, 0.1])
            if reached_at is None and (terminated or truncated):
                reached_at = k
                if terminated:
                    n_success += 1
                    show("result", f"REACHED in {k * env.control_dt:.2f}s  (success {n_success}/{ep_idx + 1})",
                         [0.0, -0.7, 0.95], 1.4, [0.0, 0.6, 0.0])
                else:
                    show("result", f"FAILED  (success {n_success}/{ep_idx + 1})", [0.0, -0.7, 0.95], 1.4, [0.8, 0.0, 0.0])
                    break
            if reached_at is not None and k - reached_at >= hold_steps:
                break
            if render:
                time.sleep(max(0.0, env.control_dt / speed - (time.time() - t0)))
        if render:
            time.sleep(1.0)
    barrier.wait()
    print(f"{label}: 성공 {n_success}/{len(seeds)}")
    if render:
        time.sleep(3.0)
    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0509 A/B 모델 동시 GUI 비교")
    parser.add_argument("--models", nargs=2, default=["e0509_2f85_sac_v2_s1.zip", "e0509_2f85_sac_nostab_s2.zip"])
    parser.add_argument("--labels", nargs=2, default=["A: stability penalty", "B: baseline (no penalty)"])
    parser.add_argument("--seed0", type=int, default=10000, help="비교 평가(compare)와 같은 목표를 보려면 10000")
    parser.add_argument("--n", type=int, default=10, help="볼 목표 개수")
    parser.add_argument("--seeds", type=int, nargs="*", help="특정 목표 seed만 보기 (예: --seeds 10041)")
    parser.add_argument("--hold", type=float, default=2.0, help="성공 후 계속 관찰할 시간 (s)")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율 (0.5 = 슬로모션)")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(args.seed0, args.seed0 + args.n))
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    procs = [ctx.Process(target=worker, args=(label, path, seeds, args.hold, args.speed, barrier))
             for label, path in zip(args.labels, args.models)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
