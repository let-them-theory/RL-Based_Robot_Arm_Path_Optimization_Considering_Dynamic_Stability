import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from Doosan_E0509_train import E0509Env

# ==============================================================================
# 실험적 주파수 응답 (보드 선도): 목표를 한 축으로 사인파 이동 → TCP 추종과 베이스 반력 측정
#   - 정책+로봇 폐루프는 비선형이므로 엄밀한 LTI 보드 선도가 아니라, 주어진 진폭에서의
#     기술 함수(describing function). 진폭을 바꾸면 결과가 달라질 수 있음
#   - 추종 전달함수  H(jw) = X_tcp / X_target          (이득 dB, 위상 deg)
#   - 반력 전달률    T(jw) = F_base / X_target  (N/m)  (같은 입력에 베이스가 얼마나 흔들리는가)
#   - 신호는 제어 주기(20Hz) 샘플, 정수 주기 구간에서 해당 주파수 성분만 단일 주파수 DFT로 추출
# ==============================================================================
AXES = {"x": 0, "y": 1, "z": 2}


def frequency_response(env, policy, freq, center, axis, amp, seed, settle_steps=60, min_cycles=4, min_time=4.0):
    dt = env.control_dt
    obs, _ = env.reset(seed=seed)
    env.target_pos = center.copy()
    obs = env._get_obs()
    for _ in range(settle_steps):  # 중심 목표에 먼저 정착
        obs, *_ = env.step(policy(obs))

    period = 1.0 / freq
    warmup = max(1, int(np.ceil(1.0 / period)))  # 초기 과도 응답 제거용 (최소 1초, 정수 주기)
    cycles = max(min_cycles, int(np.ceil(min_time / period)))
    n_total = int(round((warmup + cycles) * period / dt))
    n_warm = int(round(warmup * period / dt))

    u, y, P = [], [], [env.prev_momentum[0].copy()]
    for k in range(n_total):
        t = (k + 1) * dt
        env.target_pos = center.copy()
        env.target_pos[axis] += amp * np.sin(2 * np.pi * freq * t)
        obs = env._get_obs()  # 바뀐 목표를 반영한 관측 (rel_pos)
        obs, *_ = env.step(policy(obs))  # 종료 조건은 무시하고 계속 실행
        u.append(env.target_pos[axis] - center[axis])
        y.append(obs[2 * env.n_joints + axis] - center[axis])
        P.append(env.prev_momentum[0].copy())
    F = np.diff(np.array(P), axis=0)[:, axis] / dt  # 제어 주기 평균 동적 반력 (가진 축 성분)

    t = (np.arange(n_total) + 1) * dt
    w = np.exp(-2j * np.pi * freq * t[n_warm:])
    U = np.sum(np.array(u)[n_warm:] * w)
    Y = np.sum((np.array(y)[n_warm:] - np.mean(y[n_warm:])) * w)
    Fw = np.sum((F[n_warm:] - np.mean(F[n_warm:])) * w)
    return Y / U, Fw / U, np.abs(np.diff(y)).max() / dt


def bandwidth(freqs, gain_db):
    """이득이 처음으로 -3dB 아래로 내려가는 주파수 (로그 보간)"""
    below = np.where(gain_db < -3)[0]
    if not len(below) or below[0] == 0:
        return np.nan
    i = below[0]
    f0, f1, g0, g1 = freqs[i - 1], freqs[i], gain_db[i - 1], gain_db[i]
    return 10 ** (np.log10(f0) + (-3 - g0) / (g1 - g0) * (np.log10(f1) - np.log10(f0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0509 모델 실험적 주파수 응답 비교")
    parser.add_argument("--models", nargs=2, default=["e0509_2f85_sac.zip", "e0509_2f85_sac_nostab.zip"])
    parser.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    parser.add_argument("--center", nargs=3, type=float, default=[0.45, 0.0, 0.35], help="가진 중심 목표 (m)")
    parser.add_argument("--axis", choices=AXES, default="x")
    parser.add_argument("--amp", type=float, default=0.02, help="가진 진폭 (m)")
    parser.add_argument("--fmin", type=float, default=0.05)
    parser.add_argument("--fmax", type=float, default=4.0)
    parser.add_argument("--nfreq", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=3, help="주파수마다 초기 자세를 바꿔 반복 후 복소 평균")
    parser.add_argument("--out", default="compare_results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = E0509Env()
    center = np.array(args.center)
    axis = AXES[args.axis]
    freqs = np.geomspace(args.fmin, args.fmax, args.nfreq)

    res = {}
    for label, path in zip(args.labels, args.models):
        model = SAC.load(path, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
        H, T, vmax = [], [], []
        for f in freqs:
            out = [frequency_response(env, policy, f, center, axis, args.amp, seed=20000 + r) for r in range(args.repeats)]
            H.append(np.mean([o[0] for o in out]))
            T.append(np.mean([o[1] for o in out]))
            vmax.append(np.max([o[2] for o in out]))
        res[label] = {"H": np.array(H), "T": np.array(T), "vmax": np.array(vmax)}
        print(f"{label} 완료")

    # 진폭 A의 사인파 추종에 필요한 최대 속도 = 2πfA. 이 값이 TCP 속도 한계(제어 주기 평균 약 0.25m/s)를
    # 넘는 주파수부터는 속도 포화로 응답이 비선형이 됨
    f_slew = 0.25 / (2 * np.pi * args.amp)

    with open(os.path.join(args.out, "bode.csv"), "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["model", "freq_hz", "gain_db", "phase_deg", "force_transmissibility_N_per_m", "tcp_speed_max"])
        for label in args.labels:
            for f, h, tr, v in zip(freqs, res[label]["H"], res[label]["T"], res[label]["vmax"]):
                writer.writerow([label, f"{f:.4g}", f"{20 * np.log10(abs(h)):.3f}", f"{np.degrees(np.angle(h)):.2f}",
                                 f"{abs(tr):.3f}", f"{v:.3f}"])

    lines = [f"# 실험적 주파수 응답 ({args.axis}축 가진, 진폭 {args.amp * 1000:.0f}mm, 중심 {args.center})", "",
             "비선형 폐루프의 기술 함수(describing function): 이 진폭에서의 응답이며 엄밀한 LTI 보드 선도가 아님.",
             f"약 {f_slew:.2f}Hz 이상은 TCP 속도 한계로 포화되는 구간.", "",
             "| 모델 | 대역폭 (−3dB) | 공진 피크 (최대 이득) | 피크 주파수 | 0.5Hz 위상 | 0.5Hz 반력 전달률 | 1Hz 반력 전달률 |",
             "|---|---|---|---|---|---|---|"]
    for label in args.labels:
        g = 20 * np.log10(np.abs(res[label]["H"]))
        ph = np.degrees(np.unwrap(np.angle(res[label]["H"])))
        tr = np.abs(res[label]["T"])
        lines.append(f"| {label} | {bandwidth(freqs, g):.2f} Hz | {g.max():+.2f} dB | {freqs[np.argmax(g)]:.2f} Hz | "
                     f"{np.interp(np.log10(0.5), np.log10(freqs), ph):.0f}° | "
                     f"{np.interp(np.log10(0.5), np.log10(freqs), tr):.0f} N/m | {np.interp(0.0, np.log10(freqs), tr):.0f} N/m |")
    with open(os.path.join(args.out, "bode.md"), "w") as fmd:
        fmd.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    colors = dict(zip(args.labels, ["#1f77b4", "#d62728"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for label in args.labels:
        h = res[label]["H"]
        axes[0].semilogx(freqs, 20 * np.log10(np.abs(h)), "o-", color=colors[label], label=label)
        axes[1].semilogx(freqs, np.degrees(np.unwrap(np.angle(h))), "o-", color=colors[label])
        axes[2].loglog(freqs, np.abs(res[label]["T"]), "o-", color=colors[label])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axhline(-3, color="gray", ls=":", lw=1)
    axes[0].set_ylabel("추종 이득 |X_tcp/X_target| (dB)")
    axes[1].set_ylabel("위상 (deg)")
    axes[2].set_ylabel("반력 전달률 |F_base/X_target| (N/m)")
    axes[2].set_xlabel("가진 주파수 (Hz)")
    for ax in axes:
        ax.axvspan(f_slew, args.fmax * 1.1, color="gray", alpha=0.12)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(title=f"음영: 속도 포화 구간 (>{f_slew:.1f}Hz)")
    fig.suptitle(f"실험적 주파수 응답 ({args.axis}축, 진폭 {args.amp * 1000:.0f}mm) — 비선형 시스템의 기술 함수")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bode.png"), dpi=150)
    plt.close(fig)
