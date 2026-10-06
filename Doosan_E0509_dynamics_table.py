import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

from Doosan_E0509_train import E0509Env
from Doosan_E0509_compare import run_episode, base_wrench

# ==============================================================================
# 같은 시행(목표/초기자세)에서 A/B의 역학량을 성분별로 비교하는 표와 시계열
#   - 베이스 프레임(바닥 원점) 기준 동적 반력 Fx/Fy/Fz, 동적 모멘트 Mx/My/Mz (중력 성분 제외)
#   - 관절 J1~J6 토크: 실제 모터 토크 τ, 동적 토크 τ−τ_g (중력 보상분 제외)
#   - 모든 역학량은 제어 주기(0.05s) 평균 (PyBullet 위치 제어의 서브스텝 충격 제외)
#   - 요약표: 시드 쌍(A_k vs B_k) × 목표 n개, 둘 다 성공한 목표만. 시행 상세: 지정한 시드 쌍의 대표 시행
# ==============================================================================
JOINTS = [f"J{i}" for i in range(1, 7)]
COMPONENTS = ["Fx", "Fy", "Fz", "|F|", "Mx", "My", "Mz", "|M|"]


def episode_series(env, ep):
    """제어 주기 시계열 (동작 + 도달 후 유지): t, 거리, 속도, 베이스 힘/모멘트, 관절 토크"""
    tr = ep["trace"]
    k = env.n_substeps
    F, M, _ = base_wrench(env, tr)
    steps = len(F)
    idx = np.arange(1, steps + 1) * k
    tau = tr["tau"][1:].reshape(steps, k, -1).mean(axis=1)
    tau_g = tr["tau_g"][1:].reshape(steps, k, -1).mean(axis=1)
    return {
        "t": np.arange(1, steps + 1) * env.control_dt,
        "motion_steps": (ep["motion_end"] - 1) // k,
        "dist": np.linalg.norm(tr["tcp_pos"][idx] - ep["target"], axis=1) * 1000,
        "v": np.linalg.norm(tr["tcp_vel"][idx], axis=1),
        "F": F, "M": M, "tau": tau, "tau_dyn": tau - tau_g,
    }


def episode_stats(s, max_forces):
    """동작 구간(출발→성공)의 최댓값/RMS"""
    m = slice(0, s["motion_steps"])
    F, M, tau, td = s["F"][m], s["M"][m], s["tau"][m], s["tau_dyn"][m]
    comp = {"Fx": F[:, 0], "Fy": F[:, 1], "Fz": F[:, 2], "|F|": np.linalg.norm(F, axis=1),
            "Mx": M[:, 0], "My": M[:, 1], "Mz": M[:, 2], "|M|": np.linalg.norm(M, axis=1)}
    out = {"reach_time": s["motion_steps"] * 0.05, "final_dist": s["dist"][s["motion_steps"] - 1], "v_peak": s["v"][m].max()}
    for name, x in comp.items():
        out[f"{name} peak"] = np.abs(x).max()
        out[f"{name} rms"] = np.sqrt(np.mean(x ** 2))
    for j, name in enumerate(JOINTS):
        out[f"{name} τ peak"] = np.abs(tau[:, j]).max()
        out[f"{name} τ_dyn peak"] = np.abs(td[:, j]).max()
        out[f"{name} τ_dyn rms"] = np.sqrt(np.mean(td[:, j] ** 2))
        out[f"{name} τ_dyn peak %"] = np.abs(td[:, j]).max() / max_forces[j] * 100
    return out


# (키, 표시 이름, 단위)
ROWS = [("reach_time", "도달 시간", "s"), ("final_dist", "최종 거리", "mm"), ("v_peak", "TCP 최대 속도", "m/s")]
ROWS += [(f"{c} {k}", f"베이스 {'힘' if 'F' in c else '모멘트'} {c} {'최댓값' if k == 'peak' else 'RMS'}",
          "N" if "F" in c else "Nm") for c in COMPONENTS for k in ["peak", "rms"]]
ROWS += [(f"{j} {k}", f"{j} {name}", unit) for j in JOINTS for k, name, unit in
         [("τ peak", "실제 토크 최댓값", "Nm"), ("τ_dyn peak", "동적 토크 최댓값", "Nm"),
          ("τ_dyn rms", "동적 토크 RMS", "Nm"), ("τ_dyn peak %", "동적 토크 최댓값 (정격 대비)", "%")]]


def fmt(v):
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def write_trial(out_dir, seed, series, labels, env):
    """한 시행의 A/B 비교: 표(md), 시계열(csv), 그래프(png)"""
    stats = {l: episode_stats(series[l], env.max_forces) for l in labels}
    la, lb = labels
    lines = [f"# 시행 seed {seed} — 같은 목표/초기자세에서 A vs B", "",
             "동작 구간(출발→성공) 기준. 베이스 힘/모멘트는 바닥 원점 기준 동적 성분(중력 제외), 제어 주기(0.05s) 평균", "",
             f"| 항목 | 단위 | {la} | {lb} | 변화 (A−B)/B |", "|---|---|---|---|---|"]
    for key, name, unit in ROWS:
        a, b = stats[la][key], stats[lb][key]
        lines.append(f"| {name} | {unit} | {fmt(a)} | {fmt(b)} | {(a - b) / b * 100:+.1f}% |")
    with open(os.path.join(out_dir, f"trial_{seed}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(out_dir, f"trial_{seed}_timeseries.csv"), "w", newline="") as f:
        cols = ["dist_mm", "v_mps", "Fx", "Fy", "Fz", "Mx", "My", "Mz"] + [f"tau_{j}" for j in JOINTS] + [f"tau_dyn_{j}" for j in JOINTS]
        writer = csv.writer(f)
        writer.writerow(["t_s"] + [f"{l}_{c}" for l in labels for c in ["phase"] + cols])
        n = max(len(series[l]["t"]) for l in labels)
        for i in range(n):
            row = [f"{(i + 1) * env.control_dt:.2f}"]
            for l in labels:
                s = series[l]
                if i < len(s["t"]):
                    vals = [s["dist"][i], s["v"][i], *s["F"][i], *s["M"][i], *s["tau"][i], *s["tau_dyn"][i]]
                    row += ["motion" if i < s["motion_steps"] else "hold"] + [f"{v:.4g}" for v in vals]
                else:
                    row += [""] * (1 + len(cols))
            writer.writerow(row)

    colors = dict(zip(labels, ["#1f77b4", "#d62728"]))
    fig, axes = plt.subplots(7, 2, figsize=(13, 20), sharex=True)
    panels = [("dist", None, "목표까지 거리 (mm)"), ("v", None, "TCP 속도 (m/s)"),
              ("F", 0, "베이스 Fx (N)"), ("M", 0, "베이스 Mx (Nm)"), ("F", 1, "베이스 Fy (N)"), ("M", 1, "베이스 My (Nm)"),
              ("F", 2, "베이스 Fz (N)"), ("M", 2, "베이스 Mz (Nm)")]
    panels += [("tau_dyn", j, f"{JOINTS[j]} 동적 토크 (Nm)") for j in range(6)]
    for ax, (key, j, title) in zip(axes.flat, panels):
        for l in labels:
            s = series[l]
            y = s[key] if j is None else s[key][:, j]
            ax.plot(s["t"], y, color=colors[l], lw=1, label=l)
            ax.axvline(s["motion_steps"] * env.control_dt, color=colors[l], ls="--", lw=0.8)
        if key == "dist":
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("시간 (s) — 점선: 성공 시점")
    fig.suptitle(f"시행 seed {seed}: 같은 목표에서 A vs B 역학량 (제어 주기 평균, 베이스는 바닥 원점 기준 동적 성분)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(out_dir, f"trial_{seed}.png"), dpi=120)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="같은 시행에서 A/B 역학량 성분별 비교")
    parser.add_argument("--a", nargs="+", default=["e0509_2f85_sac_v2_s1.zip", "e0509_2f85_sac_v2_s2.zip", "e0509_2f85_sac_v2_s3.zip"])
    parser.add_argument("--b", nargs="+", default=["e0509_2f85_sac_nostab.zip", "e0509_2f85_sac_nostab_s2.zip", "e0509_2f85_sac_nostab_s3.zip"])
    parser.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=10000)
    parser.add_argument("--detail-pair", type=int, default=1, help="시행 상세에 쓸 시드 쌍 인덱스 (0부터, 기본 1 = 시드 2 쌍)")
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--out", default="dynamics_tables")
    args = parser.parse_args()
    assert len(args.a) == len(args.b), "A/B 모델 수가 같아야 시드 쌍을 만들 수 있음"

    os.makedirs(args.out, exist_ok=True)
    env = E0509Env(record=True)
    hold_steps = int(round(args.hold / env.control_dt))
    targets = list(range(args.seed0, args.seed0 + args.n))
    la, lb = args.labels

    # 시행 상세 대상: 결과와 무관하게 목표까지 초기 거리로 선정 (최단 / 중앙값 / 최장)
    d0 = []
    for s in targets:
        env.reset(seed=s)
        d0.append(np.linalg.norm(env.target_pos - env._ee_pos()))
    order = np.argsort(d0)
    detail = [targets[order[0]], targets[order[len(order) // 2]], targets[order[-1]]]

    pair_means = []  # 시드 쌍마다 {라벨: {지표: 평균}}
    for p_idx, (pa, pb) in enumerate(zip(args.a, args.b)):
        series = {}
        for label, path in [(la, pa), (lb, pb)]:
            model = SAC.load(path, device="cpu")
            policy = lambda obs: model.predict(obs, deterministic=True)[0]
            series[label] = {}
            for s in targets:
                ep = run_episode(env, policy, s, hold_steps)
                series[label][s] = episode_series(env, ep) if ep["success"] else None
        both = [s for s in targets if series[la][s] is not None and series[lb][s] is not None]
        means = {}
        for label in [la, lb]:
            st = [episode_stats(series[label][s], env.max_forces) for s in both]
            means[label] = {key: np.mean([x[key] for x in st]) for key, *_ in ROWS}
        pair_means.append(means)
        print(f"시드 쌍 {p_idx + 1} ({pa} vs {pb}): 둘 다 성공 {len(both)}/{args.n}")
        if p_idx == args.detail_pair:
            for s in detail:
                if s in both:
                    write_trial(args.out, s, {l: series[l][s] for l in [la, lb]}, [la, lb], env)

    lines = [
        "# 역학량 성분별 비교 요약 (같은 시행끼리 짝지어 평균)", "",
        f"- 시드 쌍: " + ", ".join(f"`{a}` vs `{b}`" for a, b in zip(args.a, args.b)),
        f"- 목표 {args.n}개 (seed {args.seed0}~), 시드 쌍마다 A/B 둘 다 성공한 목표만 사용 → 에피소드별 동작 구간 최댓값/RMS의 평균",
        "- 베이스 힘/모멘트: 바닥 원점(로봇 베이스 프레임) 기준 동적 성분(중력 제외). Mx/My = 전도 모멘트, Mz = 수직축 반작용 토크",
        "- 관절 토크: 실제 τ는 중력 보상 포함, 동적 τ_dyn = τ − τ_g. 모두 제어 주기(0.05s) 평균",
        "- 일관성: 3개 시드 쌍 모두에서 A < B 이면 '모두 A 낮음'",
        f"- 시행 상세(`trial_<seed>.md/.csv/.png`): 시드 쌍 {args.detail_pair + 1}, 초기 거리 최단/중앙/최장 목표 {detail}", "",
        f"| 항목 | 단위 | {la} (쌍별) | {lb} (쌍별) | {la} 평균 | {lb} 평균 | 변화 | 일관성 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, name, unit in ROWS:
        va = np.array([m[la][key] for m in pair_means])
        vb = np.array([m[lb][key] for m in pair_means])
        verdict = "모두 A 낮음" if np.all(va < vb) else ("모두 A 높음" if np.all(va > vb) else "섞임")
        lines.append(f"| {name} | {unit} | {' / '.join(fmt(v) for v in va)} | {' / '.join(fmt(v) for v in vb)} | "
                     f"{fmt(va.mean())} | {fmt(vb.mean())} | {(va.mean() - vb.mean()) / vb.mean() * 100:+.1f}% | {verdict} |")
    with open(os.path.join(args.out, "summary_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
