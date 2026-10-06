import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from Doosan_E0509_train import E0509Env
from Doosan_E0509_compare import METRICS, PRIMARY, run_episode, compute_metrics

# ==============================================================================
# 시드(학습 실행) 단위 비교: 조건마다 여러 시드 모델을 같은 목표 집합에서 평가
#   - 분석 단위 = 학습 실행 1회. 실행마다 (성공한 목표들의) 지표 평균 1개 → 조건별 실행 간 평균±표준편차
#   - 시드 3개 vs 3개로는 통계 검정력이 낮아 p값 대신 효과 크기와 일관성(모든 A 실행이 모든 B 실행보다 나은가)을 보고
# ==============================================================================


def evaluate(path, env, seeds, hold_steps):
    model = SAC.load(path, device="cpu")
    policy = lambda obs: model.predict(obs, deterministic=True)[0]
    episodes = []
    for s in seeds:
        ep = run_episode(env, policy, s, hold_steps)
        ep["metrics"] = compute_metrics(env, ep)
        del ep["trace"]  # 메모리 절약
        episodes.append(ep)
    return episodes


def run_summary(episodes):
    ok = [ep for ep in episodes if ep["success"]]
    out = {"success_rate": len(ok) / len(episodes) * 100,
           "unsettled_5mm": np.mean([np.isnan(ep["metrics"]["settling_5mm"]) for ep in ok]) * 100}
    for key, *_ in METRICS:
        out[key] = np.nanmean([ep["metrics"][key] for ep in ok])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0509 시드별 A/B 비교")
    parser.add_argument("--a", nargs="+", required=True, help="A(안정성 페널티) 모델들")
    parser.add_argument("--b", nargs="+", required=True, help="B(기준) 모델들")
    parser.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=10000)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--out", default="compare_seeds_results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = E0509Env(record=True)
    hold_steps = int(round(args.hold / env.control_dt))
    targets = list(range(args.seed0, args.seed0 + args.n))

    runs = {}
    for label, paths in zip(args.labels, [args.a, args.b]):
        runs[label] = []
        for path in paths:
            summary = run_summary(evaluate(path, env, targets, hold_steps))
            summary["path"] = path
            runs[label].append(summary)
            print(f"{label} {path}: 성공 {summary['success_rate']:.0f}%")

    extra = [("success_rate", "성공률", "%", "과제 성능", "high"), ("unsettled_5mm", "±5mm 미정착 비율", "%", "스텝 응답", "low")]
    rows = [(k, n, u, g, "low") for k, n, u, g in METRICS] + extra
    la, lb = args.labels
    lines = [
        "# 시드별 A/B 비교", "",
        f"- A = {la}: " + ", ".join(f"`{r['path']}`" for r in runs[la]),
        f"- B = {lb}: " + ", ".join(f"`{r['path']}`" for r in runs[lb]),
        f"- 목표 {args.n}개 (seed {args.seed0}~), 모든 모델 동일. 실행마다 성공한 목표의 평균 → 실행 간 평균±표준편차",
        "- 변화율 = (A 실행 평균 − B 실행 평균) / B 실행 평균. 일관성 = A의 가장 나쁜 실행이 B의 가장 좋은 실행보다 나은가",
        "", "| 지표 | 단위 | A (실행별) | B (실행별) | A 평균±표준편차 | B 평균±표준편차 | 변화율 | 일관성 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, name, unit, group, better in rows:
        va = np.array([r[key] for r in runs[la]])
        vb = np.array([r[key] for r in runs[lb]])
        consistent = (va.max() < vb.min()) if better == "low" else (va.min() > vb.max())
        star = "**" if key in [p[0] for p in PRIMARY] else ""
        lines.append(f"| {star}{name}{star} | {unit} | {', '.join(f'{v:.3g}' for v in va)} | {', '.join(f'{v:.3g}' for v in vb)} | "
                     f"{va.mean():.4g} ± {va.std(ddof=1) if len(va) > 1 else 0:.3g} | {vb.mean():.4g} ± {vb.std(ddof=1) if len(vb) > 1 else 0:.3g} | "
                     f"{(va.mean() - vb.mean()) / vb.mean() * 100:+.1f}% | {'모두 A 우세' if consistent else '겹침'} |")
    with open(os.path.join(args.out, "seeds_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    show = PRIMARY + [("unsettled_5mm", "±5mm 미정착 비율", "%", "")]
    fig, axes = plt.subplots(1, len(show), figsize=(4.5 * len(show), 4.5))
    for ax, (key, name, unit, _) in zip(axes, show):
        for x, (label, color) in enumerate(zip(args.labels, ["#1f77b4", "#d62728"])):
            v = [r[key] for r in runs[label]]
            ax.scatter(np.full(len(v), x) + np.linspace(-0.08, 0.08, len(v)), v, s=60, color=color, zorder=3)
            ax.hlines(np.mean(v), x - 0.25, x + 0.25, color=color, lw=2)
        ax.set_xticks([0, 1], args.labels)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{name} ({unit})")
        ax.grid(alpha=0.3)
    fig.suptitle("학습 실행(시드)별 결과 — 점: 실행 1회, 가로선: 평균")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "seeds_primary.png"), dpi=150)
    plt.close(fig)
