import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pybullet as p
from stable_baselines3 import SAC

from Doosan_E0509_train import E0509Env

# ==============================================================================
# 안정성 페널티 유무(A/B) 모델의 동역학 비교 평가
#   - 동일한 무작위 목표/초기자세(고정 seed)에서 두 모델을 실행하고 240Hz 원시 데이터로 역학량 계산
#   - 힘/모멘트/ZMP/가속도/jerk/토크는 제어 주기(0.05s) 평균으로 계산: PyBullet 위치 제어는 20Hz로 목표가
#     바뀔 때마다 1서브스텝(1/240s) 동안 정격 토크급 충격을 만들며, 이는 정책과 무관한 시뮬레이터 특성이라
#     240Hz 최댓값이 정책 간 차이를 가림. 위치 오차/에너지/운동 에너지는 240Hz 그대로 사용
#   - 성공 후 hold 시간만큼 정책을 계속 실행해 도달 후 안정성도 측정
# ==============================================================================
G = 9.81
DT = 1. / 240.

# (키, 표시 이름, 단위, 그룹). 모든 지표는 낮을수록 안정적/좋음
METRICS = [
    ("base_force_peak",     "동적 베이스 반력 최댓값",        "N",      "주 지표"),
    ("zmp_shift_peak",      "동적 ZMP 이동 최댓값",           "mm",     "주 지표"),
    ("jerk_rms",            "관절 jerk RMS",                  "rad/s³", "주 지표"),
    ("base_force_rms",      "동적 베이스 반력 RMS",           "N",      "보상에 포함된 역학량"),
    ("horiz_force_peak",    "수평 반력 최댓값",               "N",      "보상에 포함된 역학량"),
    ("tipping_moment_peak", "전도 모멘트 최댓값",             "Nm",     "보상에 포함된 역학량"),
    ("base_impulse",        "베이스 반력 충격량 ∫|F|dt",      "N·s",    "보상에 포함된 역학량"),
    ("zmp_shift_rms",       "동적 ZMP 이동 RMS",              "mm",     "보상에 포함된 역학량"),
    ("tau_dyn_peak",        "동적 토크 최댓값 (정격 대비)",   "%",      "보상에 포함된 역학량"),
    ("tau_dyn_rms",         "동적 토크 RMS (정격 대비)",      "%",      "보상에 포함된 역학량"),
    ("joint_acc_peak",      "관절 가속도 최댓값",             "rad/s²", "보상에 포함된 역학량"),
    ("dtau_peak",           "토크 변화율 최댓값",             "Nm/s",   "보상에 없는 역학량"),
    ("work",                "기계적 일 ∫Σ|τ·q̇|dt",            "J",      "보상에 없는 역학량"),
    ("ke_peak",             "운동 에너지 최댓값",             "J",      "보상에 없는 역학량"),
    ("tcp_acc_peak",        "TCP 가속도 최댓값",              "m/s²",   "보상에 없는 역학량"),
    ("overshoot",           "오버슈트",                       "mm",     "보상에 없는 역학량"),
    ("settle_time",         "정밀 접근 시간 (2cm 진입→성공)", "s",      "보상에 없는 역학량"),
    ("rise_time",           "상승 시간 (10%→90%)",            "s",      "스텝 응답"),
    ("settling_2pct",       "정착 시간 (±2% 밴드)",           "s",      "스텝 응답"),
    ("settling_5mm",        "정착 시간 (±5mm 허용오차)",      "s",      "스텝 응답"),
    ("overshoot_pct",       "퍼센트 오버슈트",                "%",      "스텝 응답"),
    ("hold_dev_max",        "도달 후 TCP 흔들림 최댓값",      "mm",     "도달 후 유지"),
    ("hold_dev_rms",        "도달 후 TCP 흔들림 RMS",         "mm",     "도달 후 유지"),
    ("hold_force_rms",      "도달 후 잔류 베이스 반력 RMS",   "N",      "도달 후 유지"),
    ("reach_time",          "도달 시간",                      "s",      "과제 성능"),
    ("final_err",           "최종 오차",                      "mm",     "과제 성능"),
]
PRIMARY = [m for m in METRICS if m[3] == "주 지표"]
# 유의미한 차이 판정 기준 (결과를 보기 전에 정함)
MIN_CHANGE_PCT = 10.0
MIN_BETTER_FRAC = 0.70
MAX_P = 0.01


def run_episode(env, policy, seed, hold_steps):
    """한 목표에서 정책 실행 + 성공 시 hold_steps 동안 계속 실행. 원시 trace와 메타데이터 반환"""
    obs, _ = env.reset(seed=seed)
    target = env.target_pos.copy()
    q0 = obs[:env.n_joints].copy()
    for step in range(1, env.max_steps + 1):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            break
    motion_end = len(env.trace)  # 성공(또는 종료) 시점까지의 샘플 수
    if terminated:
        for _ in range(hold_steps):
            obs, *_ = env.step(policy(obs))
    trace = {k: np.array([s[k] for s in env.trace]) for k in env.trace[0]}
    return {"seed": seed, "target": target, "q0": q0, "success": bool(terminated), "steps": step,
            "final_err": info["distance"], "motion_end": motion_end, "trace": trace}


def control_samples(env, trace, key):
    """제어 주기 경계 샘플 (trace[0], trace[12], ...): 길이 = 제어 스텝 수 + 1"""
    return trace[key][::env.n_substeps]


def base_wrench(env, trace):
    """뉴턴-오일러: 제어 주기 평균 동적 반력/모멘트(바닥 원점 기준)와 ZMP. 제어 스텝마다 1개"""
    dt = env.control_dt
    F_dyn = np.diff(control_samples(env, trace, "P"), axis=0) / dt
    M_dyn_O = np.diff(control_samples(env, trace, "L"), axis=0) / dt
    M_dyn = M_dyn_O + np.cross(env.base_origin, F_dyn)  # joint_1 원점 O → 바닥 원점 Q=(0,0,0)

    # 정적 성분: 팔 + base_link 전체 무게가 시스템 무게중심에 작용
    m_arm = env.link_masses.sum()
    m_base = p.getDynamicsInfo(env.robot, -1)[0]
    c_base = np.array(p.getBasePositionAndOrientation(env.robot)[0])
    c_sys = (m_arm * control_samples(env, trace, "com")[1:] + m_base * c_base) / (m_arm + m_base)
    F_static = np.array([0.0, 0.0, (m_arm + m_base) * G])

    F_tot = F_dyn + F_static
    M_tot = M_dyn + np.cross(c_sys, F_static)
    zmp = np.stack([-M_tot[:, 1] / F_tot[:, 2], M_tot[:, 0] / F_tot[:, 2]], axis=1)
    zmp_shift = zmp - c_sys[:, :2]  # 정지 시 ZMP = 무게중심 수평 투영 → 동역학에 의한 이동만 남김
    return F_dyn, M_dyn, zmp_shift


def compute_metrics(env, ep):
    tr = ep["trace"]
    n = ep["motion_end"]
    k = env.n_substeps
    steps = (n - 1) // k  # 동작 구간 제어 스텝 수
    dt = env.control_dt
    F_dyn, M_dyn, zmp_shift = base_wrench(env, tr)
    mo = slice(0, steps)  # 동작 구간 (제어 스텝 단위)
    ho = slice(steps, None)  # 도달 후 유지 구간

    F = np.linalg.norm(F_dyn[mo], axis=1)
    zs = np.linalg.norm(zmp_shift[mo], axis=1) * 1000
    # 제어 주기별 평균 토크 (reset 직후 샘플 tr[0]의 토크는 이전 에피소드 값이므로 제외)
    tau_ctrl = tr["tau"][1:n].reshape(steps, k, -1).mean(axis=1)
    tau_dyn = (tau_ctrl - tr["tau_g"][1:n].reshape(steps, k, -1).mean(axis=1)) / env.max_forces * 100
    acc = np.diff(control_samples(env, tr, "dq")[:steps + 1], axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    tcp = tr["tcp_pos"]
    dist = np.linalg.norm(tcp[:n] - ep["target"], axis=1)
    approach = (ep["target"] - tcp[0]) / np.linalg.norm(ep["target"] - tcp[0])

    m = {
        "base_force_peak": F.max(),
        "base_force_rms": np.sqrt(np.mean(F ** 2)),
        "horiz_force_peak": np.linalg.norm(F_dyn[mo, :2], axis=1).max(),
        "tipping_moment_peak": np.linalg.norm(M_dyn[mo, :2], axis=1).max(),
        "base_impulse": F.sum() * dt,
        "zmp_shift_peak": zs.max(),
        "zmp_shift_rms": np.sqrt(np.mean(zs ** 2)),
        "tau_dyn_peak": np.abs(tau_dyn).max(),
        "tau_dyn_rms": np.sqrt(np.mean(tau_dyn ** 2)),
        "joint_acc_peak": np.abs(acc).max(),
        "jerk_rms": np.sqrt(np.mean(jerk ** 2)),
        "dtau_peak": np.abs(np.diff(tau_ctrl, axis=0) / dt).max(),
        "work": np.sum(np.abs(tr["tau"][1:n] * tr["dq"][1:n])) * DT,
        "ke_peak": tr["ke"][:n].max(),
        "tcp_acc_peak": np.linalg.norm(np.diff(control_samples(env, tr, "tcp_vel")[:steps + 1], axis=0) / dt, axis=1).max(),
        "overshoot": max(0.0, ((tcp[:n] - ep["target"]) @ approach).max()) * 1000,
        "final_err": ep["final_err"] * 1000,
    }
    if ep["success"]:
        first_near = np.argmax(dist < 0.02)
        hold_dev = np.linalg.norm(tcp[n:] - tcp[n - 1], axis=1) * 1000
        m.update({
            "reach_time": ep["steps"] * env.control_dt,
            "settle_time": (n - 1 - first_near) * DT,
            "hold_dev_max": hold_dev.max() if len(hold_dev) else np.nan,
            "hold_dev_rms": np.sqrt(np.mean(hold_dev ** 2)) if len(hold_dev) else np.nan,
            "hold_force_rms": np.sqrt(np.mean(np.linalg.norm(F_dyn[ho], axis=1) ** 2)) if len(hold_dev) else np.nan,
        })
    else:
        m.update({k: np.nan for k in ["reach_time", "settle_time", "hold_dev_max", "hold_dev_rms", "hold_force_rms"]})
    m.update(step_response(ep, m["overshoot"]))
    return m


def step_response(ep, overshoot_mm):
    """목표 도달을 스텝 응답으로 보고 계산 (동작 + 도달 후 유지 구간 전체, 240Hz).
    정착 시간 = 오차가 밴드 안으로 들어와 끝까지 머무른 시작 시각. 마지막에 밴드 밖이면 미정착(nan)"""
    keys = ["rise_time", "settling_2pct", "settling_5mm", "overshoot_pct"]
    if not ep["success"]:
        return {k: np.nan for k in keys}
    d = np.linalg.norm(ep["trace"]["tcp_pos"] - ep["target"], axis=1)
    d0 = d[0]
    progress = 1 - d / d0
    out = {"rise_time": (np.argmax(progress >= 0.9) - np.argmax(progress >= 0.1)) * DT,
           "overshoot_pct": overshoot_mm / (d0 * 1000) * 100}
    for key, band in [("settling_2pct", 0.02 * d0), ("settling_5mm", 0.005)]:
        outside = np.where(d > band)[0]
        out[key] = np.nan if (len(outside) and outside[-1] == len(d) - 1) else (outside[-1] + 1 if len(outside) else 0) * DT
    return out


def summarize(results, labels):
    """두 모델이 모두 성공한 목표만 짝지어 비교: 평균±표준편차, 변화율, A가 낮은 목표 비율, Wilcoxon p"""
    a, b = (results[l] for l in labels)
    both = [i for i in range(len(a)) if a[i]["success"] and b[i]["success"]]
    rows = []
    for key, name, unit, group in METRICS:
        va = np.array([a[i]["metrics"][key] for i in both])
        vb = np.array([b[i]["metrics"][key] for i in both])
        ok = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[ok], vb[ok]
        change = (va.mean() - vb.mean()) / vb.mean() * 100 if vb.mean() != 0 else np.nan
        better = np.mean(va < vb) if len(va) else np.nan
        try:
            pval = wilcoxon(va, vb).pvalue
        except ValueError:  # 모든 차이가 0인 경우 등
            pval = np.nan
        rows.append({"key": key, "name": name, "unit": unit, "group": group, "n": len(va),
                     "a_mean": va.mean(), "a_std": va.std(), "b_mean": vb.mean(), "b_std": vb.std(),
                     "change": change, "better": better, "p": pval})
    return rows, both


def is_significant(row):
    return (row["change"] <= -MIN_CHANGE_PCT and row["better"] >= MIN_BETTER_FRAC and row["p"] < MAX_P)


def write_outputs(results, labels, rows, both, args, env):
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "per_episode.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", "target_x", "target_y", "target_z", "success"] + [k for k, *_ in METRICS])
        for label in labels:
            for ep in results[label]:
                writer.writerow([label, ep["seed"], *np.round(ep["target"], 5), ep["success"]] +
                                [f"{ep['metrics'][k]:.6g}" for k, *_ in METRICS])

    succ = {l: sum(ep["success"] for ep in results[l]) for l in labels}
    lines = [
        "# 안정성 페널티 비교 결과", "",
        f"- A = `{labels[0]}` ({args.models[0]}), B = `{labels[1]}` ({args.models[1]})",
        f"- 목표 {args.n}개 (seed {args.seed0}~{args.seed0 + args.n - 1}, 두 모델 동일), 도달 후 유지 {args.hold}초, 240Hz 기록",
        "- 힘/모멘트/ZMP/가속도/jerk/토크는 제어 주기(0.05s) 평균값 (PyBullet 위치 제어의 서브스텝 충격 제외). 에너지/위치 오차는 240Hz",
        f"- 성공: A {succ[labels[0]]}/{args.n}, B {succ[labels[1]]}/{args.n} → 둘 다 성공한 {len(both)}개 목표로 짝지어 비교",
        "- 스텝 응답: 출발→목표를 스텝 입력으로 보고 동작+유지 구간 전체로 계산. 정착 시간은 오차가 밴드 안에 들어와 끝까지 머문 시각 (유지 종료 시 밴드 밖이면 미정착으로 제외, n에 반영)",
        f"- 변화율 = (A평균 − B평균) / B평균. 음수면 A가 더 낮음(더 안정적). 'A가 낮은 비율' = 같은 목표에서 A < B인 비율",
        f"- 유의미 판정 기준(사전 정의): 변화율 ≤ −{MIN_CHANGE_PCT:.0f}% AND A가 낮은 비율 ≥ {MIN_BETTER_FRAC:.0%} AND Wilcoxon p < {MAX_P}",
        "- 주의: 학습 시드 1개씩의 비교. p값은 목표 간 편차만 반영하며 학습 실행 간 편차는 반영하지 않음", "",
    ]
    for group in dict.fromkeys(r["group"] for r in rows):
        lines += [f"## {group}", "",
                  "| 지표 | 단위 | A 평균±표준편차 | B 평균±표준편차 | 변화율 | A가 낮은 비율 | p | 판정 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["group"] == group):
            verdict = "**유의미**" if is_significant(r) else ""
            lines.append(f"| {r['name']} | {r['unit']} | {r['a_mean']:.4g} ± {r['a_std']:.3g} | {r['b_mean']:.4g} ± {r['b_std']:.3g} | "
                         f"{r['change']:+.1f}% | {r['better']:.0%} | {r['p']:.2g} | {verdict} |")
        lines.append("")
    # 정착 시간 평균은 미정착 에피소드를 제외하므로, 미정착 비율을 따로 보고 (제외가 한쪽에 유리하게 작용할 수 있음)
    lines += ["## 미정착 (유지 종료 시점에 밴드 밖)", "", "| 밴드 | " + " | ".join(labels) + " |", "|---|---|---|"]
    for key, name in [("settling_2pct", "±2%"), ("settling_5mm", "±5mm")]:
        counts = [f"{sum(np.isnan(results[l][i]['metrics'][key]) for i in both)}/{len(both)}" for l in labels]
        lines.append(f"| {name} | " + " | ".join(counts) + " |")
    lines.append("")
    primary_sig = [is_significant(r) for r in rows if r["group"] == "주 지표"]
    lines.append(f"**주 지표 유의미: {sum(primary_sig)}/{len(primary_sig)}**")
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_all(results, labels, both, args, env)


def plot_all(results, labels, both, args, env):
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    colors = {labels[0]: "#1f77b4", labels[1]: "#d62728"}

    # 1) 대표 목표(A 도달 시간이 중앙값인 목표)의 시계열
    times_a = [results[labels[0]][i]["metrics"]["reach_time"] for i in both]
    rep = both[int(np.argsort(times_a)[len(times_a) // 2])]
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    for label in labels:
        ep = results[label][rep]
        tr = ep["trace"]
        t = np.arange(len(tr["P"])) * DT
        F_dyn, _, zmp_shift = base_wrench(env, tr)
        tc = np.arange(1, len(F_dyn) + 1) * env.control_dt
        axes[0].plot(t, np.linalg.norm(tr["tcp_pos"] - ep["target"], axis=1) * 1000, color=colors[label], label=label)
        axes[1].plot(t, np.linalg.norm(tr["tcp_vel"], axis=1), color=colors[label])
        axes[2].step(tc, np.linalg.norm(F_dyn, axis=1), where="post", color=colors[label], lw=0.9)
        axes[3].step(tc, np.linalg.norm(zmp_shift, axis=1) * 1000, where="post", color=colors[label], lw=0.9)
        for ax in axes:
            ax.axvline((ep["motion_end"] - 1) * DT, color=colors[label], ls="--", lw=0.8)
    axes[0].set_yscale("log")
    for ax, yl in zip(axes, ["목표까지 거리 (mm)", "TCP 속도 (m/s)", "동적 베이스 반력 (N)\n(제어 주기 평균)", "동적 ZMP 이동 (mm)\n(제어 주기 평균)"]):
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("시간 (s)  — 점선: 성공 시점, 이후는 도달 후 유지 구간")
    fig.suptitle(f"대표 목표 (seed {results[labels[0]][rep]['seed']}) 시계열 비교")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "timeseries.png"), dpi=150)
    plt.close(fig)

    # 2) 같은 목표의 동적 ZMP 이동 궤적 (위에서 본 그림)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label in labels:
        ep = results[label][rep]
        _, _, zmp_shift = base_wrench(env, ep["trace"])
        zs = zmp_shift[:(ep["motion_end"] - 1) // env.n_substeps] * 1000
        ax.plot(zs[:, 0], zs[:, 1], color=colors[label], lw=0.8, label=label)
    ax.plot(0, 0, "k+", ms=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("동적 ZMP 이동 궤적 (원점 = 정적 ZMP)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "zmp_path.png"), dpi=150)
    plt.close(fig)

    # 3) 속도 대비 안정성 산점도 (속도 차이 보정)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, name, unit) in zip(axes, [("base_force_peak", "동적 베이스 반력 최댓값", "N"),
                                            ("zmp_shift_peak", "동적 ZMP 이동 최댓값", "mm"),
                                            ("base_impulse", "베이스 반력 충격량", "N·s")]):
        for label in labels:
            x = [results[label][i]["metrics"]["reach_time"] for i in both]
            y = [results[label][i]["metrics"][key] for i in both]
            ax.scatter(x, y, s=14, alpha=0.7, color=colors[label], label=label)
        ax.set_xlabel("도달 시간 (s)")
        ax.set_ylabel(f"{name} ({unit})")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("속도 대비 안정성 (왼쪽 아래일수록 빠르고 안정적)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "speed_vs_stability.png"), dpi=150)
    plt.close(fig)

    # 4) 스텝 응답: 정규화 진행률(1 - d/d0)과 목표 근처 오차, 짝지은 목표 전체의 중앙값과 25~75% 구간
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    t_max = max(len(results[l][i]["trace"]["tcp_pos"]) for l in labels for i in both)
    t = np.arange(t_max) * DT
    for label in labels:
        prog, err = [], []
        for i in both:
            ep = results[label][i]
            d = np.linalg.norm(ep["trace"]["tcp_pos"] - ep["target"], axis=1)
            d = np.pad(d, (0, t_max - len(d)), mode="edge")  # 유지 종료 후는 마지막 값 유지
            prog.append(1 - d / d[0])
            err.append(d * 1000)
        for ax, data in zip(axes, [np.array(prog), np.array(err)]):
            q25, q50, q75 = np.percentile(data, [25, 50, 75], axis=0)
            ax.plot(t, q50, color=colors[label], label=f"{label} (중앙값)")
            ax.fill_between(t, q25, q75, color=colors[label], alpha=0.2)
        for key, ls in [("settling_2pct", "--"), ("settling_5mm", ":")]:
            v = np.nanmedian([results[label][i]["metrics"][key] for i in both])
            axes[1].axvline(v, color=colors[label], ls=ls, lw=1)
    axes[0].axhline(1.0, color="k", lw=0.8)
    axes[0].set_ylabel("진행률 1 − d/d0")
    axes[0].legend()
    axes[1].axhline(5, color="gray", ls=":", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.3, 500)
    axes[1].set_ylabel("목표까지 거리 (mm)")
    axes[1].set_xlabel("시간 (s)  — 세로선: 정착 시간 중앙값 (점선 ±2%, 점 ±5mm)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("스텝 응답 비교 (중앙값, 음영: 25~75%)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "step_response.png"), dpi=150)
    plt.close(fig)

    # 5) 주 지표 박스플롯
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(5 * len(PRIMARY), 4.5))
    for ax, (key, name, unit, _) in zip(axes, PRIMARY):
        data = [[results[label][i]["metrics"][key] for i in both] for label in labels]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks([1, 2], labels)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.5)
        ax.set_title(f"{name} ({unit})")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "primary_boxplot.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0509 안정성 페널티 유무 모델 동역학 비교")
    parser.add_argument("--models", nargs=2, default=["e0509_2f85_sac.zip", "e0509_2f85_sac_nostab.zip"])
    parser.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    parser.add_argument("--n", type=int, default=100, help="목표 개수")
    parser.add_argument("--seed0", type=int, default=10000, help="목표 생성 시작 seed (학습/이전 테스트와 겹치지 않게)")
    parser.add_argument("--hold", type=float, default=2.0, help="성공 후 유지 시간 (s)")
    parser.add_argument("--out", default="compare_results")
    args = parser.parse_args()

    # 한 프로세스에 환경 1개만 사용 (PyBullet 기본 연결 공유 문제 방지)
    env = E0509Env(record=True)
    hold_steps = int(round(args.hold / env.control_dt))
    results = {}
    for label, path in zip(args.labels, args.models):
        model = SAC.load(path, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
        results[label] = []
        for i in range(args.n):
            ep = run_episode(env, policy, args.seed0 + i, hold_steps)
            ep["metrics"] = compute_metrics(env, ep)
            results[label].append(ep)
        print(f"{label}: 성공 {sum(ep['success'] for ep in results[label])}/{args.n}")

    # 두 모델이 같은 목표/초기자세에서 시작했는지 확인
    for ea, eb in zip(*(results[l] for l in args.labels)):
        assert np.allclose(ea["target"], eb["target"]) and np.allclose(ea["q0"], eb["q0"]), "목표/초기자세 불일치"

    rows, both = summarize(results, args.labels)
    write_outputs(results, args.labels, rows, both, args, env)
