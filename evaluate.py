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

from e0509_env import E0509Env

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# 학습된 모델의 동역학 평가 (서브커맨드)
#   pair     : 한 쌍 상세 비교 (역학 지표, 스텝 응답, 사전 정의 기준 판정, 그래프)
#   seeds    : 시드(학습 실행) 단위 A/B 비교
#   dynamics : 같은 시행에서 베이스 힘·모멘트 / 관절 토크 성분별 표와 시계열
#   bode     : 실험적 주파수 응답 (보드 선도)
#   공통: 한 프로세스에 환경 1개만 사용 (PyBullet 기본 연결을 공유하므로 환경 2개면 로봇끼리 간섭)
# ==============================================================================

# ==============================================================================
# [pair] 한 쌍 상세 비교 (역학 지표, 스텝 응답, 사전 정의 기준 판정, 그래프)
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


def cmd_pair(args):

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

# ==============================================================================
# [seeds] 시드(학습 실행) 단위 A/B 비교
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


def cmd_seeds(args):

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

# ==============================================================================
# [dynamics] 같은 시행에서 베이스 힘·모멘트 / 관절 토크 성분별 표와 시계열
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


def cmd_dynamics(args):
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

# ==============================================================================
# [bode] 실험적 주파수 응답 (보드 선도)
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


def cmd_bode(args):

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0509 모델 동역학 평가")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("pair", help="한 쌍 상세 비교 (역학 지표, 스텝 응답, 사전 정의 기준 판정, 그래프)")
    sp.add_argument("--models", nargs=2, default=["models/stability_v1_s1.zip", "models/baseline_s1.zip"])
    sp.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    sp.add_argument("--n", type=int, default=100, help="목표 개수")
    sp.add_argument("--seed0", type=int, default=10000, help="목표 생성 시작 seed (학습/이전 테스트와 겹치지 않게)")
    sp.add_argument("--hold", type=float, default=2.0, help="성공 후 유지 시간 (s)")
    sp.add_argument("--out", default="results/pair_seed1")
    sp.set_defaults(func=cmd_pair)

    sp = sub.add_parser("seeds", help="시드(학습 실행) 단위 A/B 비교")
    sp.add_argument("--a", nargs="+", default=["models/stability_s1.zip", "models/stability_s2.zip", "models/stability_s3.zip"], help="A(안정성 페널티) 모델들")
    sp.add_argument("--b", nargs="+", default=["models/baseline_s1.zip", "models/baseline_s2.zip", "models/baseline_s3.zip"], help="B(기준) 모델들")
    sp.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--seed0", type=int, default=10000)
    sp.add_argument("--hold", type=float, default=2.0)
    sp.add_argument("--out", default="results/seeds")
    sp.set_defaults(func=cmd_seeds)

    sp = sub.add_parser("dynamics", help="같은 시행에서 베이스 힘·모멘트 / 관절 토크 성분별 표와 시계열")
    sp.add_argument("--a", nargs="+", default=["models/stability_s1.zip", "models/stability_s2.zip", "models/stability_s3.zip"])
    sp.add_argument("--b", nargs="+", default=["models/baseline_s1.zip", "models/baseline_s2.zip", "models/baseline_s3.zip"])
    sp.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--seed0", type=int, default=10000)
    sp.add_argument("--detail-pair", type=int, default=1, help="시행 상세에 쓸 시드 쌍 인덱스 (0부터, 기본 1 = 시드 2 쌍)")
    sp.add_argument("--hold", type=float, default=2.0)
    sp.add_argument("--out", default="results/dynamics")
    sp.set_defaults(func=cmd_dynamics)

    sp = sub.add_parser("bode", help="실험적 주파수 응답 (보드 선도)")
    sp.add_argument("--models", nargs=2, default=["models/stability_v1_s1.zip", "models/baseline_s1.zip"])
    sp.add_argument("--labels", nargs=2, default=["A_안정성", "B_기준"])
    sp.add_argument("--center", nargs=3, type=float, default=[0.45, 0.0, 0.35], help="가진 중심 목표 (m)")
    sp.add_argument("--axis", choices=AXES, default="x")
    sp.add_argument("--amp", type=float, default=0.02, help="가진 진폭 (m)")
    sp.add_argument("--fmin", type=float, default=0.05)
    sp.add_argument("--fmax", type=float, default=4.0)
    sp.add_argument("--nfreq", type=int, default=14)
    sp.add_argument("--repeats", type=int, default=3, help="주파수마다 초기 자세를 바꿔 반복 후 복소 평균")
    sp.add_argument("--out", default="results/pair_seed1")
    sp.set_defaults(func=cmd_bode)

    args = parser.parse_args()
    args.func(args)
