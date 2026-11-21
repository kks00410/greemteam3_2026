import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 한글 폰트 등록
for path in [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]:
    try:
        fm.fontManager.addfont(path)
    except Exception:
        pass
plt.rcParams["font.family"] = "Nanum Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 데이터 로드 ─────────────────────────────────────────────────────────
g = pd.read_csv("/home/user/uploads/3_________________1_.csv", dtype=str)
g.columns = [str(c).strip() for c in g.columns]
for c in g.columns:
    if c != "개체번호" and not pd.api.types.is_numeric_dtype(g[c]):
        num = pd.to_numeric(
            g[c].astype(str).str.replace(",", "").
            str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
        )
        if num.notna().mean() >= 0.5:
            g[c] = num
g = g.rename(columns={
    "지하부질량(g)": "지하부 질량",
    "잎 수(장)": "잎 수",
})

below = g["지하부 질량"].dropna()
leaf  = g["잎 수"].dropna()
common = below.index.intersection(leaf.index)
bw = below.loc[common]
lf = leaf.loc[common]

mean_b = bw.mean()
mean_l = lf.mean()
t_per  = ((bw - mean_b).abs() + (lf - mean_l).abs()) / 2.0
t_mean = t_per.mean()

n_total  = len(t_per)
n_below  = int((t_per <= t_mean).sum())
n_above  = int((t_per >  t_mean).sum())
pct_below = n_below / n_total * 100
pct_above = n_above / n_total * 100

print(f"t 평균 = {t_mean:.4f}  |  총 {n_total}개체")
print(f"이하 {n_below}개 ({pct_below:.1f}%) / 초과 {n_above}개 ({pct_above:.1f}%)")

# ── 그림: 왼쪽 원 그래프 + 오른쪽 t 분포 산점도 ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
fig.patch.set_facecolor("#FFF0F5")

# --- 왼쪽: 도넛 원 그래프 ---
ax1 = axes[0]
ax1.set_facecolor("#FFF0F5")
wedges, texts, autotexts = ax1.pie(
    [n_below, n_above],
    labels=[
        f"t 평균 이하\n{n_below}개체 · {pct_below:.1f}%",
        f"t 평균 초과\n{n_above}개체 · {pct_above:.1f}%",
    ],
    colors=["#FFB6C1", "#FF69B4"],
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5),
    autopct=lambda p: f"{p:.1f}%",
    textprops=dict(fontsize=12, color="#333333"),
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(13)
ax1.text(0, 0, f"총\n{n_total}개체", ha="center", va="center",
         fontsize=15, color="#555555", fontweight="bold")
ax1.set_title(f"t 평균 {t_mean:.4f} 기준\n개체 비율", fontsize=14,
              color="#222222", pad=12)

# --- 오른쪽: t 분포 산점도 ---
ax2 = axes[1]
ax2.set_facecolor("#FFF0F5")
xs = np.arange(1, n_total + 1)
ys = t_per.sort_index().values
below_mask = ys <= t_mean
ax2.scatter(xs[below_mask], ys[below_mask],
            c="#27AE60", s=70, alpha=0.85, edgecolors="white",
            linewidths=1.2, label=f"평균 이하 ({n_below}개)", zorder=3)
ax2.scatter(xs[~below_mask], ys[~below_mask],
            c="#C0392B", s=70, alpha=0.85, edgecolors="white",
            linewidths=1.2, label=f"평균 초과 ({n_above}개)", zorder=3)
ax2.axhline(t_mean, color="#C0392B", linewidth=2.2, linestyle="--",
            label=f"t 평균 = {t_mean:.4f}")
ax2.set_xlabel("개체 순번", fontsize=12, color="#333333")
ax2.set_ylabel("t (편차 평균)", fontsize=12, color="#333333")
ax2.set_title("개체별 t 값 분포", fontsize=14, color="#222222", pad=12)
ax2.tick_params(labelsize=10)
ax2.grid(True, color="#F0D9E0", linewidth=0.8)
ax2.legend(fontsize=11, loc="upper right", framealpha=0.85,
           facecolor="white", edgecolor="#E0C0CE")
ax2.set_ylim(bottom=0)

plt.tight_layout(pad=2.0)
plt.savefig("/home/user/outputs/t_analysis_pink.png", dpi=130,
            facecolor="#FFF0F5", bbox_inches="tight")
print("저장 완료: /home/user/outputs/t_analysis_pink.png")
