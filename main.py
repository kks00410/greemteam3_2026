import io, os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 (Streamlit) ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="송도고등학교 그린3팀 결과", layout="wide")

# ── CSV 읽기 (인코딩 자동 전환) ─────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, dtype=str)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 (KeyError 방지) ──────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").
                str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                          pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

def school_name_from_filename(filename):
    base = os.path.basename(filename)
    m = re.match(r"^([가-힣A-Za-z0-9]+(?:고|중|초|대학교|학교)?)", base)
    return m.group(1) if m else "우리 학교"

# ── 산점도 + 추세선 헬퍼 ─────────────────────────────────────────────────
def scatter_trend(x, y, x_label, y_label, color="#2196F3"):
    xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(xy) < 2:
        return None
    slope, intercept = np.polyfit(xy["x"], xy["y"], 1)
    pred = slope * xy["x"] + intercept
    r = np.corrcoef(xy["x"], xy["y"])[0, 1]
    p_val = None
    try:
        from scipy import stats
        _, p_val = stats.pearsonr(xy["x"], xy["y"])
    except Exception:
        pass
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy["x"], y=xy["y"], mode="markers",
                             marker=dict(color=color, size=8, opacity=0.85),
                             name="데이터"))
    fig.add_trace(go.Scatter(x=xy["x"], y=pred, mode="lines",
                             line=dict(color="red", width=2),
                             name=f"추세선 (r={r:.3f})"))
    title = f"{x_label} vs {y_label}  (상관계수 r = {r:.3f}"
    if p_val is not None:
        title += f", p = {p_val:.4f}"
    title += ")"
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

# ════════════════════════════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════════════════════════════
st.sidebar.title("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

@st.cache_data
def load_data(env_file, growth_file):
    env_df, growth_df = pd.DataFrame(), pd.DataFrame()
    school = "우리 학교"
    if env_file is not None:
        school = school_name_from_filename(env_file.name)
        env_df = std_cols(read_csv(env_file))
    if growth_file is not None:
        s2 = school_name_from_filename(growth_file.name)
        if school == "우리 학교":
            school = s2
        growth_df = std_cols(read_csv(growth_file))
    return env_df, growth_df, school

with st.sidebar.spinner("데이터를 불러오는 중..."):
    env_df, growth_df, school = load_data(env_file, growth_file)

st.sidebar.markdown(f"🏫 **{school}** 극지식물 연구")
st.sidebar.caption("파일명 앞부분에서 학교명을 자동으로 읽어요.")

# ════════════════════════════════════════════════════════════════════════
#  탭 구성
# ════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📋 실험 소개", "📊 생육 데이터 값", "✅ 결과 및 가설 확인"])

# ────────────────────────────────────────────────────────────────────────
#  Tab 1 — 실험 소개
# ────────────────────────────────────────────────────────────────────────
with tab1:
    try:
        st.header("📋 실험 소개")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
**연구 목적**
극지식물 개체마다 잎의 수와 지하부(뿌리) 질량이 어떻게 다른지 알아보고,
두 특성 사이에 어떤 관계가 있는지 확인하는 실험입니다.

**개체 수**
- 업로드한 생육 CSV의 행 수가 곧 개체 수입니다.
- 아래 데이터 요약에서 자동으로 표시됩니다.
""")
        with col2:
            st.markdown("""
**연구 가설**
> "지하부 질량이 무거울수록(뿌리가 더 많이 자랄수록) 잎의 개수가 많을 것이다."

이 가설은 식물이 뿌리를 통해 물과 양분을 흡수하므로,
뿌리가 잘 발달한 개체일수록 잎도 더 많이 낼 것이라는 생각에서 출발합니다.

이 가설이 실제로 맞는지 아래 분석에서 확인해 봅니다.
""")

        # 데이터 요약
        st.subheader("📋 데이터 요약")
        if growth_df.empty and env_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 올려주세요.")
        else:
            rows = []
            if not growth_df.empty:
                n = len(growth_df)
                cols_str = ", ".join(
                    [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                 "생중량", "지상부 질량", "지하부 질량"]
                     if c in growth_df.columns]
                )
                rows.append(["생육 결과 데이터", f"{n}개체", cols_str])
            if not env_df.empty:
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                            if c in env_df.columns]
                rows.append(["환경 데이터", f"{len(env_df)}행",
                             ", ".join(env_cols) if env_cols else "없음"])
            if rows:
                st.dataframe(pd.DataFrame(rows, columns=["데이터", "규모", "열"]),
                             width="stretch", hide_index=True)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ────────────────────────────────────────────────────────────────────────
#  Tab 2 — 생육 데이터 값 (t 계산, 산점도, 막대·꺾은선)
# ────────────────────────────────────────────────────────────────────────
with tab2:
    try:
        st.header("📊 생육 데이터 값")

        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 분석이 표시됩니다.")
        else:
            below_col = "지하부 질량"
            leaf_col = "잎 수"

            # ── 분석에 필요한 열 존재 확인 ─────────────────────────────
            if below_col not in growth_df.columns or leaf_col not in growth_df.columns:
                st.info(f"지하부 질량({below_col})과 잎 수({leaf_col}) 열이 모두 필요합니다. "
                        "업로드한 CSV에 해당 열이 있는지 확인해주세요.")
            else:
                # ── 1단계: 각 개체의 지하부 질량·잎 수에서 평균 뺀 절댓값 → t ──
                gw = growth_df[below_col].dropna()
                lf = growth_df[leaf_col].dropna()
                # 두 열이 같은 행 인덱스를 공유하도록 교집합
                common = gw.index.intersection(lf.index)
                gw_c = gw.loc[common]
                lf_c = lf.loc[common]

                if len(gw_c) < 2 or len(lf_c) < 2:
                    st.info("데이터가 부족해 분석을 건너뛰었습니다.")
                else:
                    mean_below = gw_c.mean()
                    mean_leaf = lf_c.mean()

                    # 개체별 편차 절댓값
                    abs_below = (gw_c - mean_below).abs()
                    abs_leaf = (lf_c - mean_leaf).abs()

                    # t = 두 절댓값의 평균 (개체별)
                    t_per = (abs_below + abs_leaf) / 2.0

                    # t의 전체 평균
                    t_mean = t_per.mean()

                    # ── 결과 카드 ────────────────────────────────────────
                    st.subheader("📐 t 값 계산 요약")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("지하부 질량 평균", f"{mean_below:.4f} g")
                    m2.metric("잎 수평균", f"{mean_leaf:.1f} 장")
                    m3.metric("t 평균 (개체 편차 평균)",
                              f"{t_mean:.4f}")

                    st.caption(
                        "• t = (|지하부질량−평균| + |잎수−평균|) ÷ 2  — "
                        "개체가 평균에서 얼마나 떨어져 있는지 나타내는 값\n"
                        "• t가 작을수록 평균에 가까운 '보통' 개체, 클수록 눈에 띄는 개체"
                    )

                    # ── 2-1: t의 분포 산점도 (x=개체 순번, y=t) ──────────
                    st.subheader("📈 개체별 t 값 분포 (산점도)")
                    idx_sorted = t_per.sort_index()
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(
                        x=list(range(1, len(idx_sorted)+1)),
                        y=idx_sorted.values,
                        mode="markers+lines",
                        marker=dict(color="#FF9800", size=8),
                        line=dict(color="#FF9800", width=1.5, dash="dot"),
                        name="t 값",
                    ))
                    fig_t.add_hline(y=t_mean, line_dash="dash", line_color="red",
                                    annotation_text=f"t 평균 = {t_mean:.4f}",
                                    annotation_position="top right")
                    fig_t.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        title="개체별 t 값 (점선 = t 전체 평균)",
                        xaxis_title="개체 순번",
                        yaxis_title="t (편차 평균)",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_t, width="stretch")

                    # ── 2-2: 막대 그래프 — 각 개체 t vs t 평균 ────────────
                    st.subheader("📊 개체별 t vs t 평균 (막대 그래프)")
                    fig_bar = go.Figure()
                    t_vals = idx_sorted.values
                    colors_bar = [
                        "#4CAF50" if v <= t_mean else "#F44336"
                        for v in t_vals
                    ]
                    fig_bar.add_trace(go.Bar(
                        x=list(range(1, len(t_vals)+1)),
                        y=t_vals,
                        marker_color=colors_bar,
                        name="개체별 t",
                    ))
                    fig_bar.add_hline(y=t_mean, line_dash="dash", line_color="black",
                                      annotation_text="t 평균", annotation_position="top right")
                    fig_bar.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        title="개체별 t 값 — 🟢 평균 이하 · 🔴 평균 초과",
                        xaxis_title="개체 순번",
                        yaxis_title="t",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                    # ── 2-3: 꺾은선 그래프 — t 평균에서 각 개체 t까지 거리 ──
                    st.subheader("📉 t 평균 대비 각 개체 t 위치 (꺾은선 그래프)")
                    fig_line = go.Figure()
                    fig_line.add_hline(y=t_mean, line_dash="dash", line_color="gray",
                                       annotation_text="t 평균", annotation_position="top right")
                    # 평균 선에서 각 개체 t로 가는 선분
                    for i, (pos, tv) in enumerate(zip(range(1, len(t_vals)+1), t_vals)):
                        fig_line.add_trace(go.Scatter(
                            x=[pos, pos], y=[t_mean, tv],
                            mode="lines",
                            line=dict(color="#90CAF9", width=1.5),
                            showlegend=False,
                        ))
                    # 개체 t 점
                    fig_line.add_trace(go.Scatter(
                        x=list(range(1, len(t_vals)+1)),
                        y=t_vals,
                        mode="markers",
                        marker=dict(color="#1565C0", size=9),
                        name="개체 t",
                    ))
                    fig_line.add_trace(go.Scatter(
                        x=[0], y=[t_mean],
                        mode="markers",
                        marker=dict(color="gray", size=12, symbol="diamond"),
                        name="t 평균",
                    ))
                    fig_line.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        title="t 평균(회색 마름모)에서 각 개체 t(파란 점)까지의 거리",
                        xaxis_title="개체 순번",
                        yaxis_title="t",
                        margin=dict(l=40, r=20, t=50, b=40),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_line, width="stretch")

                    # ── 3: 지하부 질량 vs 잎 수 산점도 + 추세선 ──────────
                    st.subheader("🔗 지하부 질량 vs 잎 수 (상관계수·추세선)")
                    fig_corr = scatter_trend(gw_c, lf_c,
                                             "지하부 질량 (g)", "잎 수 (장)",
                                             color="#2196F3")
                    if fig_corr is not None:
                        st.plotly_chart(fig_corr, width="stretch")
                    else:
                        st.info("데이터가 부족해 상관관계 산점도를 그리지 못했습니다.")

                    # ── 상관계수 수치 카드 ────────────────────────────────
                    r_val = np.corrcoef(gw_c, lf_c)[0, 1]
                    p_val = None
                    try:
                        from scipy import stats
                        _, p_val = stats.pearsonr(gw_c, lf_c)
                    except Exception:
                        pass
                    st.subheader("📐 상관계수 요약")
                    r_col, p_col = st.columns(2)
                    r_col.metric("상관계수 r", f"{r_val:.4f}")
                    if p_val is not None:
                        sig = "✅ 통계적으로 유의" if p_val < 0.05 else "⚠️ 유의하지 않음 (p ≥ 0.05)"
                        p_col.metric("p 값", f"{p_val:.4f}", delta=sig)
                    else:
                        p_col.metric("p 값", "계산 불가")
                    st.info(
                        "상관계수 r은 -1 ~ +1 사이 값입니다. "
                        "+1에 가까울수록 두 값이 함께 증가(정방향), "
                        "-1에 가까울수록 한쪽 증가 시 다른 쪽 감소(역방향), "
                        "0에 가까우면 관계가 약합니다."
                    )

        # ── Expander: 생육 원본 + CSV 다운로드 ─────────────────────────
        with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
            if not growth_df.empty:
                disp = growth_df.copy()
                if "개체번호" in disp.columns:
                    disp["개체번호"] = disp["개체번호"].apply(
                        lambda v: int(v) if isinstance(v, float) and v == int(v) else v
                    )
                st.dataframe(disp, width="stretch", hide_index=True)
                csv_b = disp.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇ 생육 결과 CSV 다운로드",
                    data=csv_b,
                    file_name=f"{school}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ────────────────────────────────────────────────────────────────────────
#  Tab 3 — 결과 및 가설 확인
# ────────────────────────────────────────────────────────────────────────
with tab3:
    try:
        st.header("✅ 결과 및 가설 확인")

        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 결과 해석이 표시됩니다.")
        else:
            below_col = "지하부 질량"
            leaf_col = "잎 수"

            if below_col not in growth_df.columns or leaf_col not in growth_df.columns:
                st.info(f"지하부 질량({below_col})과 잎 수({leaf_col}) 열이 모두 필요합니다.")
            else:
                gw = growth_df[below_col].dropna()
                lf = growth_df[leaf_col].dropna()
                common = gw.index.intersection(lf.index)
                gw_c = gw.loc[common]
                lf_c = lf.loc[common]

                if len(gw_c) < 2:
                    st.info("데이터가 부족해 가설 확인을 건너뛰었습니다.")
                else:
                    r_val = np.corrcoef(gw_c, lf_c)[0, 1]
                    p_val = None
                    try:
                        from scipy import stats
                        _, p_val = stats.pearsonr(gw_c, lf_c)
                    except Exception:
                        pass

                    # ── 가설 판정 ──────────────────────────────────────────
                    st.subheader("🔬 가설 검토")
                    hyp_col1, hyp_col2 = st.columns(2)
                    with hyp_col1:
                        st.markdown("""
**가설**
> "지하부 질량이 무거울수록 잎의 개수가 많을 것이다."
""")
                    with hyp_col2:
                        if p_val is not None and p_val < 0.05:
                            direction = "정방향" if r_val > 0 else "역방향"
                            verdict = (
                                f"✅ 가설이 **지지됨** (p < 0.05, r = {r_val:.3f}, {direction})"
                                if r_val > 0 else
                                f"❌ 가설이 **기각됨** — 오히려 역방향 관계 (r = {r_val:.3f})"
                            )
                        elif p_val is not None:
                            verdict = (
                                f"⚠️ **유의하지 않음** (p = {p_val:.3f} ≥ 0.05) — "
                                "이 데이터만으로는 가설의 옳고 그름을 단정하기 어려움"
                            )
                        else:
                            verdict = (
                                f"상관계수 r = {r_val:.3f} — "
                                "p 값을 계산할 수 없어 통계적 유의성은 판단 불가"
                            )
                        st.markdown(f"""
**판단 결과**
{verdict}
""")

                    # ── 가설이 틀렸을 때 원인 분석 ────────────────────────
                    if r_val <= 0 or (p_val is not None and p_val >= 0.05):
                        st.subheader("🔍 가설이 맞지 않을 수 있는 원인")
                        st.markdown("""
지하부 질량과 잎 수 사이에 뚜렷한 정방향 관계가 나타나지 않은 이유는
다음과 같이 생각해볼 수 있습니다.

1. **지하부 질량이 잎 수를 직접 결정하지 않을 수 있음**
   잎의 수는 유전, 광량, 생장 단계, 호르몬 등 여러 요인의 영향을 받아요.
   뿌리 질량만으로 잎 수를 예측하기는 어렵습니다.

2. **개체 간 생장 단계가 다를 수 있음**
   아직 어린 개체는 잎이 적어도 뿌리가 먼저 자라고,
   성숙한 개체는 잎이 많지만 뿌리 성장이 둔화되었을 수 있어요.
   같은 시기에 측정했더라도 실제 발달 단계는 다를 수 있습니다.

3. **환경이 개체마다 균일하지 않았을 수 있음**
   같은 실험실 안에서도 위치에 따라 빛·온도·수분 조건이 조금씩 다를 수 있고,
   이것이 뿌리 성장과 잎 수에 서로 다른 영향을 주었을 수 있습니다.

4. **측정 개체 수가 적거나 편차가 클 수 있음**
   상관계수는 데이터가 적고 편차가 크면 불안정해집니다.
   81개체라도 개체 간 차이가 매우 크면 r이 작게 나올 수 있어요.

5. **두 변수가 서로 다른 '자원 배분 전략'을 반영할 수 있음**
   어떤 개체는 뿌리(지하부)에 자원을 많이 투자하고 잎은 적게 내는 전략,
   다른 개체는 잎을 많이 내되 뿌리는 적게 키우는 전략을 취할 수 있어요.
   이 경우 두 값은 오히려 역방향이거나 무관할 수 있습니다.
""")
                    else:
                        st.markdown("""
**해석**
상관계수 r이 양수이고 통계적으로 유의하므로,
이 데이터에서는 "지하부 질량이 무거울수록 잎 수가 많다"는 가설이
 지지된다고 볼 수 있습니다.

다만 상관관계는 인과관계를 보장하지 않으므로,
"뿌리가 무거워서 잎이 많아졌다"라고 단정하기보다는
"두 특성이 함께 커지는 경향이 관찰되었다"고 해석하는 것이 적절합니다.
""")

                    # ── 종합 정리 ─────────────────────────────────────────
                    st.subheader("📋 종합 결과")
                    total_col1, total_col2 = st.columns(2)
                    with total_col1:
                        st.dataframe(
                            pd.DataFrame({
                                "지표": ["개체 수", "지하부 질량 평균 (g)",
                                         "잎 수평균 (장)", "상관계수 r"],
                                "값": [str(len(gw_c)),
                                       f"{gw_c.mean():.4f}",
                                       f"{lf_c.mean():.1f}",
                                       f"{r_val:.4f}"],
                            }),
                            width="stretch", hide_index=True,
                        )
                    with total_col2:
                        st.markdown(f"""
**한 줄 요약**
- 총 {len(gw_c)}개체를 분석한 결과,
- 지하부 질량과 잎 수의 상관계수는 **r = {r_val:.3f}**
  {'(정방향)' if r_val > 0 else '(역방향 또는 무관)'}
- {'가설 지지' if (r_val > 0 and p_val is not None and p_val < 0.05) else '가설 지지 불명확 또는 기각'}
""")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
