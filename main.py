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
st.set_page_config(page_title="극지식물 연구 가설과 분석 결과", layout="wide")

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

# ── 공통 색상/스타일 상수 ──────────────────────────────────────────────
BG_PINK  = "rgba(255,245,250,0.95)"
BG_BEIGE = "rgba(255,250,245,0.95)"
FONTC    = "#000000"
TFONTC   = "#111111"
GRIDC    = "#E8E0DA"
ZEROC    = "#CCCCCC"
HOVBG    = "#FFF8F0"

# ── hline + 주석 추가 헬퍼 (annotation_font 대체) ──────────────────────
def add_hline_annot(fig, y, color, label, pos="top right", size=12):
    """점선 hline + 오른쪽 위 라벨 주석을 한 번에 추가."""
    fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=2)
    fig.add_annotation(
        x=1.0, y=y,
        xref="paper", yref="y",
        text=label,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color=color, size=size),
    )

# ── 공통 레이아웃 (글자 전부 검정, 배경은 연한 핑크/베이지) ──────────
def base_layout(fig, title, x_title, y_title, bg=BG_PINK):
    # title 속 font로 제목 글꼴 지정 (titlefont → title.font 로 수정)
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color=FONTC),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        hoverlabel=dict(font=dict(color=FONTC), bgcolor=HOVBG,
                        bordercolor="#CCCCCC"),
        title=dict(text=title,
                   font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                             color=TFONTC, size=15),
                   x=0.03, xanchor="left"),
        xaxis=dict(
            title=dict(text=x_title,
                       font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                 color=FONTC, size=13)),
            color=FONTC,
            gridcolor=GRIDC, zerolinecolor=ZEROC,
            tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                          color=FONTC, size=12),
        ),
        yaxis=dict(
            title=dict(text=y_title,
                       font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                 color=FONTC, size=13)),
            color=FONTC,
            gridcolor=GRIDC, zerolinecolor=ZEROC,
            tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                          color=FONTC, size=12),
        ),
        margin=dict(l=60, r=30, t=65, b=60),
        hovermode="closest",
    )
    return fig

# ── 산점도 + 추세선 (점만 표시) ─────────────────────────────────────────
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
    fig.add_trace(go.Scatter(
        x=xy["x"], y=xy["y"],
        mode="markers",
        marker=dict(color=color, size=9, opacity=0.88,
                    line=dict(width=1.2, color="white")),
        name="데이터",
        hovertemplate="%{x:.2f}<br>%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xy["x"], y=pred,
        mode="lines",
        line=dict(color="#C0392B", width=2.5),
        name=f"추세선 (r={r:.3f})",
    ))
    title = f"{x_label} vs {y_label}  (상관계수 r = {r:.3f}"
    if p_val is not None:
        title += f", p = {p_val:.4f}"
    title += ")"
    base_layout(fig, title, x_label, y_label)
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
#  Tab 2 — 생육 데이터 값
# ────────────────────────────────────────────────────────────────────────
with tab2:
    try:
        st.header("📊 생육 데이터 값")

        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 분석이 표시됩니다.")
        else:
            below_col = "지하부 질량"
            leaf_col  = "잎 수"

            if below_col not in growth_df.columns or leaf_col not in growth_df.columns:
                st.info(f"지하부 질량({below_col})과 잎 수({leaf_col}) 열이 모두 필요합니다. "
                        "업로드한 CSV에 해당 열이 있는지 확인해주세요.")
            else:
                gw = growth_df[below_col].dropna()
                lf = growth_df[leaf_col].dropna()
                common = gw.index.intersection(lf.index)
                gw_c = gw.loc[common]
                lf_c = lf.loc[common]

                if len(gw_c) < 2 or len(lf_c) < 2:
                    st.info("데이터가 부족해 분석을 건너뛰었습니다.")
                else:
                    mean_below = gw_c.mean()
                    mean_leaf  = lf_c.mean()
                    abs_below  = (gw_c - mean_below).abs()
                    abs_leaf   = (lf_c - mean_leaf).abs()
                    t_per      = (abs_below + abs_leaf) / 2.0
                    t_mean     = t_per.mean()

                    n_total   = len(t_per)
                    n_below   = int((t_per <= t_mean).sum())
                    n_above   = int((t_per >  t_mean).sum())
                    pct_below = n_below  / n_total * 100
                    pct_above = n_above  / n_total * 100

                    # ── 결과 카드 ────────────────────────────────────────
                    st.subheader("📐 t 값 계산 요약")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("지하부 질량 평균", f"{mean_below:.4f} g")
                    m2.metric("잎 수평균", f"{mean_leaf:.1f} 장")
                    m3.metric("t 평균 (개체 편차 평균)", f"{t_mean:.4f}")

                    st.caption(
                        "• t = (|지하부질량−평균| + |잎수−평균|) ÷ 2  — "
                        "개체가 평균에서 얼마나 떨어져 있는지 나타내는 값\n"
                        "• t가 작을수록 평균에 가까운 '보통' 개체, 클수록 눈에 띄는 개체"
                    )

                    # ── 원 그래프: 평균 이하 vs 초과 (분홍 배경, 글씨 검정) ─
                    st.subheader("🥧 t 평균 기준 개체 비율 (원 그래프)")
                    fig_pie = go.Figure()
                    fig_pie.add_trace(go.Pie(
                        labels=[
                            f"t 평균 이하<br>{n_below}개체 · {pct_below:.1f}%",
                            f"t 평균 초과<br>{n_above}개체 · {pct_above:.1f}%",
                        ],
                        values=[n_below, n_above],
                        marker=dict(colors=["#FFB6C1", "#FF69B4"]),
                        hole=0.45,
                        textinfo="label+percent",
                        textposition="outside",
                        textfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                      color=FONTC, size=13),
                        hoverinfo="label+value+percent",
                        hoverlabel=dict(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                                  color=FONTC, size=13),
                                        bgcolor=HOVBG, bordercolor="#CCCCCC"),
                    ))
                    base_layout(fig_pie,
                                f"t 평균 {t_mean:.4f} 기준 — 총 {n_total}개체",
                                "", "")
                    fig_pie.update_layout(margin=dict(l=30, r=30, t=65, b=40))
                    st.plotly_chart(fig_pie, width="stretch")

                    # ── t 분포 산점도 (점만) ─────────────────────────────
                    st.subheader("📈 개체별 t 값 분포 (산점도)")
                    idx_sorted = t_per.sort_index()
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(
                        x=list(range(1, len(idx_sorted) + 1)),
                        y=idx_sorted.values,
                        mode="markers",
                        marker=dict(
                            color="#E67E22", size=11, opacity=0.85,
                            line=dict(width=1.5, color="white"),
                        ),
                        name="t 값",
                        hovertemplate="개체 %{x}<br>t = %{y:.4f}<extra></extra>",
                    ))
                    add_hline_annot(fig_t, t_mean, "#C0392B",
                                    f"t 평균 = {t_mean:.4f}", size=12)
                    base_layout(fig_t,
                                "개체별 t 값 (주황 점 = 개별 개체, 빨간 점선 = t 평균)",
                                "개체 순번", "t (편차 평균)")
                    st.plotly_chart(fig_t, width="stretch")

                    # ── 막대 그래프 ───────────────────────────────────────
                    st.subheader("📊 개체별 t vs t 평균 (막대 그래프)")
                    colors_map = {False: "#27AE60", True: "#C0392B"}
                    bar_colors = [colors_map[v > t_mean] for v in idx_sorted.values]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=list(range(1, len(idx_sorted) + 1)),
                        y=idx_sorted.values,
                        marker=dict(color=bar_colors, line=dict(width=0.8, color="white"),
                                    opacity=0.90),
                        name="개체별 t",
                        hovertemplate="개체 %{x}<br>t = %{y:.4f}<extra></extra>",
                    ))
                    add_hline_annot(fig_bar, t_mean, "#555555", "t 평균", size=12)
                    base_layout(fig_bar,
                                "개체별 t 값 — 🟢 평균 이하 · 🔴 평균 초과",
                                "개체 순번", "t", bg=BG_BEIGE)
                    st.plotly_chart(fig_bar, width="stretch")

                    # ── 꺾은선 그래프 ─────────────────────────────────────
                    st.subheader("📉 t 평균 대비 각 개체 t 위치 (꺾은선 그래프)")
                    fig_line = go.Figure()
                    add_hline_annot(fig_line, t_mean, "#AAAAAA", "t 평균", size=12)
                    for i, (pos, tv) in enumerate(
                        zip(range(1, len(idx_sorted) + 1), idx_sorted.values)
                    ):
                        fig_line.add_trace(go.Scatter(
                            x=[pos, pos], y=[t_mean, tv],
                            mode="lines", line=dict(color="#85C1E9", width=1.8),
                            showlegend=False, hoverinfo="skip",
                        ))
                    fig_line.add_trace(go.Scatter(
                        x=list(range(1, len(idx_sorted) + 1)),
                        y=idx_sorted.values,
                        mode="markers",
                        marker=dict(color="#1A5276", size=10, opacity=0.92,
                                    line=dict(width=1.5, color="white")),
                        name="개체 t",
                        hovertemplate="개체 %{x}<br>t = %{y:.4f}<extra></extra>",
                    ))
                    fig_line.add_trace(go.Scatter(
                        x=[0], y=[t_mean],
                        mode="markers",
                        marker=dict(color="#7F8C8D", size=14, symbol="diamond",
                                    line=dict(width=1.5, color="white")),
                        name="t 평균",
                        hovertemplate="t 평균 = %{y:.4f}<extra></extra>",
                    ))
                    base_layout(fig_line,
                                "t 평균(회색 마름모)에서 각 개체 t(파란 점)까지의 거리",
                                "개체 순번", "t", bg=BG_BEIGE)
                    st.plotly_chart(fig_line, width="stretch")

                    # ── 상관관계 산점도 ──────────────────────────────────
                    st.subheader("🔗 지하부 질량 vs 잎 수 (상관계수·추세선)")
                    fig_corr = scatter_trend(gw_c, lf_c,
                                             "지하부 질량 (g)", "잎 수 (장)",
                                             color="#2E86C1")
                    if fig_corr is not None:
                        st.plotly_chart(fig_corr, width="stretch")
                    else:
                        st.info("데이터가 부족해 상관관계 산점도를 그리지 못했습니다.")

                    # ── 상관계수 카드 ────────────────────────────────────
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
                        sig = ("✅ 통계적으로 유의" if p_val < 0.05
                               else "⚠️ 유의하지 않음 (p ≥ 0.05)")
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
            leaf_col  = "잎 수"

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

                    st.subheader("📋 종합 결과")
                    total_col1, total_col2 = st.columns(2)
                    with total_col1:
                        st.dataframe(
                            pd.DataFrame({
                                "지표": ["개체 수", "지하부 질량 평균 (g)",
                                         "잎 수평균 (장)", "상관계수 r"],
                                "값": [str(len(gw_c)), f"{gw_c.mean():.4f}",
                                       f"{lf_c.mean():.1f}", f"{r_val:.4f}"],
                            }),
                            width="stretch", hide_index=True,
                        )
                    with total_col2:
                        stance = ("가설 지지"
                                  if (r_val > 0 and p_val is not None
                                      and p_val < 0.05)
                                  else "가설 지지 불명확 또는 기각")
                        direction_note = ("(정방향)" if r_val > 0 else "(역방향 또는 무관)")
                        st.markdown(f"""
**한 줄 요약**
- 총 {len(gw_c)}개체를 분석한 결과,
- 지하부 질량과 잎 수의 상관계수는 **r = {r_val:.3f}** {direction_note}
- {stance}
""")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
