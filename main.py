import io, os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

# ── CSV 읽기 (인코딩 자동 전환) ─────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """file.getvalue()로 바이트를 꺼내 인코딩 순회 시도."""
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
    """파일명에서 학교명 추출 (예: 송도고_환경데이터.csv → 송도고)."""
    base = os.path.basename(filename)
    m = re.match(r"^([가-힣A-Za-z0-9]+(?:고|중|초|대학교|학교)?)", base)
    return m.group(1) if m else "우리 학교"

# ── 그래프 헬퍼 ─────────────────────────────────────────────────────────
def plotly_layout(fig, title=None):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="closest",
    )

def bar_mean(df, col, title, color="#4CAF50"):
    """평균 막대 그래프 (개체번호 제외, 값이 있는 행만)."""
    d = df[[col]].dropna()
    if len(d) < 1:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(d)+1)), y=d[col].values,
                         marker_color=color, name=col))
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        title=title,
        xaxis_title="개체 순번",
        yaxis_title=col,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

def scatter_with_trend(x, y, x_label, y_label, color="#2196F3"):
    """산점도 + np.polyfit 추세선 (값 2개 미만이면 None)."""
    xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(xy) < 2:
        return None
    slope, intercept = np.polyfit(xy["x"], xy["y"], 1)
    pred = slope * xy["x"] + intercept
    r = np.corrcoef(xy["x"], xy["y"])[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy["x"], y=xy["y"], mode="markers",
                             marker=dict(color=color, size=7, opacity=0.8),
                             name="데이터"))
    fig.add_trace(go.Scatter(x=xy["x"], y=pred, mode="lines",
                             line=dict(color="red", width=2),
                             name=f"추세선 (r={r:.3f})"))
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        title=f"{x_label} vs {y_label}  (상관계수 r = {r:.3f})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

# ════════════════════════════════════════════════════════════════════════
#  사이드바: CSV 업로드
# ════════════════════════════════════════════════════════════════════════
st.sidebar.title("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

growth_options = ["생중량(g)", "잎 수", "지상부 길이", "지하부 길이"]
growth_sel = st.sidebar.multiselect(
    "생육 측정 항목 선택 (Tab 3 그래프에서 사용)",
    growth_options,
    default=["생중량(g)", "잎 수", "지상부 길이", "지하부 길이"],
)

# ── 업로드된 파일 읽기 (한 번만, 캐싱 활용) ─────────────────────────────
@st.cache_data
def load_data(env_file, growth_file):
    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    school = "우리 학교"
    if env_file is not None:
        school = school_name_from_filename(env_file.name)
        env_df = std_cols(read_csv(env_file))
    if growth_file is not None:
        school2 = school_name_from_filename(growth_file.name)
        if school == "우리 학교":
            school = school2
        growth_df = std_cols(read_csv(growth_file))
    return env_df, growth_df, school

with st.sidebar.spinner("데이터를 불러오는 중..."):
    env_df, growth_df, school = load_data(env_file, growth_file)

st.sidebar.markdown(f"🏫 **{school}** 극지식물 대시보드")
st.sidebar.caption(
    "파일명 앞부분에서 학교명을 자동으로 읽어요. "
    "다른 학교 CSV를 올려도 바로 동작합니다."
)

# ════════════════════════════════════════════════════════════════════════
#  탭 구성
# ════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ────────────────────────────────────────────────────────────────────────
#  Tab 1 — 실험 개요
# ────────────────────────────────────────────────────────────────────────
with tab1:
    try:
        st.header("📖 실험 개요")

        # 연구 배경 및 목적
        col_bg, col_obj = st.columns([1, 2])
        with col_bg:
            st.markdown("""
**연구 배경**
극지식물은 낮은 온도와 적은 물에서도 살아남는 독특한 식물이에요.
학교 실험실에서 키운 극지식물이 개체마다 얼마나 다르게 자라는지 알아보면,
식물의 생존 전략을 이해하는 데 도움이 됩니다.
""")
        with col_obj:
            st.markdown("""
**연구 목적**
1. 개체별 생중량 분포와 차이를 평균·표준편차·변동계수(CV)로 알아보기
2. 잎 수·지상부 길이와 생중량의 관계를 상관계수·산점도로 확인하기
3. 생중량 상위 25% 그룹과 하위 25% 그룹을 비교하기

> **통계 용어 간단 설명**
> - **평균**: 모든 값을 더해 개수로 나눈 값 — 대표적인 중심 값
> - **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 수 — 클수록 차이가 큼
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 값(%) — 단위가 다른 그룹끼리 비교할 때 유용
> - **상관계수(r)**: 두 값이 함께 움직이는 정도 — -1 ~ +1 사이, 0에 가까울수록 관계 약함
""")

        # 데이터 요약 표
        if env_df.empty and growth_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 올려주세요. 업로드하면 아래 내용이 자동으로 채워집니다.")
        else:
            summary_rows = []
            if growth_df is not None and not growth_df.empty:
                n_growth = len(growth_df)
                growth_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"] if c in growth_df.columns]
                summary_rows.append(["생육 결과 데이터", f"{n_growth}개체", ", ".join(growth_cols) if growth_cols else "없음"])
            if env_df is not None and not env_df.empty:
                time_col = "time" if "time" in env_df.columns else None
                if time_col:
                    t_min = env_df[time_col].min()
                    t_max = env_df[time_col].max()
                    period = f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}"
                else:
                    period = "확인 불가"
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                summary_rows.append(["환경 데이터", f"{len(env_df)}행", f"측정 기간: {period}  |  항목: {', '.join(env_cols) if env_cols else '없음'}"])

            if summary_rows:
                st.subheader("📋 데이터 요약")
                st.dataframe(pd.DataFrame(summary_rows, columns=["데이터", "규모", "측정 항목 / 기간"]),
                             width="stretch", hide_index=True)

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")
        m1, m2, m3, m4 = st.columns(4)

        # 총 개체수
        if not growth_df.empty and "개체번호" in growth_df.columns:
            m1.metric("총 개체수", f"{len(growth_df)}개체")
        else:
            m1.metric("총 개체수", "—")

        # 평균 온도
        if not env_df.empty and "temperature" in env_df.columns:
            t_mean = env_df["temperature"].dropna().mean()
            if t_mean == t_mean:  # NaN 체크
                m2.metric("평균 온도", f"{t_mean:.1f} °C")
            else:
                m2.metric("평균 온도", "—")
        else:
            m2.metric("평균 온도", "—")

        # 평균 습도
        if not env_df.empty and "humidity" in env_df.columns:
            h_mean = env_df["humidity"].dropna().mean()
            if h_mean == h_mean:
                m3.metric("평균 습도", f"{h_mean:.1f} %")
            else:
                m3.metric("평균 습도", "—")
        else:
            m3.metric("평균 습도", "—")

        # 평균 생중량
        if not growth_df.empty and "생중량" in growth_df.columns:
            gw_mean = growth_df["생중량"].dropna().mean()
            if gw_mean == gw_mean:
                m4.metric("평균 생중량", f"{gw_mean:.3f} g")
            else:
                m4.metric("평균 생중량", "—")
        else:
            m4.metric("평균 생중량", "—")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ────────────────────────────────────────────────────────────────────────
#  Tab 2 — 환경 데이터
# ────────────────────────────────────────────────────────────────────────
with tab2:
    try:
        st.header("🌡️ 환경 데이터")
        if env_df.empty:
            st.info("환경 데이터 CSV를 업로드하면 환경 요약·시계열 그래프가 표시됩니다.")
        else:
            # 환경 요약 표 (있는 열만)
            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                            if c in env_df.columns]
            if env_num_cols:
                st.subheader("📋 환경 요약 (평균 · 최소 · 최대)")
                summary = pd.DataFrame({
                    "항목": env_num_cols,
                    "평균": [round(env_df[c].dropna().mean(), 3) for c in env_num_cols],
                    "최소": [round(env_df[c].dropna().min(), 3) for c in env_num_cols],
                    "최대": [round(env_df[c].dropna().max(), 3) for c in env_num_cols],
                })
                st.dataframe(summary, width="stretch", hide_index=True)

            # 일별 시계열
            if "time" in env_df.columns:
                st.subheader("📈 날짜별 시계열 (일별 평균)")
                env_df["date"] = env_df["time"].dt.date

                temp_daily = env_df.groupby("date")["temperature"].mean() if "temperature" in env_df.columns else None
                hum_daily = env_df.groupby("date")["humidity"].mean() if "humidity" in env_df.columns else None
                ph_daily = env_df.groupby("date")["ph"].mean() if "ph" in env_df.columns else None
                ec_daily = env_df.groupby("date")["ec"].mean() if "ec" in env_df.columns else None

                fig_timeline = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.12,
                    subplot_titles=("온도 · 습도 변화", "pH · EC 변화 (있는 경우만)"),
                )

                color_temp, color_hum = "#FF5722", "#2196F3"
                if temp_daily is not None:
                    fig_timeline.add_trace(
                        go.Scatter(x=temp_daily.index, y=temp_daily.values,
                                   mode="lines", name="평균 온도 (°C)",
                                   line=dict(color=color_temp, width=2)),
                        row=1, col=1)
                if hum_daily is not None:
                    fig_timeline.add_trace(
                        go.Scatter(x=hum_daily.index, y=hum_daily.values,
                                   mode="lines", name="평균 습도 (%)",
                                   line=dict(color=color_hum, width=2),
                                   yaxis="y2"),
                        row=1, col=1)

                if temp_daily is not None or hum_daily is not None:
                    fig_timeline.update_yaxes(title_text="°C / %", row=1, col=1)
                    if temp_daily is not None and hum_daily is not None:
                        fig_timeline.update_yaxes(title_text="%", row=1, col=1, secondary_y=True)

                color_ph, color_ec = "#9C27B0", "#4CAF50"
                if ph_daily is not None:
                    fig_timeline.add_trace(
                        go.Scatter(x=ph_daily.index, y=ph_daily.values,
                                   mode="lines", name="평균 pH",
                                   line=dict(color=color_ph, width=2, dash="dot")),
                        row=2, col=1)
                if ec_daily is not None:
                    fig_timeline.add_trace(
                        go.Scatter(x=ec_daily.index, y=ec_daily.values,
                                   mode="lines", name="평균 EC (mS/cm)",
                                   line=dict(color=color_ec, width=2)),
                        row=2, col=1)

                fig_timeline.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(l=40, r=20, t=50, b=40),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_timeline, width="stretch")

            # Expander: 원본 테이블 + CSV 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                if not env_df.empty:
                    disp = env_df.copy()
                    # 시간 열은 string으로 표시
                    if "time" in disp.columns:
                        disp["time"] = disp["time"].astype(str)
                    st.dataframe(disp, width="stretch", hide_index=True)
                    csv_bytes = disp.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇ 환경 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school}_환경데이터.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ────────────────────────────────────────────────────────────────────────
#  Tab 3 — 생육 결과
# ────────────────────────────────────────────────────────────────────────
with tab3:
    try:
        st.header("📊 생육 결과")
        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 생육 분석 그래프가 표시됩니다.")
        else:
            gw_col = "생중량"
            leaf_col = "잎 수"
            above_col = "지상부 길이"
            below_col = "지하부 길이"
            above_mass_col = "지상부 질량"
            below_mass_col = "지하부 질량"

            # ── 🥇 핵심 결과 카드 ──────────────────────────────────────
            st.subheader("🥇 핵심 결과")
            if gw_col in growth_df.columns:
                gw_clean = growth_df[gw_col].dropna()
                if len(gw_clean) >= 1:
                    gw_mean = gw_clean.mean()
                    gw_max_val = gw_clean.max()
                    gw_min_val = gw_clean.min()
                    idx_max = gw_clean.idxmax()
                    idx_min = gw_clean.idxmin()
                    max_id = growth_df.loc[idx_max, "개체번호"] if "개체번호" in growth_df.columns else "?"
                    min_id = growth_df.loc[idx_min, "개체번호"] if "개체번호" in growth_df.columns else "?"

                    c_max, c_mean, c_min = st.columns(3)
                    c_mean.metric("평균 생중량", f"{gw_mean:.3f} g")
                    c_max.metric("최대 생중량", f"{gw_max_val:.3f} g",
                                 delta=f"개체 {max_id}")
                    c_min.metric("최소 생중량", f"{gw_min_val:.3f} g",
                                 delta=f"개체 {min_id}")
                else:
                    st.info("생중량 데이터가 없습니다.")
            else:
                st.info("생중량(g) 열이 없어 핵심 결과 카드를 표시할 수 없습니다.")

            # ── 개체별 생육 비교 (막대 그래프) ──────────────────────────
            st.subheader("📈 개체별 생육 비교")
            plot_cols = [c for c in [gw_col, leaf_col, above_col, below_col] if c in growth_df.columns]
            plot_cols = [c for c in growth_sel if c in plot_cols] if growth_sel else plot_cols
            if plot_cols:
                for col in plot_cols:
                    d = growth_df[[col]].dropna()
                    if len(d) < 1:
                        continue
                    # 생중량은 상위·하위 25%를 다른 색으로
                    if col == gw_col:
                        q1 = d[gw_col].quantile(0.25)
                        q3 = d[gw_col].quantile(0.75)
                        colors = [
                            "#F44336" if v <= q1 else
                            "#4CAF50" if v >= q3 else
                            "#90CAF9"
                            for v in d[gw_col]
                        ]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(range(1, len(d)+1)),
                            y=d[gw_col].values,
                            marker_color=colors,
                            name=col,
                        ))
                        fig.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            title=f"{col} — 개체별 (🔴 하위 25% · 🟢 상위 25% 강조)",
                            xaxis_title="개체 순번",
                            yaxis_title=col,
                            margin=dict(l=40, r=20, t=50, b=40),
                        )
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(range(1, len(d)+1)),
                            y=d[col].values,
                            marker_color="#7E57C2",
                            name=col,
                        ))
                        fig.update_layout(
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            title=f"{col} — 개체별",
                            xaxis_title="개체 순번",
                            yaxis_title=col,
                            margin=dict(l=40, r=20, t=50, b=40),
                        )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("선택한 생육 측정 항목이 데이터에 없습니다.")

            # ── 생중량 분포 (박스플롯) ──────────────────────────────────
            if gw_col in growth_df.columns:
                gw_clean = growth_df[gw_col].dropna()
                if len(gw_clean) >= 3:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=gw_clean.values,
                        name="생중량",
                        fillcolor="#4CAF50",
                        marker=dict(color="#4CAF50"),
                        boxpoints="outliers",
                    ))
                    fig_box.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        title="생중량 분포 (박스플롯) — 상자 안 선이 중앙값",
                        yaxis_title="생중량 (g)",
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("생중량 데이터 수가 적어 박스플롯을 그리지 못했습니다.")
            else:
                st.info("생중량(g) 열이 없어 박스플롯을 그릴 수 없습니다.")

            # ── 상관관계 분석 (산점도 2개 + 추세선) ────────────────────
            st.subheader("🔗 상관관계 분석")
            corr_pairs = []
            if gw_col in growth_df.columns:
                if leaf_col in growth_df.columns:
                    corr_pairs.append((leaf_col, gw_col, "잎 수", "생중량 (g)"))
                if above_col in growth_df.columns:
                    corr_pairs.append((above_col, gw_col, "지상부 길이", "생중량 (g)"))

            if corr_pairs:
                for x_col, y_col, x_lbl, y_lbl in corr_pairs:
                    x = growth_df[x_col]
                    y = growth_df[y_col]
                    fig_corr = scatter_with_trend(x, y, x_lbl, y_lbl, color="#2196F3")
                    if fig_corr is not None:
                        st.plotly_chart(fig_corr, width="stretch")
                    else:
                        st.info(f"{x_lbl} vs {y_lbl} — 데이터가 부족해 상관분석을 건너뛰었습니다.")
            else:
                st.info("생중량과 비교할 잎 수·지상부 길이 열이 없습니다.")

            # ── 분산/변동계수 정보 카드 ─────────────────────────────────
            st.subheader("📐 개체별 차이 지표")
            if gw_col in growth_df.columns:
                gw_clean = growth_df[gw_col].dropna()
                if len(gw_clean) >= 2:
                    mean_v = gw_clean.mean()
                    std_v = gw_clean.std()
                    cv_v = (std_v / mean_v * 100) if mean_v != 0 else 0
                    info_box = st.container(border=True)
                    info_box.markdown(f"""
**생중량 변동 요약**
| 지표 | 값 | 의미 |
|------|----|------|
| 평균 | {mean_v:.3f} g | 개체들의 대표 생중량 |
| 표준편차 | {std_v:.3f} g | 평균에서 얼마나 퍼져 있는지 |
| 변동계수(CV) | {cv_v:.1f}% | 단위 없는 비교용 퍼진 정도 |
| 범위 | {gw_clean.min():.3f} ~ {gw_clean.max():.3f} g | 가장 작은 개체와 큰 개체 차이 |
""")
                else:
                    st.info("생중량 데이터 수가 적어 변동계수를 계산하지 못했습니다.")
            else:
                st.info("생중량(g) 열이 없어 변동 요약표를 표시할 수 없습니다.")

            # ── Expander: 생육 원본 + CSV 다운로드 ─────────────────────
            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                if not growth_df.empty:
                    disp_g = growth_df.copy()
                    if "개체번호" in disp_g.columns:
                        # 개체번호를 원래 값 그대로 표시 (소수점 제거)
                        disp_g["개체번호"] = disp_g["개체번호"].apply(
                            lambda v: int(v) if isinstance(v, float) and v == int(v) else v
                        )
                    st.dataframe(disp_g, width="stretch", hide_index=True)
                    csv_bytes_g = disp_g.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇ 생육 결과 CSV 다운로드",
                        data=csv_bytes_g,
                        file_name=f"{school}_생육결과데이터.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
