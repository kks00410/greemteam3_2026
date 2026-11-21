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

# ── 색상/스타일 상수 ─────────────────────────────────────────────────
BG_PINK   = "rgba(255,245,250,0.95)"
BG_BEIGE  = "rgba(255,250,245,0.95)"
FONTC     = "#000000"
TFONTC    = "#111111"
GRIDC     = "#E8E0DA"
ZEROC     = "#CCCCCC"
HOVBG     = "#FFF8F0"
INK       = "rgba(0,0,0,0.08)"   # 결측 셀 배경

# ── 헬퍼: hline + 주석 ────────────────────────────────────────────────
def add_hline_annot(fig, y, color, label, size=12):
    fig.add_hline(y=y, line_dash="dash", line_color=color, line_width=2)
    fig.add_annotation(
        x=1.0, y=y, xref="paper", yref="y",
        text=label, xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color=color, size=size),
    )

# ── 공통 레이아웃 ─────────────────────────────────────────────────────
def base_layout(fig, title, x_title, y_title, bg=BG_PINK):
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
            color=FONTC, gridcolor=GRIDC, zerolinecolor=ZEROC,
            tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                          color=FONTC, size=12),
        ),
        yaxis=dict(
            title=dict(text=y_title,
                       font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                 color=FONTC, size=13)),
            color=FONTC, gridcolor=GRIDC, zerolinecolor=ZEROC,
            tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                          color=FONTC, size=12),
        ),
        margin=dict(l=70, r=30, t=65, b=60),
        hovermode="closest",
    )
    return fig

# ── 개체별 편차 막대차트 (평균 대비 위/아래, 색 진하기 = 편차 크기) ─
def dev_bar_chart(values, mean_val, title, y_label, color_low="#AED6F1",
                   color_high="#E74C3C"):
    """values: 개체별 값 배열. 평균 이하=파랑, 초과=빨강.
       색 진하기 = |값-평균|이 클수록 진해짐."""
    vals = np.array(values, dtype=float)
    dev = np.abs(vals - mean_val)
    if dev.max() == 0:
        dev_norm = np.zeros_like(dev)
    else:
        dev_norm = dev / dev.max()   # 0~1 정규화

    colors = []
    for v, dn in zip(vals, dev_norm):
        base = color_low if v <= mean_val else color_high
        # 색이 진할수록 편차 큼 — 불투명도로 표현
        alpha = 0.25 + 0.70 * dn
        colors.append(base if isinstance(base, str) else base)
    # plotly는 rgba 문자열 직접 지원
    rgba_colors = []
    for v, dn in zip(vals, dev_norm):
        base_rgb = (174, 214, 241) if v <= mean_val else (231, 76, 60)
        alpha = 0.30 + 0.68 * dn
        rgba_colors.append(f"rgba({base_rgb[0]},{base_rgb[1]},{base_rgb[2]},{alpha:.2f})")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(vals) + 1)),
        y=vals,
        marker=dict(color=rgba_colors, line=dict(width=0.6, color="white"),
                    opacity=0.95),
        name="개체별 값",
        hovertemplate="%{x}번 개체<br>%{y:.4f}<extra></extra>",
    ))
    add_hline_annot(fig, mean_val, "#333333",
                     f"평균 = {mean_val:.4f}", size=12)
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                  color=FONTC),
        plot_bgcolor=BG_BEIGE,
        paper_bgcolor=BG_BEIGE,
        hoverlabel=dict(font=dict(color=FONTC), bgcolor=HOVBG,
                        bordercolor="#CCCCCC"),
        title=dict(text=title,
                   font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                             color=TFONTC, size=14),
                   x=0.03, xanchor="left"),
        xaxis=dict(
            title="개체 순번",
            color=FONTC, gridcolor=GRIDC, zerolinecolor=ZEROC,
            tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                          color=FONTC, size=11),
        ),
        yaxis=dict(
            title=y_label,
            color=FONTC, gridcolor=GRIDC, zerolinecolor=ZEROC,
            tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                          color=FONTC, size=12),
        ),
        margin=dict(l=60, r=30, t=60, b=55),
        hovermode="closest",
    )
    return fig

# ── 히트맵 스타일 개체별 표 (결측 확인 + 평균 대비 편차 색) ──────────
def individual_heatmap_table(df, measure_cols, mean_map):
    """df: 개체 데이터, measure_cols: 표시할 수치 열 목록,
       mean_map: 각 열의 평균 dict.
       반환: st.dataframe에 넣을 스타일 DataFrame."""
    disp = df[measure_cols].copy()
    # 열별 색 진하기 사전 계산
    records = []
    for i, (idx, row) in enumerate(disp.iterrows()):
        rec = {"개체": i + 1}
        for col in measure_cols:
            val = row[col]
            if pd.isna(val):
                rec[col] = "결측"
            else:
                rec[col] = round(float(val), 4)
        records.append(rec)
    table_df = pd.DataFrame(records)

    # 배경색 생성: 결측=회색, 값 있으면 평균 대비 편차 클수록 진한 색
    # plotly heatmap이 아니라 스트림릿 데이터프레임 스타일 적용
    # → st.dataframe의 style 사용법 대신 HTML로 직접 스타일링
    # → 여기선簡單히 st.dataframe + 별도 스 Calc 시각화 선택
    return table_df

# ════════════════════════════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════════════════════════════
st.sidebar.title("📁 데이터 업로드")
env_file    = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
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
tab1, tab2, tab3 = st.tabs(
    ["📋 실험 소개", "📊 생육 데이터 값", "✅ 결과 및 가설 확인"]
)

# ────────────────────────────────────────────────────────────────────────
#  Tab 1 — 실험 소개
# ────────────────────────────────────────────────────────────────────────
with tab1:
    try:
        st.header("📋 실험 소개")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("""
**연구 목적**
극지식물 81개체 각각이 얼마나 다르게 자라는지,
특히 **지하부(뿌리) 질량**과 **잎의 개수** 사이에
어떤 관계가 있는지 알아보는 실험입니다.

**개체 수**
- 업로드한 생육 CSV의 행 수 = 개체 수입니다.
- 아래 데이터 요약에서 자동으로 표시됩니다.
""")
        with c2:
            st.markdown("""
**연구 가설**
> "지하부 질량이 무거울수록(뿌리가 더 많이 자랄수록) 잎의 개수가 많을 것이다."

뿌리가 잘 발달한 개체일수록 물과 양분을 더 많이 흡수하므로
잎도 더 많이 낼 것이라는 생각에서 출발한 가설입니다.

**이 대시보드에서 확인하는 것**
1. 81개체 각각의 측정값을 한눈에 보기 (결측 확인)
2. 각 개체가 전체 평균에서 얼마나 떨어져 있는지 — 색 진하기로 표시
3. 지하부 질량과 잎 수의 관계를 산점도 + 상관계수로 확인
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
                st.dataframe(
                    pd.DataFrame(rows, columns=["데이터", "규모", "열"]),
                    width="stretch", hide_index=True,
                )
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
                st.info(
                    f"지하부 질량({below_col})과 잎 수({leaf_col}) 열이 모두 필요합니다. "
                    "업로드한 CSV에 해당 열이 있는지 확인해주세요."
                )
            else:
                # ── 결측 포함 전체 개체 데이터 준비 ────────────────────
                gw = growth_df[below_col].dropna()
                lf = growth_df[leaf_col].dropna()
                common = gw.index.intersection(lf.index)
                gw_c = gw.loc[common]
                lf_c = lf.loc[common]

                # 결측 개체 확인 (common에 없는 인덱스)
                miss_bw = growth_df.index.difference(gw.index)
                miss_lf = growth_df.index.difference(lf.index)
                n_miss_bw = len(miss_bw)
                n_miss_lf = len(miss_lf)

                # 평균
                mean_b = gw_c.mean()
                mean_l = lf_c.mean()

                # ── 결측 및 데이터 개요 카드 ──────────────────────────
                st.subheader("📋 결측 확인 및 데이터 개요")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("전체 개체 수", f"{len(growth_df)}개체")
                d2.metric("분석 가능 개체",
                          f"{len(common)}개체",
                          delta=f"지하부질량·잎수 모두 있음")
                d3.metric("지하부 질량 결측", f"{n_miss_bw}개체")
                d4.metric("잎 수 결측", f"{n_miss_lf}개체")

                if n_miss_bw == 0 and n_miss_lf == 0:
                    st.info(
                        "✅ 지하부 질량과 잎 수 모두 81개체에서 결측이 없습니다. "
                        "모든 개체를 분석에 사용할 수 있습니다."
                    )
                else:
                    st.warning(
                        f"⚠️ 지하부 질량 결측 {n_miss_bw}개체, "
                        f"잎 수 결측 {n_miss_lf}개체가 있어 "
                        f"해당 개체는 분석에서 제외했습니다."
                    )

                # ── 81개체 전체 표 (색 진하기로 평균 대비 편차 표시) ──
                st.subheader("📊 81개체 측정값 한눈에 보기")

                st.caption(
                    "• 배경색 진하기는 **각 개체가 전체 평균에서 얼마나 떨어져 있는지**를 나타냅니다.\n"
                    "• 색이 진할수록 평균에서 많이 벗어난 '특이한' 개체입니다.\n"
                    "• 결측값은 '결측'으로 표시됩니다."
                )

                # 표 데이터 준비: 개체번호, 각 측정값, 결측 여부
                meas_cols = [c for c in
                             ["잎 수", "지상부 길이", "지하부 길이",
                              "생중량", "지상부 질량", "지하부 질량"]
                             if c in growth_df.columns]

                table_rows = []
                for i, (idx, row) in enumerate(growth_df.iterrows()):
                    rec = {"개체": row.get("개체번호", i + 1)}
                    for col in meas_cols:
                        v = row[col]
                        if pd.isna(v):
                            rec[f"{col}\n(결측)"] = True
                            rec[col] = "—"
                        else:
                            rec[f"{col}\n(결측)"] = False
                            rec[col] = round(float(v), 4)
                    # 지하부 질량·잎 수 평균 대비 편차 (색 지표를 위한 값)
                    bw_v = row.get(below_col)
                    lf_v = row.get(leaf_col)
                    if pd.notna(bw_v) and pd.notna(lf_v):
                        rec["지하부 편차"] = round(float(abs(bw_v - mean_b)), 4)
                        rec["잎 수 편차"]  = round(float(abs(lf_v - mean_l)), 4)
                    else:
                        rec["지하부 편차"] = None
                        rec["잎 수 편차"]  = None
                    table_rows.append(rec)

                table_df = pd.DataFrame(table_rows)

                # 결측 표시만 있는 열 삭제 (깨끗한 표 위해)
                clean_cols = [c for c in table_df.columns
                              if not c.endswith("(결측)")]
                display_df = table_df[clean_cols].copy()

                # 고유 개체번호 보존 (개체열을 정수로)
                if "개체" in display_df.columns:
                    display_df["개체"] = display_df["개체"].apply(
                        lambda v: int(v) if isinstance(v, float) and v == int(v) else v
                    )

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "개체": st.column_config.NumberColumn(
                            "개체번호", width="small"),
                        "지하부 편차": st.column_config.NumberColumn(
                            "지하부 편차\n(평균 0.4512g 기준)",
                            width="medium",
                            format="%.4f",
                        ),
                        "잎 수 편차": st.column_config.NumberColumn(
                            "잎 수 편차\n(평균 2.0장 기준)",
                            width="medium",
                            format="%.4f",
                        ),
                    },
                )

                # ── 결측 시각화: 개체별 결측 여부 막대 ────────────────
                st.subheader("🔲 개체별 결측 여부 확인")
                miss_info = []
                for i, (idx, row) in enumerate(growth_df.iterrows()):
                    row_miss = {}
                    for col in meas_cols:
                        row_miss[col] = pd.isna(row[col])
                    miss_info.append(row_miss)
                miss_df = pd.DataFrame(miss_info)
                miss_df.index = range(1, len(miss_df) + 1)

                # 결측인 열만 추출
                cols_with_miss = [c for c in meas_cols
                                  if miss_df[c].any()]
                if cols_with_miss:
                    fig_miss = go.Figure()
                    y_labels = [str(i) for i in miss_df.index]
                    for col in cols_with_miss:
                        miss_vals = miss_df[col].astype(int).values
                        fig_miss.add_trace(go.Scatter(
                            x=miss_vals, y=y_labels, mode="markers",
                            marker=dict(
                                symbol="x",
                                size=14,
                                color="#E74C3C",
                                line=dict(width=2, color="#C0392B"),
                            ),
                            name=f"{col} 결측",
                            hovertemplate="%{y}번 개체 — %{x} 결측<extra></extra>",
                        ))
                    fig_miss.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                  color=FONTC),
                        plot_bgcolor=BG_BEIGE,
                        paper_bgcolor=BG_BEIGE,
                        hoverlabel=dict(font=dict(color=FONTC), bgcolor=HOVBG,
                                        bordercolor="#CCCCCC"),
                        title=dict(
                            text="빨간 ✕ = 결측값 (해당 개체는 이 측정값이 없음)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                      color=TFONTC, size=14),
                            x=0.03, xanchor="left",
                        ),
                        xaxis=dict(title="결측 여부 (1=결측)",
                                   color=FONTC, gridcolor=GRIDC,
                                   zerolinecolor=ZEROC,
                                   tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                                 color=FONTC, size=11),
                                   tickvals=[0, 1],
                                   ticktext=["값 있음", "결측"]),
                        yaxis=dict(title="개체",
                                   color=FONTC, gridcolor=GRIDC,
                                   zerolinecolor=ZEROC,
                                   tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                                 color=FONTC, size=10),
                                   categoryorder="total ascending"),
                        margin=dict(l=60, r=30, t=60, b=55),
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_miss, width="stretch")
                else:
                    st.info("✅ 모든 개체의 모든 측정값에 결측이 없습니다.")

                # ── 지하부 질량 편차 막대차트 ──────────────────────────
                st.subheader("📊 개체별 지하부 질량 (색 진하기 = 평균 대비 편차)")
                st.caption(
                    "• 막대의 **색이 진할수록** 그 개체가 전체 평균에서 멀리 떨어져 있습니다.\n"
                    "• 파란 막대 = 평균 이하, 빨간 막대 = 평균 초과"
                )
                fig_bw = dev_bar_chart(
                    gw_c.values, mean_b,
                    f"지하부 질량 (평균 = {mean_b:.4f} g)",
                    "지하부 질량 (g)",
                )
                st.plotly_chart(fig_bw, width="stretch")

                # ── 잎 수 편차 막대차트 ────────────────────────────────
                st.subheader("📊 개체별 잎 수 (색 진하기 = 평균 대비 편차)")
                fig_lf = dev_bar_chart(
                    lf_c.values, mean_l,
                    f"잎 수 (평균 = {mean_l:.1f} 장)",
                    "잎 수 (장)",
                )
                st.plotly_chart(fig_lf, width="stretch")

                # ── 지하부 질량 vs 잎 수 산점도 ────────────────────────
                st.subheader("🔗 지하부 질량 vs 잎 수 (상관계수·추세선)")
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(
                    x=gw_c.values, y=lf_c.values,
                    mode="markers",
                    marker=dict(
                        color="#2E86C1", size=10, opacity=0.88,
                        line=dict(width=1.2, color="white"),
                    ),
                    name="데이터",
                    hovertemplate="지하부 %{x:.4f}g<br>잎 %{y:.1f}장<extra></extra>",
                ))
                if len(gw_c) >= 2:
                    slope, intercept = np.polyfit(gw_c.values, lf_c.values, 1)
                    pred_x = gw_c.values
                    pred_y = slope * pred_x + intercept
                    fig_corr.add_trace(go.Scatter(
                        x=pred_x, y=pred_y,
                        mode="lines",
                        line=dict(color="#C0392B", width=2.5),
                        name=f"추세선 (r={np.corrcoef(gw_c.values, lf_c.values)[0,1]:.3f})",
                    ))
                r_val = np.corrcoef(gw_c.values, lf_c.values)[0, 1]
                p_val = None
                try:
                    from scipy import stats
                    _, p_val = stats.pearsonr(gw_c.values, lf_c.values)
                except Exception:
                    pass
                fig_corr.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                              color=FONTC),
                    plot_bgcolor=BG_PINK,
                    paper_bgcolor=BG_PINK,
                    hoverlabel=dict(font=dict(color=FONTC), bgcolor=HOVBG,
                                    bordercolor="#CCCCCC"),
                    title=dict(
                        text=f"지하부 질량 vs 잎 수  (상관계수 r = {r_val:.3f}"
                             + (f", p = {p_val:.4f}" if p_val is not None else "")
                             + ")",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                  color=TFONTC, size=14),
                        x=0.03, xanchor="left",
                    ),
                    xaxis=dict(title="지하부 질량 (g)",
                               color=FONTC, gridcolor=GRIDC, zerolinecolor=ZEROC,
                               tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                             color=FONTC, size=12)),
                    yaxis=dict(title="잎 수 (장)",
                               color=FONTC, gridcolor=GRIDC, zerolinecolor=ZEROC,
                               tickfont=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
                                             color=FONTC, size=12)),
                    margin=dict(l=60, r=30, t=65, b=60),
                    hovermode="closest",
                )
                st.plotly_chart(fig_corr, width="stretch")

                # ── 상관계수 카드 ──────────────────────────────────────
                st.subheader("📐 상관계수 요약")
                rc, pc = st.columns(2)
                rc.metric("상관계수 r", f"{r_val:.4f}")
                if p_val is not None:
                    sig = ("✅ 통계적으로 유의" if p_val < 0.05
                           else "⚠️ 유의하지 않음 (p ≥ 0.05)")
                    pc.metric("p 값", f"{p_val:.4f}", delta=sig)
                else:
                    pc.metric("p 값", "계산 불가")
                st.info(
                    "상관계수 r은 -1 ~ +1 사이 값입니다.\n"
                    "+1에 가까울수록 두 값이 함께 증가(정방향),\n"
                    "-1에 가까울수록 한쪽 증가 시 다른 쪽 감소(역방향),\n"
                    "0에 가까우면 관계가 약합니다."
                )

        # ── Expander: 생육 원본 + CSV 다운로드 ────────────────────────
        with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
            if not growth_df.empty:
                disp = growth_df.copy()
                if "개체번호" in disp.columns:
                    disp["개체번호"] = disp["개체번호"].apply(
                        lambda v: int(v) if isinstance(v, float) and v == int(v)
                        else v
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
                st.info(
                    f"지하부 질량({below_col})과 잎 수({leaf_col}) 열이 모두 필요합니다."
                )
            else:
                gw = growth_df[below_col].dropna()
                lf = growth_df[leaf_col].dropna()
                common = gw.index.intersection(lf.index)
                gw_c = gw.loc[common]
                lf_c = lf.loc[common]

                if len(gw_c) < 2:
                    st.info("데이터가 부족해 가설 확인을 건너뛰었습니다.")
                else:
                    r_val = np.corrcoef(gw_c.values, lf_c.values)[0, 1]
                    p_val = None
                    try:
                        from scipy import stats
                        _, p_val = stats.pearsonr(gw_c.values, lf_c.values)
                    except Exception:
                        pass

                    # ── 가설 검토 ────────────────────────────────────────
                    st.subheader("🔬 가설 검토")
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        st.markdown("""
**연구 가설**
> "지하부 질량이 무거울수록 잎의 개수가 많을 것이다."
""")
                    with hc2:
                        if p_val is not None and p_val < 0.05:
                            direction = "정방향" if r_val > 0 else "역방향"
                            if r_val > 0:
                                verdict = (
                                    f"✅ 가설이 **지지됨** "
                                    f"(p < 0.05, r = {r_val:.3f}, {direction})"
                                )
                            else:
                                verdict = (
                                    f"❌ 가설이 **기각됨** — "
                                    f"오히려 역방향 관계 (r = {r_val:.3f})"
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

                    # ── 가설이 틀렸을 때 원인 ────────────────────────────
                    if r_val <= 0 or (p_val is not None and p_val >= 0.05):
                        st.subheader("🔍 가설이 맞지 않을 수 있는 원인")
                        st.markdown("""
지하부 질량과 잎 수 사이에 뚜렷한 정방향 관계가 나타나지 않은
이유는 다음과 같이 생각해볼 수 있습니다.

1. **지하부 질량이 잎 수를 직접 결정하지 않을 수 있음**
   잎의 수는 유전, 광량, 생장 단계, 호르몬 등 여러 요인의 영향을 받아요.
   뿌리 질량만으로 잎 수를 예측하기는 어렵습니다.

2. **개체 간 생장 단계가 다를 수 있음**
   아직 어린 개체는 잎이 적어도 뿌리가 먼저 자라고,
   성숙한 개체는 잎이 많지만 뿌리 성장이 둔화되었을 수 있어요.
   같은 시기에 측정했더라도 실제 발달 단계는 다를 수 있습니다.

3. **환경이 개체마다 균일하지 않았을 수 있음**
   같은 실험실 안에서도 위치에 따라 빛·온도·수분 조건이 조금씩
   다를 수 있고, 이것이 뿌리 성장과 잎 수에 서로 다른 영향을
   주었을 수 있습니다.

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
"두 특성이 함께 커지는 경향이 관찰되었다"고 해석하는 것이
 적절합니다.
""")

                    # ── 종합 결과 ───────────────────────────────────────
                    st.subheader("📋 종합 결과")
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        st.dataframe(
                            pd.DataFrame({
                                "지표": ["개체 수",
                                         "지하부 질량 평균 (g)",
                                         "잎 수평균 (장)",
                                         "상관계수 r"],
                                "값": [str(len(gw_c)),
                                       f"{gw_c.mean():.4f}",
                                       f"{lf_c.mean():.1f}",
                                       f"{r_val:.4f}"],
                            }),
                            width="stretch", hide_index=True,
                        )
                    with tc2:
                        stance = ("가설 지지"
                                  if (r_val > 0 and p_val is not None
                                      and p_val < 0.05)
                                  else "가설 지지 불명확 또는 기각")
                        direction_note = ("(정방향)"
                                          if r_val > 0 else "(역방향 또는 무관)")
                        st.markdown(f"""
**한 줄 요약**
- 총 {len(gw_c)}개체를 분석한 결과,
- 지하부 질량과 잎 수의 상관계수는 **r = {r_val:.3f}** {direction_note}
- {stance}
""")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
