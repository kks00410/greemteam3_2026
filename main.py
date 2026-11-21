
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============ 한글 폰트 설정 ============
def setup_korean_font():
    """Streamlit Cloud에서도 한글 깨짐 없이 렌더링되도록 폰트 설정"""
    import platform
    system = platform.system()

    if system == "Linux":  # Streamlit Cloud 환경
        possible_fonts = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
            '/usr/share/fonts/opentype/nanum/NanumGothic.otf',
            '/usr/share/fonts/truetype/naver/D2Coding.ttf',
        ]
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                matplotlib.rc('font', family='NanumGothic')
                matplotlib.rc('axes', unicode_minus=False)
                return
        matplotlib.rc('font', family='sans-serif')
        matplotlib.rc('axes', unicode_minus=False)
    else:
        matplotlib.rc('font', family='NanumGothic')
        matplotlib.rc('axes', unicode_minus=False)

setup_korean_font()

# ============ 데이터 로드 유틸리티 ============
def load_csv_with_encoding(uploaded_file):
    """
    인코딩 오류를 방지하기 위해 여러 인코딩을 시도합니다.
    - utf-8-sig (BOM 포함 UTF-8)
    - cp949 (윈도우 한국어)
    - utf-8 (일반 UTF-8)
    """
    encodings = ['utf-8-sig', 'cp949', 'utf-8', 'euc-kr']

    for enc in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=enc)
            return df
        except (UnicodeDecodeError, Exception):
            continue

    return pd.read_csv(uploaded_file, encoding='latin-1', low_memory=False)


def normalize_env_columns(df):
    """
    학교마다 다른 환경 데이터 열 이름을 표준화합니다.

    형식 A: time, temperature, humidity, ph, ec, water_temp, co2
    형식 B: categories, 온도, 습도, EC, PH

    반환: time, temperature, humidity, ph, ec, water_temp, co2 중 존재하는 열만 포함
    """
    col_mapping = {
        'time': ['time', '날짜', 'datetime', 'timestamp', '시간', 'categories'],
        'temperature': ['temperature', '온도', 'temp', 'temper'],
        'humidity': ['humidity', '습도', 'humid'],
        'ph': ['ph', 'PH', '수소이온농도', '산도'],
        'ec': ['ec', 'EC', '전기전도도', 'ec_value'],
        'water_temp': ['water_temp', '수온', 'watertemperature', '물온도'],
        'co2': ['co2', 'CO2', '이산화탄소'],
    }

    normalized_cols = {}

    for std_name, possible_names in col_mapping.items():
        for col in df.columns:
            if col in possible_names:
                normalized_cols[std_name] = col
                break

    result = pd.DataFrame()
    for std_name, orig_col in normalized_cols.items():
        result[std_name] = df[orig_col]

    return result, normalized_cols


def normalize_growth_columns(df):
    """
    학교마다 다른 생육 결과 데이터 열 이름을 표준화합니다.

    표준 열: 개체번호, 잎_수, 지상부_길이, 지하부_길이, 생중량, 지상부_질량, 지하부_질량
    """
    col_mapping = {
        '개체번호': ['개체번호', '개체', 'id', 'ID', '번호', 'sample_id'],
        '잎_수': ['잎 수(장)', '잎수', '잎 수', 'leaf_count', 'leaves'],
        '지상부_길이': ['지상부 길이(mm)', '지상부길이', '지상부 길이', '지상부길이(mm)', 'shoot_length', 'aboveground_length'],
        '지하부_길이': ['지하부길이(mm)', '지하부 길이', '지하부길이', '지하부 길이(mm)', 'root_length', 'underground_length'],
        '생중량': ['생중량(g)', '생중량', 'fresh_weight', 'total_weight'],
        '지상부_질량': ['지상부질량(g)', '지상부 질량', '지상부질량', '지상부질량(g)', 'shoot_mass', 'aboveground_mass'],
        '지하부_질량': ['지하부질량(g)', '지하부 질량', '지하부질량', '지하부질량(g)', 'root_mass', 'underground_mass'],
    }

    normalized_cols = {}

    for std_name, possible_names in col_mapping.items():
        for col in df.columns:
            if col in possible_names:
                normalized_cols[std_name] = col
                break

    result = pd.DataFrame()
    for std_name, orig_col in normalized_cols.items():
        result[std_name] = df[orig_col]

    return result, normalized_cols


# ============ 통계 용어 설명 ============
def explain_stat_term(term):
    """고등학생이 이해할 수 있도록 통계 용어 설명"""
    explanations = {
        '평균': '모든 값을 더한 후 개수로 나눈 값 (대표값)',
        '표준편차': '값들이 평균에서 얼마나 퍼져있는지 나타내는 지표 (클수록 다양함)',
        'CV(변동계수)': '표준편차를 평균으로 나눈 값 (단위가 다른 데이터도 비교 가능)',
        '상관계수': '-1에서 1 사이 값, 1에 가까울수록 정비례 관계 (0에 가까우면 관계 약함)',
    }
    return explanations.get(term, '')


# ============ Streamlit 앱 ============
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# 사이드바
st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 (CSV)", type=["csv"], key="env_upload")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 (CSV)", type=["csv"], key="growth_upload")

st.sidebar.markdown("---")
st.sidebar.header("📊 생육 측정 항목 선택")
selected_metrics = st.sidebar.multiselect(
    "분석할 항목",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    help="체크된 항목만 탭 3에서 분석합니다"
)

# 기본 데이터 상태
env_data = None
growth_data = None
env_normalized = None
growth_normalized = None

# 데이터 처리
if env_file is not None:
    try:
        df_env_raw = load_csv_with_encoding(env_file)
        env_normalized, env_col_map = normalize_env_columns(df_env_raw)
        if 'time' in env_normalized.columns:
            env_normalized['time'] = pd.to_datetime(env_normalized['time'], errors='coerce')
        env_data = env_normalized
    except Exception as e:
        st.sidebar.error(f"환경 데이터 로딩 오류: {str(e)}")

if growth_file is not None:
    try:
        df_growth_raw = load_csv_with_encoding(growth_file)
        growth_normalized, growth_col_map = normalize_growth_columns(df_growth_raw)
        growth_data = growth_normalized
    except Exception as e:
        st.sidebar.error(f"생육 데이터 로딩 오류: {str(e)}")

# ============ 탭 구성 ============
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ----------------------------------------
# TAB 1: 실험 개요
# ----------------------------------------
with tab1:
    st.header("🔬 연구 배경 및 목적")

    st.markdown("""
    ### 배경
    극지식물은 혹독한 환경에서도 생존할 수 있는 특별한 적응 능력을 가진 식물입니다. 
    이 실험에서는 다양한 환경 조건(온도, 습도, pH, EC, CO₂ 등)에서 극지식물이 어떻게 자라는지 관찰합니다.

    ### 목적
    - **개체별 생육 차이 분석**: 같은 조건에서 자란 개체들 사이에 왜 차이가 나는지 이해합니다.
    - **환경 요인과의 관계 파악**: 온도, 습도 등 환경 요인이 생육에 어떤 영향을 미치는지 살펴봅니다.
    - **통계 개념 학습**: 평균, 표준편차, 상관관계 등 기본 통계 개념을 실제 데이터로 학습합니다.
    """)

    if env_data is not None and growth_data is not None:
        st.header("📋 데이터 요약")

        n_individuals = len(growth_data)

        if 'time' in env_data.columns:
            time_min = env_data['time'].min()
            time_max = env_data['time'].max()
            period_text = f"{time_min.strftime('%Y-%m-%d')} ~ {time_max.strftime('%Y-%m-%d')}"
        else:
            period_text = "정보 없음"

        env_items = [col for col in env_data.columns if col != 'time']
        growth_items = ['잎_수', '지상부_길이', '지하부_길이', '생중량', '지상부_질량', '지하부_질량']
        growth_items_present = [item for item in growth_items if item in growth_data.columns]

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("총 개체 수", f"{n_individuals}개체")
        with summary_col2:
            st.metric("측정 기간", period_text)
        with summary_col3:
            st.metric("환경 측정 항목", f"{len(env_items)}개")

        st.markdown("---")

        st.subheader("📊 주요 지표")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("총 개체수", n_individuals, help="생육 데이터가 있는 개체의 총 개수")

        with col2:
            if 'temperature' in env_data.columns:
                avg_temp = env_data['temperature'].mean()
                st.metric("평균 온도", f"{avg_temp:.1f}°C", help="전체 기간 평균 온도")
            else:
                st.metric("평균 온도", "데이터 없음")

        with col3:
            if 'humidity' in env_data.columns:
                avg_hum = env_data['humidity'].mean()
                st.metric("평균 습도", f"{avg_hum:.1f}%", help="전체 기간 평균 습도")
            else:
                st.metric("평균 습도", "데이터 없음")

        with col4:
            if '생중량' in growth_data.columns:
                avg_fw = growth_data['생중량'].mean()
                st.metric("평균 생중량", f"{avg_fw:.2f}g", help="개체별 생중량 평균")
            elif '지상부_질량' in growth_data.columns and '지하부_질량' in growth_data.columns:
                growth_data = growth_data.copy()
                growth_data['생중량'] = growth_data['지상부_질량'] + growth_data['지하부_질량']
                avg_fw = growth_data['생중량'].mean()
                st.metric("평균 생중량", f"{avg_fw:.2f}g", help="지상부+지하부 질량 합계 평균")
            else:
                st.metric("평균 생중량", "데이터 없음")

        st.markdown("---")

        st.subheader("📚 통계 용어 간단 설명")
        terms_to_explain = ['평균', '표준편차', 'CV(변동계수)', '상관계수']
        for term in terms_to_explain:
            st.info(f"**{term}**: {explain_stat_term(term)}")
    else:
        st.info("📁 사이드바에서 CSV 파일을 업로드하면 데이터가 표시됩니다.")

# ----------------------------------------
# TAB 2: 환경 데이터
# ----------------------------------------
with tab2:
    if env_data is not None and len(env_data) > 0:
        st.header("🌡️ 환경 데이터 분석")

        st.subheader("📈 환경 요약 통계")

        numeric_env_cols = [col for col in env_data.columns if col != 'time']

        if numeric_env_cols:
            env_summary = env_data[numeric_env_cols].agg(['mean', 'min', 'max', 'std']).round(2)
            env_summary.columns = ['평균', '최소값', '최대값', '표준편차']

            col_names_with_unit = []
            for col in numeric_env_cols:
                if col == 'temperature':
                    col_names_with_unit.append('온도 (°C)')
                elif col == 'humidity':
                    col_names_with_unit.append('습도 (%)')
                elif col == 'ph':
                    col_names_with_unit.append('pH')
                elif col == 'ec':
                    col_names_with_unit.append('EC (mS/cm)')
                elif col == 'water_temp':
                    col_names_with_unit.append('수온 (°C)')
                elif col == 'co2':
                    col_names_with_unit.append('CO₂ (ppm)')
                else:
                    col_names_with_unit.append(col)

            summary_df = env_summary.copy()
            summary_df.index = col_names_with_unit
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("환경 데이터가 없습니다.")

        st.subheader("📅 날짜별 환경 변화 (일별 평균)")

        if 'time' in env_data.columns:
            env_data_copy = env_data.copy()
            env_data_copy['date'] = env_data_copy['time'].dt.date

            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                if 'temperature' in env_data_copy.columns:
                    st.markdown("#### 온도 변화")
                    daily_temp = env_data_copy.groupby('date')['temperature'].mean()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(daily_temp.index, daily_temp.values, marker='o', linewidth=2, color='#E74C3C')
                    ax.set_xlabel('')
                    ax.set_ylabel('온도 (°C)')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            with plot_col2:
                if 'humidity' in env_data_copy.columns:
                    st.markdown("#### 습도 변화")
                    daily_hum = env_data_copy.groupby('date')['humidity'].mean()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(daily_hum.index, daily_hum.values, marker='s', linewidth=2, color='#3498DB')
                    ax.set_xlabel('')
                    ax.set_ylabel('습도 (%)')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            if 'ph' in env_data_copy.columns or 'ec' in env_data_copy.columns:
                st.markdown("#### pH · EC 변화")
                plot_col3, plot_col4 = st.columns(2)

                with plot_col3:
                    if 'ph' in env_data_copy.columns:
                        daily_ph = env_data_copy.groupby('date')['ph'].mean()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(daily_ph.index, daily_ph.values, marker='^', linewidth=2, color='#27AE60')
                        ax.set_xlabel('')
                        ax.set_ylabel('pH')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(0, 14)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                with plot_col4:
                    if 'ec' in env_data_copy.columns:
                        daily_ec = env_data_copy.groupby('date')['ec'].mean()
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(daily_ec.index, daily_ec.values, marker='d', linewidth=2, color='#8E44AD')
                        ax.set_xlabel('')
                        ax.set_ylabel('EC (mS/cm)')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            if 'co2' in env_data_copy.columns:
                st.markdown("#### CO₂ 변화")
                daily_co2 = env_data_copy.groupby('date')['co2'].mean()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(daily_co2.index, daily_co2.values, marker='o', linewidth=2, color='#F39C12')
                ax.set_xlabel('')
                ax.set_ylabel('CO₂ (ppm)')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("시간 정보가 없어 시계열 그래프를 그릴 수 없습니다.")

        with st.expander("📄 환경 데이터 원본 테이블 보기"):
            st.dataframe(env_data, use_container_width=True)

            csv_buffer = BytesIO()
            env_data.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv_buffer.getvalue(),
                file_name="환경데이터_다운로드.csv",
                mime="text/csv"
            )
    else:
        st.info("📁 사이드바에서 환경 데이터 CSV를 업로드하면 분석 결과가 표시됩니다.")

# ----------------------------------------
# TAB 3: 생육 결과
# ----------------------------------------
with tab3:
    if growth_data is not None and len(growth_data) > 0:
        st.header("📊 생육 결과 분석")

        # 생중량이 없으면 계산
        if '생중량' not in growth_data.columns:
            if '지상부_질량' in growth_data.columns and '지하부_질량' in growth_data.columns:
                growth_data = growth_data.copy()
                growth_data['생중량'] = growth_data['지상부_질량'] + growth_data['지하부_질량']
                st.success("⚠️ 생중량(g) 열이 없어 지상부 질량 + 지하부 질량 합계로 계산했습니다.")
            else:
                st.warning("생중량 데이터가 없습니다.")

        # 핵심 결과 카드
        st.subheader("🥇 핵심 결과")

        if '생중량' in growth_data.columns:
            avg_fw = growth_data['생중량'].mean()
            max_fw = growth_data['생중량'].max()
            min_fw = growth_data['생중량'].min()
            max_idx = growth_data['생중량'].idxmax()
            min_idx = growth_data['생중량'].idxmin()

            if '개체번호' in growth_data.columns:
                max_individual = growth_data.loc[max_idx, '개체번호']
                min_individual = growth_data.loc[min_idx, '개체번호']
            else:
                max_individual = max_idx + 1
                min_individual = min_idx + 1

            card1, card2, card3 = st.columns(3)
            with card1:
                st.metric("평균 생중량", f"{avg_fw:.2f}g")
            with card2:
                st.metric("최대 생중량", f"{max_fw:.2f}g", help=f"개체 {max_individual}")
            with card3:
                st.metric("최소 생중량", f"{min_fw:.2f}g", help=f"개체 {min_individual}")

        # 개체별 생육 비교 (막대 그래프)
        st.subheader("📊 개체별 생육 비교")

        metric_to_col = {
            '생중량': '생중량',
            '잎 수': '잎_수',
            '지상부 길이': '지상부_길이',
            '지하부 길이': '지하부_길이'
        }

        selected_cols_for_plot = []
        selected_labels = []
        for metric in selected_metrics:
            if metric in metric_to_col:
                col = metric_to_col[metric]
                if col in growth_data.columns:
                    selected_cols_for_plot.append(col)
                    if metric == '생중량':
                        selected_labels.append('생중량 (g)')
                    elif metric == '잎 수':
                        selected_labels.append('잎 수 (장)')
                    elif metric == '지상부 길이':
                        selected_labels.append('지상부 길이 (mm)')
                    elif metric == '지하부 길이':
                        selected_labels.append('지하부 길이 (mm)')

        if selected_cols_for_plot:
            n_cols = min(len(selected_cols_for_plot), 4)
            n_rows = (len(selected_cols_for_plot) + n_cols - 1) // n_cols

            for row_idx in range(n_rows):
                cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    data_idx = row_idx * n_cols + col_idx
                    if data_idx < len(selected_cols_for_plot):
                        with cols[col_idx]:
                            col_name = selected_cols_for_plot[data_idx]
                            label = selected_labels[data_idx]

                            st.markdown(f"#### {label}")

                            fig, ax = plt.subplots(figsize=(8, 4))
                            values = growth_data[col_name].values

                            if '개체번호' in growth_data.columns:
                                x_labels = growth_data['개체번호'].astype(str).values
                            else:
                                x_labels = [str(i+1) for i in range(len(growth_data))]

                            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
                            ax.bar(range(len(values)), values, color=colors, edgecolor='white')
                            ax.set_xlabel('개체')
                            ax.set_ylabel(label)
                            ax.set_xticks(range(0, len(values), max(1, len(values)//20)))
                            ax.set_xticklabels([str(i+1) for i in range(0, len(values), max(1, len(values)//20))], rotation=45)
                            ax.grid(True, alpha=0.3, axis='y')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    else:
                        with cols[col_idx]:
                            st.empty()
        else:
            st.warning("선택된 측정 항목이 없습니다. 사이드바에서 항목을 선택해주세요.")

        # 생중량 분포 (박스플롯)
        st.subheader("📦 생중량 분포")

        if '생중량' in growth_data.columns:
            col_dist1, col_dist2 = st.columns(2)

            with col_dist1:
                st.markdown("#### 박스플롯 (Box Plot)")
                fig, ax = plt.subplots(figsize=(8, 5))
                bp = ax.boxplot(growth_data['생중량'].values, vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor('#3498DB')
                ax.set_ylabel('생중량 (g)')
                ax.set_xticklabels(['전체'])
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                fw_values = growth_data['생중량'].dropna()
                mean_val = fw_values.mean()
                std_val = fw_values.std()
                cv_val = (std_val / mean_val * 100) if mean_val > 0 else 0

                st.markdown(f"""
                **분포 통계:**
                - 평균: {mean_val:.2f}g
                - 표준편차: {std_val:.2f}g ({explain_stat_term('표준편차')})
                - CV(변동계수): {cv_val:.1f}% ({explain_stat_term('CV(변동계수)')})
                """)

            with col_dist2:
                st.markdown("#### 바이올린 플롯 (Violin Plot)")
                fig, ax = plt.subplots(figsize=(8, 5))
                parts = ax.violinplot(growth_data['생중량'].dropna().values, vert=True, showmeans=True, showmedians=True)

                for partname in ('cbars', 'cbody', 'cquantiles'):
                    vp = parts[partname]
                    vp.set_facecolor('#E74C3C')
                    vp.set_edgecolor('black')
                    vp.set_alpha(0.7)

                ax.set_ylabel('생중량 (g)')
                ax.set_xticklabels(['전체'])
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with st.expander("📋 생중량 상세 통계"):
                fw_stats = pd.DataFrame({
                    '통계항목': ['평균', '표준편차', '최소값', '최대값', '중앙값', '사분위수(Q1)', '사분위수(Q3)'],
                    '값': [
                        fw_values.mean(),
                        fw_values.std(),
                        fw_values.min(),
                        fw_values.max(),
                        fw_values.median(),
                        fw_values.quantile(0.25),
                        fw_values.quantile(0.75)
                    ]
                }).round(3)
                st.dataframe(fw_stats, use_container_width=True)

        # 상관관계 분석
        st.subheader("🔗 상관관계 분석")

        correlation_pairs = []

        if '잎_수' in growth_data.columns and '생중량' in growth_data.columns:
            correlation_pairs.append(('잎_수', '잎 수', '생중량'))

        if '지상부_길이' in growth_data.columns and '생중량' in growth_data.columns:
            correlation_pairs.append(('지상부_길이', '지상부 길이', '생중량'))

        for x_col, x_label, y_col in correlation_pairs:
            if '개체번호' in growth_data.columns:
                x_vals = growth_data[x_col].values
                y_vals = growth_data[y_col].values

                fig, ax = plt.subplots(figsize=(8, 5))

                ax.scatter(x_vals, y_vals, alpha=0.7, s=60, color='#3498DB', edgecolors='white')

                if len(x_vals) > 1:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(x_line, p(x_line), '--', color='#E74C3C', linewidth=2, label='추세선')

                    r_value = np.corrcoef(x_vals, y_vals)[0, 1]
                    ax.set_title(f'{x_label} vs 생중량 (상관계수 r = {r_value:.3f})', fontsize=12)
                    ax.legend()

                ax.set_xlabel(x_label)
                ax.set_ylabel('생중량 (g)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.info(f"""
                **상관계수 (r) = {np.corrcoef(x_vals, y_vals)[0, 1]:.3f}**
                {explain_stat_term('상관계수')}

                - r > 0.7: 강한 양의 상관관계
                - 0.3 < r < 0.7: 중간 상관관계
                - r < 0.3: 약한 상관관계
                """)

        if len(correlation_pairs) == 0:
            st.warning("상관관계 분석을 위해 '잎 수'와 '생중량' 또는 '지상부 길이'와 '생중량' 열이 필요합니다.")

        with st.expander("📄 생육 데이터 원본 테이블 보기"):
            st.dataframe(growth_data, use_container_width=True)

            csv_buffer = BytesIO()
            growth_data.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv_buffer.getvalue(),
                file_name="생육결과데이터_다운로드.csv",
                mime="text/csv"
            )
    else:
        st.info("📁 사이드바에서 생육 결과 데이터 CSV를 업로드하면 분석 결과가 표시됩니다.")

# ============ 푸터 ============
st.markdown("---")
st.markdown("""
**참고**: 
- 이 앱은 Streamlit Cloud를 포함한 다양한 환경에서 실행되도록 설계되었습니다.
- 인코딩 문제가 있으면 CSV 파일을 다시 업로드해보세요 (UTF-8 또는 CP949 인코딩 사용).
- 각 학교의 열 이름이 다르더라도 자동으로 표준화하여 분석합니다.
""")
