
from pathlib import Path
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st


# 페이지 설정
st.set_page_config(
    page_title="지역별 발전량 비교실",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 지역별 발전량 비교실")
st.write(
    "Adelaide, Perth, Sydney, Tasmania 지역의 파력 발전 데이터를 비교하여 "
    "지역별 발전량의 특징을 살펴봅니다."
)

# 데이터 파일 경로
APP_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = APP_DIR / "wave+energy+converters.zip"

if not ZIP_PATH.exists():
    st.error(f"파일을 찾을 수 없습니다: {ZIP_PATH}")
    st.stop()

# 데이터 컬럼명
FEATURE_COLS = (
    [f"X{i}" for i in range(1, 17)]
    + [f"Y{i}" for i in range(1, 17)]
)

POWER_COLS = [f"P{i}" for i in range(1, 17)]

DATA_COLS = FEATURE_COLS + POWER_COLS + ["Total Power"]

# 지역명 한글 변환
REGION_NAMES = {
    "Adelaide": "애들레이드",
    "Perth": "퍼스",
    "Sydney": "시드니",
    "Tasmania": "태즈메이니아"
}


# 데이터 불러오기
@st.cache_data
def load_data(zip_path):
    region_data = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():
            if not file_name.lower().endswith(".csv"):
                continue

            file_path = Path(file_name)
            region = file_path.stem

            if region not in REGION_NAMES:
                continue

            with zip_file.open(file_name) as file:
                df = pd.read_csv(
                    file,
                    header=None,
                    names=DATA_COLS
                )

            df["지역"] = REGION_NAMES[region]

            region_data.append(df)

    if not region_data:
        raise ValueError("ZIP 파일에서 지역별 CSV 데이터를 찾을 수 없습니다.")

    return pd.concat(region_data, ignore_index=True)


# 파일 확인 및 데이터 불러오기
if not ZIP_PATH.exists():
    st.error("wave+energy+converters.zip 파일을 프로젝트 최상위 폴더에 넣어 주세요.")
    st.stop()

try:
    data = load_data(str(ZIP_PATH))
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# 지역별 요약 데이터 계산
region_order = ["애들레이드", "퍼스", "시드니", "태즈메이니아"]

summary = (
    data.groupby("지역")["Total Power"]
    .agg(
        데이터_개수="count",
        평균_발전량="mean",
        최대_발전량="max",
        최소_발전량="min"
    )
    .reindex(region_order)
    .reset_index()
)

# 상단 요약
st.subheader("지역별 데이터 개수")

cols = st.columns(4)

for col, region in zip(cols, region_order):
    count = summary.loc[
        summary["지역"] == region, "데이터_개수"
    ].iloc[0]

    col.metric(region, f"{int(count):,}개")


st.divider()

# 평균 발전량 비교
st.subheader("지역별 평균 전체 발전량")

fig_mean = px.bar(
    summary,
    x="지역",
    y="평균_발전량",
    text_auto=".2s",
    labels={
        "지역": "지역",
        "평균_발전량": "평균 전체 발전량"
    }
)

fig_mean.update_layout(
    xaxis_title="지역",
    yaxis_title="평균 전체 발전량",
    showlegend=False
)

st.plotly_chart(fig_mean, use_container_width=True)


# 발전량 분포 비교
st.subheader("지역별 전체 발전량 분포")

fig_box = px.box(
    data,
    x="지역",
    y="Total Power",
    category_orders={"지역": region_order},
    labels={
        "지역": "지역",
        "Total Power": "전체 발전량"
    },
    points="outliers"
)

fig_box.update_layout(
    xaxis_title="지역",
    yaxis_title="전체 발전량",
    showlegend=False
)

st.plotly_chart(fig_box, use_container_width=True)


# 최대 및 최소 발전량
st.subheader("지역별 최대·최소 전체 발전량")

result_table = summary[
    ["지역", "최대_발전량", "최소_발전량"]
].copy()

result_table.columns = [
    "지역",
    "최대 발전량",
    "최소 발전량"
]

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True
)


# 데이터 설명
st.divider()
st.caption(
    f"전체 데이터: {len(data):,}개 행 | "
    "분석 대상: 4개 지역의 전체 발전량"
)
