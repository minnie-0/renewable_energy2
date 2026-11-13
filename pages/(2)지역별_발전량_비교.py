
from pathlib import Path
from io import BytesIO
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="지역별 장치 배치 특성 분석실",
    page_icon="🌊",
    layout="wide"
)

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = APP_DIR / "wave_energy_reduced.zip"

DATA_COLUMNS = (
    [f"X{i}" for i in range(1, 17)]
    + [f"Y{i}" for i in range(1, 17)]
    + [f"P{i}" for i in range(1, 17)]
    + ["Total Power"]
)

REGION_NAMES = {
    "Adelaide": "애들레이드",
    "Perth": "퍼스",
    "Sydney": "시드니",
    "Tasmania": "태즈메이니아"
}

REGION_ORDER = [
    "애들레이드",
    "퍼스",
    "시드니",
    "태즈메이니아"
]

X_COLUMNS = [f"X{i}" for i in range(1, 17)]
Y_COLUMNS = [f"Y{i}" for i in range(1, 17)]


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data(zip_path):
    region_data = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for file_name in zip_file.namelist():

            if not file_name.lower().endswith(".csv"):
                continue

            region_code = (
                Path(file_name).stem.replace("_Data", "")
            )

            if region_code not in REGION_NAMES:
                continue

            with zip_file.open(file_name) as file:
                df = pd.read_csv(
                    BytesIO(file.read()),
                    header=None,
                    names=DATA_COLUMNS
                )

            if df.shape[1] != len(DATA_COLUMNS):
                continue

            # 숫자형으로 변환
            for column in DATA_COLUMNS:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

            df["지역"] = REGION_NAMES[region_code]
            region_data.append(df)

    if not region_data:
        raise ValueError(
            "ZIP 파일에서 지역별 CSV 데이터를 찾지 못했습니다."
        )

    return pd.concat(region_data, ignore_index=True)


# --------------------------------------------------
# 장치별 좌표 데이터 만들기
# --------------------------------------------------

def make_coordinate_data(df):
    coordinate_data = []

    for i in range(1, 17):
        device_data = pd.DataFrame({
            "지역": df["지역"],
            "장치": f"{i}번 장치",
            "X좌표": df[f"X{i}"],
            "Y좌표": df[f"Y{i}"],
            "전체 발전량": df["Total Power"]
        })

        coordinate_data.append(device_data)

    return pd.concat(coordinate_data, ignore_index=True)


# --------------------------------------------------
# 분석용 데이터 만들기
# --------------------------------------------------

def make_analysis_data(df):
    result = df[["지역", "Total Power"]].copy()

    # 한 행에 포함된 16개 장치의 평균 좌표와 좌표 분산
    result["평균 X좌표"] = df[X_COLUMNS].mean(axis=1)
    result["평균 Y좌표"] = df[Y_COLUMNS].mean(axis=1)

    result["X좌표 표준편차"] = df[X_COLUMNS].std(axis=1)
    result["Y좌표 표준편차"] = df[Y_COLUMNS].std(axis=1)

    result = result.rename(
        columns={"Total Power": "전체 발전량"}
    )

    return result


# --------------------------------------------------
# 페이지 제목
# --------------------------------------------------

st.title("🌊 지역별 장치 배치 특성 분석실")

st.write(
    "지역별 파력 발전 장치의 좌표 분포를 비교하고 "
    "장치 배치 특성과 전체 발전량 사이의 관계를 분석합니다."
)

st.divider()


# --------------------------------------------------
# 데이터 불러오기 및 오류 처리
# --------------------------------------------------

try:
    if not ZIP_PATH.exists():
        st.error(
            "wave_energy_reduced.zip 파일을 "
            "프로젝트 최상위 폴더에 넣어 주세요."
        )
        st.stop()

    data = load_data(str(ZIP_PATH))

except zipfile.BadZipFile:
    st.error("ZIP 파일을 읽을 수 없습니다. 파일이 손상되었는지 확인해 주세요.")
    st.stop()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


coordinate_data = make_coordinate_data(data)
analysis_data = make_analysis_data(data)

# 발전량이나 좌표가 없는 행은 해당 분석에서 제외
coordinate_data = coordinate_data.dropna(
    subset=["X좌표", "Y좌표"]
)

analysis_data = analysis_data.dropna(
    subset=["전체 발전량", "평균 X좌표", "평균 Y좌표"]
)


# --------------------------------------------------
# 1. 지역별 데이터 개수와 평균 발전량
# --------------------------------------------------

st.header("1. 지역별 데이터 현황")

summary = (
    data.groupby("지역")["Total Power"]
    .agg(
        데이터_개수="count",
        평균_발전량="mean"
    )
    .reindex(REGION_ORDER)
    .reset_index()
)

summary = summary.rename(columns={
    "지역": "지역",
    "데이터_개수": "데이터 개수",
    "평균_발전량": "평균 전체 발전량"
})

columns = st.columns(4)

for i, region in enumerate(REGION_ORDER):
    row = summary[summary["지역"] == region]

    if row.empty:
        continue

    count = int(row["데이터 개수"].iloc[0])
    mean_power = row["평균 전체 발전량"].iloc[0]

    with columns[i]:
        st.metric(
            label=f"{region} 데이터 개수",
            value=f"{count:,}개"
        )
        st.metric(
            label="평균 전체 발전량",
            value=f"{mean_power:,.2f}"
        )

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.divider()


# --------------------------------------------------
# 2. 지역별 X좌표 분포
# --------------------------------------------------

st.header("2. 지역별 장치 X좌표 분포")

st.write(
    "각 지역에 배치된 16개 장치의 X좌표를 모아 "
    "분포를 비교합니다. 상자그림을 통해 좌표의 중앙값과 "
    "분포 범위를 확인할 수 있습니다."
)

fig_x = px.box(
    coordinate_data,
    x="지역",
    y="X좌표",
    color="지역",
    category_orders={"지역": REGION_ORDER},
    points="outliers",
    title="지역별 X좌표 분포",
    labels={
        "지역": "지역",
        "X좌표": "X좌표"
    }
)

fig_x.update_layout(
    height=650,
    showlegend=False,
    xaxis_title="지역",
    yaxis_title="X좌표"
)

st.plotly_chart(fig_x, use_container_width=True)


# --------------------------------------------------
# 3. 지역별 Y좌표 분포
# --------------------------------------------------

st.header("3. 지역별 장치 Y좌표 분포")

st.write(
    "Y좌표의 분포도 같은 방식으로 비교합니다. "
    "X좌표 분포와 함께 살펴보면 지역별 장치 배치 위치의 "
    "차이를 파악할 수 있습니다."
)

fig_y = px.box(
    coordinate_data,
    x="지역",
    y="Y좌표",
    color="지역",
    category_orders={"지역": REGION_ORDER},
    points="outliers",
    title="지역별 Y좌표 분포",
    labels={
        "지역": "지역",
        "Y좌표": "Y좌표"
    }
)

fig_y.update_layout(
    height=650,
    showlegend=False,
    xaxis_title="지역",
    yaxis_title="Y좌표"
)

st.plotly_chart(fig_y, use_container_width=True)


# --------------------------------------------------
# 4. 지역별 실제 장치 배치 위치
# --------------------------------------------------

st.header("4. 지역별 장치 배치 위치")

st.write(
    "16개 장치의 X좌표와 Y좌표를 2차원 평면에 표시합니다. "
    "각 지역의 점들이 모여 있는 위치와 분포 범위를 비교할 수 있습니다."
)

fig_position = px.scatter(
    coordinate_data,
    x="X좌표",
    y="Y좌표",
    color="지역",
    category_orders={"지역": REGION_ORDER},
    opacity=0.35,
    title="지역별 장치 좌표 분포",
    labels={
        "X좌표": "X좌표",
        "Y좌표": "Y좌표",
        "지역": "지역"
    },
    hover_data=["장치"]
)

fig_position.update_layout(
    height=750,
    xaxis_title="X좌표",
    yaxis_title="Y좌표",
    legend_title="지역"
)

st.plotly_chart(fig_position, use_container_width=True)


# --------------------------------------------------
# 5. 장치 배치 좌표와 전체 발전량의 관계
# --------------------------------------------------

st.header("5. 장치 배치 좌표와 전체 발전량의 관계")

st.write(
    "각 데이터에서 16개 장치의 평균 X좌표와 평균 Y좌표를 계산한 뒤 "
    "전체 발전량과 비교합니다. 점 하나는 하나의 배치 사례를 의미합니다. "
    "평균 좌표만으로는 개별 장치의 배치 형태가 모두 드러나지 않는다는 "
    "점에 유의해야 합니다."
)

col1, col2 = st.columns(2)

with col1:
    fig_mean_x = px.scatter(
        analysis_data,
        x="평균 X좌표",
        y="전체 발전량",
        color="지역",
        category_orders={"지역": REGION_ORDER},
        opacity=0.55,
        title="평균 X좌표와 전체 발전량",
        labels={
            "평균 X좌표": "16개 장치의 평균 X좌표",
            "전체 발전량": "전체 발전량",
            "지역": "지역"
        }
    )

    fig_mean_x.update_layout(height=550)
    st.plotly_chart(fig_mean_x, use_container_width=True)

with col2:
    fig_mean_y = px.scatter(
        analysis_data,
        x="평균 Y좌표",
        y="전체 발전량",
        color="지역",
        category_orders={"지역": REGION_ORDER},
        opacity=0.55,
        title="평균 Y좌표와 전체 발전량",
        labels={
            "평균 Y좌표": "16개 장치의 평균 Y좌표",
            "전체 발전량": "전체 발전량",
            "지역": "지역"
        }
    )

    fig_mean_y.update_layout(height=550)
    st.plotly_chart(fig_mean_y, use_container_width=True)


# --------------------------------------------------
# 6. 좌표와 발전량의 상관관계
# --------------------------------------------------

st.header("6. 좌표와 발전량의 상관관계")

st.write(
    "평균 좌표와 전체 발전량 사이의 피어슨 상관계수를 계산합니다. "
    "상관계수는 두 변수 사이의 선형적인 관계를 나타내며 "
    "좌표가 발전량 차이의 원인이라는 사실을 직접 증명하지는 않습니다."
)

correlation_data = []

for region in REGION_ORDER:
    region_df = analysis_data[
        analysis_data["지역"] == region
    ]

    x_corr = region_df[
        ["평균 X좌표", "전체 발전량"]
    ].corr().iloc[0, 1]

    y_corr = region_df[
        ["평균 Y좌표", "전체 발전량"]
    ].corr().iloc[0, 1]

    correlation_data.append({
        "지역": region,
        "평균 X좌표와 발전량의 상관계수": x_corr,
        "평균 Y좌표와 발전량의 상관계수": y_corr
    })

correlation_df = pd.DataFrame(correlation_data)

st.dataframe(
    correlation_df.round(3),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "상관계수가 1에 가까우면 강한 양의 선형 관계를, "
    "-1에 가까우면 강한 음의 선형 관계를 나타냅니다. "
    "0에 가까우면 선형 관계가 약하다는 뜻입니다. "
    "상관관계만으로 인과관계를 판단할 수는 없습니다."
)


# --------------------------------------------------
# 7. 분석 결과 안내
# --------------------------------------------------

st.divider()

st.header("7. 분석 결과 해석")

st.write(
    "지역별 X좌표와 Y좌표의 상자그림 및 2차원 산점도를 통해 "
    "장치 배치 위치와 분포 범위를 비교할 수 있습니다."
)

st.write(
    "평균 좌표와 전체 발전량의 산점도 및 상관계수를 함께 살펴보면 "
    "장치 배치 위치와 발전량 사이에 선형적인 관계가 나타나는지 "
    "확인할 수 있습니다."
)

st.info(
    "상관계수가 낮거나 산점도에서 뚜렷한 경향이 보이지 않는다면 "
    "좌표와 발전량 사이의 선형 관계가 뚜렷하지 않다고 해석해야 합니다. "
    "장치 배치 형태나 다른 조건도 발전량에 영향을 줄 수 있으므로 "
    "좌표만으로 발전량 차이를 단정하지 않습니다."
)
