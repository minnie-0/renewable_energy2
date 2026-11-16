
from pathlib import Path
from io import BytesIO
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="최적 장치 배치 탐색실",
    page_icon="🌊",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = APP_DIR / "wave_energy_reduced.zip"

X_COLUMNS = [f"X{i}" for i in range(1, 17)]
Y_COLUMNS = [f"Y{i}" for i in range(1, 17)]

FEATURE_COLUMNS = X_COLUMNS + Y_COLUMNS
TARGET_COLUMN = "Total Power"

DATA_COLUMNS = (
    X_COLUMNS
    + Y_COLUMNS
    + [f"P{i}" for i in range(1, 17)]
    + [TARGET_COLUMN]
)

REGION_NAMES = {
    "Adelaide": "애들레이드",
    "Perth": "퍼스",
    "Sydney": "시드니",
    "Tasmania": "태즈메이니아"
}


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data(zip_path):
    all_data = []

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

            for column in DATA_COLUMNS:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

            df["지역"] = REGION_NAMES[region_code]
            all_data.append(df)

    if not all_data:
        raise ValueError("지역별 CSV 데이터를 찾지 못했습니다.")

    result = pd.concat(all_data, ignore_index=True)

    # 모델 학습에 필요한 값이 없는 행 제외
    result = result.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    return result


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------

@st.cache_resource
def train_model(zip_path):
    df = load_data(zip_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(X, y)

    return model, df


# --------------------------------------------------
# 세션 상태 초기화
# --------------------------------------------------

if "배치_사례" not in st.session_state:
    st.session_state["배치_사례"] = []

if "최근_예측" not in st.session_state:
    st.session_state["최근_예측"] = None


# --------------------------------------------------
# 제목
# --------------------------------------------------

st.title("🌊 최적 장치 배치 탐색실")

st.write(
    "16개 파력 발전 장치의 좌표를 직접 설정하고 "
    "선형 회귀 모델을 이용해 전체 발전량을 예측합니다."
)

st.divider()


# --------------------------------------------------
# 데이터 및 모델 준비
# --------------------------------------------------

try:
    if not ZIP_PATH.exists():
        st.error(
            "wave_energy_reduced.zip 파일을 "
            "프로젝트 최상위 폴더에 넣어 주세요."
        )
        st.stop()

    model, data = train_model(str(ZIP_PATH))

except zipfile.BadZipFile:
    st.error("ZIP 파일을 읽을 수 없습니다. 파일 상태를 확인해 주세요.")
    st.stop()

except Exception as e:
    st.error(f"데이터 또는 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 기존 데이터의 좌표 범위
# --------------------------------------------------

x_min = float(data[X_COLUMNS].min().min())
x_max = float(data[X_COLUMNS].max().max())

y_min = float(data[Y_COLUMNS].min().min())
y_max = float(data[Y_COLUMNS].max().max())

# 각 장치의 기본 좌표는 해당 장치 좌표의 중앙값 사용
default_x = [
    float(data[column].median())
    for column in X_COLUMNS
]

default_y = [
    float(data[column].median())
    for column in Y_COLUMNS
]


# --------------------------------------------------
# 좌표 입력
# --------------------------------------------------

st.header("1. 장치 배치 좌표 설정")

st.write(
    "각 장치의 X좌표와 Y좌표를 입력해 주세요. "
    "처음 표시되는 값은 기존 데이터에서 계산한 각 장치 좌표의 중앙값입니다."
)

st.caption(
    f"기존 데이터의 X좌표 범위: {x_min:,.2f} ~ {x_max:,.2f}"
)

st.caption(
    f"기존 데이터의 Y좌표 범위: {y_min:,.2f} ~ {y_max:,.2f}"
)

scenario_name = st.text_input(
    "배치 사례 이름",
    placeholder="예: 배치 사례 1"
)

input_x = []
input_y = []

with st.form("coordinate_form"):

    st.subheader("장치별 좌표 입력")

    for i in range(16):
        col_x, col_y = st.columns(2)

        with col_x:
            x_value = st.number_input(
                f"{i + 1}번 장치 X좌표",
                value=default_x[i],
                step=1000.0,
                format="%.2f",
                key=f"input_x_{i}"
            )

        with col_y:
            y_value = st.number_input(
                f"{i + 1}번 장치 Y좌표",
                value=default_y[i],
                step=1000.0,
                format="%.2f",
                key=f"input_y_{i}"
            )

        input_x.append(x_value)
        input_y.append(y_value)

    submitted = st.form_submit_button(
        "전체 발전량 예측 및 사례 저장",
        use_container_width=True
    )


# --------------------------------------------------
# 예측 실행 및 사례 저장
# --------------------------------------------------

if submitted:

    input_data = {}

    for i in range(16):
        input_data[f"X{i + 1}"] = input_x[i]

    for i in range(16):
        input_data[f"Y{i + 1}"] = input_y[i]

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURE_COLUMNS
    )

    predicted_power = float(
        model.predict(input_df)[0]
    )

    # 좌표가 기존 데이터의 전체 범위를 벗어나는지 확인
    outside_x = [
        i + 1
        for i, value in enumerate(input_x)
        if value < x_min or value > x_max
    ]

    outside_y = [
        i + 1
        for i, value in enumerate(input_y)
        if value < y_min or value > y_max
    ]

    # 사례 이름이 비어 있으면 자동으로 이름 지정
    if scenario_name.strip():
        saved_name = scenario_name.strip()
    else:
        saved_name = f"배치 사례 {len(st.session_state['배치_사례']) + 1}"

    scenario = {
        "이름": saved_name,
        "예측 발전량": predicted_power,
        "X좌표": input_x.copy(),
        "Y좌표": input_y.copy(),
        "범위 벗어난 X": outside_x,
        "범위 벗어난 Y": outside_y
    }

    st.session_state["배치_사례"].append(scenario)
    st.session_state["최근_예측"] = scenario


# --------------------------------------------------
# 예측 결과
# --------------------------------------------------

st.divider()
st.header("2. 전체 발전량 예측 결과")

latest = st.session_state["최근_예측"]

if latest is None:

    st.info(
        "좌표를 입력한 뒤 '전체 발전량 예측 및 사례 저장' 버튼을 누르면 "
        "예측 결과가 여기에 표시됩니다."
    )

else:

    st.metric(
        label="예측된 전체 발전량",
        value=f"{latest['예측 발전량']:,.2f}"
    )

    st.write(f"배치 사례 이름: **{latest['이름']}**")

    if latest["범위 벗어난 X"] or latest["범위 벗어난 Y"]:

        st.warning(
            "일부 좌표가 기존 데이터의 범위를 벗어났습니다. "
            "이 경우 모델이 학습한 범위 밖에서 예측하므로 "
            "결과의 신뢰도가 낮을 수 있습니다."
        )

        if latest["범위 벗어난 X"]:
            st.write(
                "기존 X좌표 범위를 벗어난 장치: "
                + ", ".join(
                    f"{i}번" for i in latest["범위 벗어난 X"]
                )
            )

        if latest["범위 벗어난 Y"]:
            st.write(
                "기존 Y좌표 범위를 벗어난 장치: "
                + ", ".join(
                    f"{i}번" for i in latest["범위 벗어난 Y"]
                )
            )

    else:
        st.success(
            "입력한 모든 좌표가 기존 데이터의 전체 좌표 범위 안에 있습니다."
        )

    st.caption(
        "좌표가 기존 데이터의 범위 안에 있더라도 "
        "예측값이 실제 발전량과 일치한다고 보장할 수는 없습니다."
    )


# --------------------------------------------------
# 입력한 장치의 배치 위치 시각화
# --------------------------------------------------

st.divider()
st.header("3. 장치 배치 위치 확인")

if latest is not None:

    position_data = pd.DataFrame({
        "장치": [f"{i}번 장치" for i in range(1, 17)],
        "X좌표": latest["X좌표"],
        "Y좌표": latest["Y좌표"]
    })

    fig_position = px.scatter(
        position_data,
        x="X좌표",
        y="Y좌표",
        text="장치",
        hover_name="장치",
        title=f"{latest['이름']}의 장치 배치 위치",
        labels={
            "X좌표": "X좌표",
            "Y좌표": "Y좌표"
        }
    )

    fig_position.update_traces(
        textposition="top center",
        marker_size=11
    )

    fig_position.update_layout(
        height=700,
        xaxis_title="X좌표",
        yaxis_title="Y좌표"
    )

    st.plotly_chart(
        fig_position,
        use_container_width=True
    )


# --------------------------------------------------
# 저장한 배치 사례 비교
# --------------------------------------------------

st.divider()
st.header("4. 배치 사례별 예측 발전량 비교")

scenarios = st.session_state["배치_사례"]

if not scenarios:

    st.info(
        "아직 저장된 배치 사례가 없습니다. "
        "좌표를 입력하고 예측을 실행해 주세요."
    )

else:

    comparison_data = pd.DataFrame([
        {
            "배치 사례": item["이름"],
            "예측 발전량": item["예측 발전량"]
        }
        for item in scenarios
    ])

    comparison_data = comparison_data.sort_values(
        "예측 발전량",
        ascending=False
    ).reset_index(drop=True)

    comparison_data.insert(
        0,
        "순위",
        range(1, len(comparison_data) + 1)
    )

    st.subheader("예측 발전량 순위")

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )

    fig_comparison = px.bar(
        comparison_data,
        x="배치 사례",
        y="예측 발전량",
        color="배치 사례",
        text_auto=".2s",
        title="배치 사례별 예측 전체 발전량 비교",
        labels={
            "배치 사례": "배치 사례",
            "예측 발전량": "예측 전체 발전량"
        }
    )

    fig_comparison.update_layout(
        height=600,
        showlegend=False,
        xaxis_title="배치 사례",
        yaxis_title="예측 전체 발전량"
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )

    st.subheader("저장된 배치 사례 관리")

    selected_index = st.selectbox(
        "삭제할 배치 사례 선택",
        options=range(len(scenarios)),
        format_func=lambda i: scenarios[i]["이름"]
    )

    if st.button("선택한 사례 삭제"):
        st.session_state["배치_사례"].pop(selected_index)
        st.rerun()


# --------------------------------------------------
# 분석 시 유의사항
# --------------------------------------------------

st.divider()

st.header("5. 분석 시 유의사항")

st.write(
    "이 페이지는 기존 파력 발전 데이터의 장치 배치 좌표와 전체 발전량을 "
    "학습한 선형 회귀 모델을 이용합니다. 입력한 좌표에 따른 발전량은 "
    "모델의 예측값이며 실제 측정값이 아닙니다."
)

st.write(
    "예측 발전량이 높은 배치 사례가 실제 환경에서도 가장 효율적이라고 "
    "단정할 수 없습니다. 파도의 높이와 주기 등 모델에 포함되지 않은 "
    "조건도 발전량에 영향을 줄 수 있습니다."
)
