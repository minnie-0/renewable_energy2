import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ==========================================
# 기본 설정
# ==========================================

st.set_page_config(
    page_title="장치 배치 분석실",
    page_icon="📊",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent.parent
ZIP_NAME = "wave+energy+converters.zip"
ZIP_PATH = APP_DIR / ZIP_NAME

X_COLS = [f"X{i}" for i in range(1, 17)]
Y_COLS = [f"Y{i}" for i in range(1, 17)]
P_COLS = [f"P{i}" for i in range(1, 17)]

FEATURE_COLS = X_COLS + Y_COLS
DATA_COLS = FEATURE_COLS + P_COLS + ["Total Power"]

REGION_FILES = {
    "Adelaide": "Adelaide_Data.csv",
    "Perth": "Perth_Data.csv",
    "Sydney": "Sydney_Data.csv",
    "Tasmania": "Tasmania_Data.csv",
}


# ==========================================
# 데이터 압축 파일 찾기
# ==========================================

def locate_zip():

    if ZIP_PATH.exists():
        return ZIP_PATH

    candidates = list(APP_DIR.glob("*.zip"))

    for candidate in candidates:

        normalized = (
            candidate.name
            .lower()
            .replace(" ", "")
        )

        if "wave" in normalized and "energy" in normalized:
            return candidate

    return None


# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data(
    show_spinner="파력 발전 데이터를 불러오는 중..."
)
def load_data(zip_path, zip_mtime):

    del zip_mtime

    frames = []

    with zipfile.ZipFile(zip_path, "r") as zf:

        names = zf.namelist()

        for region, filename in REGION_FILES.items():

            matches = [
                name
                for name in names
                if Path(name).name.lower() == filename.lower()
            ]

            if not matches:

                raise FileNotFoundError(
                    f"압축 파일 내부에서 {filename} 파일을 찾지 못했습니다."
                )

            with zf.open(matches[0]) as file:

                frame = pd.read_csv(
                    file,
                    header=None
                )

            if frame.shape[1] != 49:

                raise ValueError(
                    f"{filename}의 열 개수가 49개가 아닙니다."
                )

            frame.columns = DATA_COLS
            frame["지역"] = region

            frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True
    )


# ==========================================
# 페이지 제목
# ==========================================

st.title("📊 장치 배치 분석실")

st.caption(
    "장치의 배치 좌표를 변화시키며 전체 발전량 예측 결과를 비교합니다."
)

st.write(
    "16개 장치 중 하나를 선택하고 해당 장치의 가로 또는 세로 좌표를 "
    "변화시켜 발전량 예측 결과를 분석할 수 있습니다. "
    "분석하는 좌표를 제외한 나머지 31개 좌표는 고정됩니다."
)


# ==========================================
# 데이터 불러오기
# ==========================================

zip_file = locate_zip()

if zip_file is None:

    st.error(
        "데이터 압축 파일을 찾을 수 없습니다. "
        "main.py와 같은 폴더에 압축 파일이 있는지 확인해 주세요."
    )

    st.stop()


try:

    data = load_data(
        str(zip_file),
        zip_file.stat().st_mtime
    )

except (
    zipfile.BadZipFile,
    OSError,
    pd.errors.ParserError,
    ValueError,
    FileNotFoundError
) as e:

    st.error(
        f"데이터를 불러오지 못했습니다: {e}"
    )

    st.stop()


# ==========================================
# 모델 학습
# ==========================================

model_data = data[
    FEATURE_COLS + ["Total Power"]
].dropna()

if len(model_data) < 10:

    st.error(
        "모델 학습에 필요한 데이터가 충분하지 않습니다."
    )

    st.stop()


X = model_data[FEATURE_COLS]
y = model_data["Total Power"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================
# 기준 배치 설정
# ==========================================

default_row = data.iloc[0]

# 메인 페이지에서 입력한 좌표가 세션에 남아 있다면 사용하고
# 그렇지 않으면 데이터의 첫 번째 행을 기준으로 설정

base_coord = {}

for col in FEATURE_COLS:

    saved_value = st.session_state.get(
        f"coord_{col}",
        default_row[col]
    )

    base_coord[col] = float(saved_value)


st.divider()


# ==========================================
# 분석할 장치 선택
# ==========================================

st.subheader("1. 분석할 장치 선택")

device = st.selectbox(
    "좌표를 변화시킬 장치",
    options=list(range(1, 17)),
    format_func=lambda x: f"{x}번 장치"
)


direction = st.radio(
    "변화시킬 좌표 방향",
    options=["가로 좌표", "세로 좌표"],
    horizontal=True
)


if direction == "가로 좌표":

    target_col = f"X{device}"

else:

    target_col = f"Y{device}"


current_value = base_coord[target_col]


st.info(
    f"선택한 장치: {device}번 장치 | "
    f"변화시킬 좌표: {direction} | "
    f"현재 좌표: {current_value:,.4f}"
)


# ==========================================
# 좌표 변화 범위 설정
# ==========================================

st.divider()

st.subheader("2. 좌표 변화 범위 설정")

observed_min = float(
    data[target_col].min()
)

observed_max = float(
    data[target_col].max()
)

coordinate_range = observed_max - observed_min

if coordinate_range > 0:

    default_min = observed_min
    default_max = observed_max
    default_step = coordinate_range / 50

else:

    default_min = current_value - 1
    default_max = current_value + 1
    default_step = 0.1


if default_step <= 0:
    default_step = 0.1


st.caption(
    "기본 범위는 데이터에 기록된 해당 좌표의 최솟값과 최댓값입니다. "
    "원하는 범위와 변화 간격으로 수정할 수 있습니다."
)


range_cols = st.columns(3)

with range_cols[0]:

    min_value = st.number_input(
        "좌표 최솟값",
        value=float(default_min),
        format="%.6f",
        key="analysis_min"
    )


with range_cols[1]:

    max_value = st.number_input(
        "좌표 최댓값",
        value=float(default_max),
        format="%.6f",
        key="analysis_max"
    )


with range_cols[2]:

    step_value = st.number_input(
        "좌표 변화 간격",
        min_value=0.000001,
        value=float(default_step),
        format="%.6f",
        key="analysis_step"
    )


# ==========================================
# 분석 실행
# ==========================================

st.divider()

st.subheader("3. 좌표 변화에 따른 발전량 분석")


if st.button(
    "좌표 변화 분석하기",
    type="primary"
):

    if min_value >= max_value:

        st.error(
            "최댓값은 최솟값보다 커야 합니다."
        )

    elif step_value <= 0:

        st.error(
            "좌표 변화 간격은 0보다 커야 합니다."
        )

    elif not min_value <= current_value <= max_value:

        st.warning(
            "현재 좌표가 설정한 범위에 포함되지 않습니다. "
            "현재 좌표를 포함하도록 범위를 수정해 주세요."
        )

    else:

        coordinate_values = np.arange(
            min_value,
            max_value + step_value * 1e-9,
            step_value
        )

        if len(coordinate_values) == 0:

            st.error(
                "분석할 좌표가 없습니다."
            )

            st.stop()

        elif len(coordinate_values) > 1000:

            st.error(
                "분석할 좌표가 1,000개를 초과합니다. "
                "변화 간격을 늘려 주세요."
            )

            st.stop()


        # ==========================================
        # 좌표별 예측 데이터 생성
        # ==========================================

        input_data = pd.DataFrame(
            [base_coord.copy() for _ in coordinate_values],
            columns=FEATURE_COLS
        )

        # 선택한 좌표만 변경

        input_data[target_col] = coordinate_values


        # ==========================================
        # 발전량 예측
        # ==========================================

        predictions = model.predict(
            input_data
        )


        result_df = pd.DataFrame(
            {
                "변화시킨 좌표": coordinate_values,
                "예측 전체 발전량": predictions
            }
        )


        # ==========================================
        # 현재 배치의 발전량
        # ==========================================

        current_input = pd.DataFrame(
            [base_coord],
            columns=FEATURE_COLS
        )

        current_prediction = float(
            model.predict(current_input)[0]
        )


        # ==========================================
        # 가장 높은 예측 발전량
        # ==========================================

        best_index = result_df[
            "예측 전체 발전량"
        ].idxmax()

        best_row = result_df.loc[best_index]

        best_coordinate = float(
            best_row["변화시킨 좌표"]
        )

        best_prediction = float(
            best_row["예측 전체 발전량"]
        )


        # ==========================================
        # 분석 결과
        # ==========================================

        st.subheader("분석 결과")

        result_cols = st.columns(3)

        result_cols[0].metric(
            "현재 좌표",
            f"{current_value:,.4f}"
        )

        result_cols[1].metric(
            "현재 배치의 예측 발전량",
            f"{current_prediction:,.4f}"
        )

        result_cols[2].metric(
            "가장 높은 예측 발전량",
            f"{best_prediction:,.4f}"
        )


        st.write(
            f"{device}번 장치의 {direction}를 "
            f"{best_coordinate:,.4f}로 설정했을 때 "
            f"가장 높은 예측 발전량이 나타났습니다."
        )


        # ==========================================
        # 좌표 변화 그래프
        # ==========================================

        st.subheader("좌표에 따른 예측 발전량 변화")

        fig = px.line(
            result_df,

            x="변화시킨 좌표",

            y="예측 전체 발전량",

            markers=True,

            title=f"{device}번 장치의 {direction} 변화에 따른 발전량"
        )


        # 현재 배치 표시

        fig.add_scatter(
            x=[current_value],
            y=[current_prediction],
            mode="markers",
            name="현재 배치",
            marker=dict(
                size=12,
                symbol="diamond"
            )
        )


        # 가장 높은 예측값 표시

        fig.add_scatter(
            x=[best_coordinate],
            y=[best_prediction],
            mode="markers",
            name="가장 높은 예측값",
            marker=dict(
                size=13,
                symbol="star"
            )
        )


        fig.update_layout(
            xaxis_title="장치의 좌표",
            yaxis_title="예측 전체 발전량",
            height=550,
            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ==========================================
        # 결과 데이터
        # ==========================================

        with st.expander("좌표별 예측 결과 확인"):

            st.dataframe(
                result_df,
                hide_index=True,
                use_container_width=True
            )


        # ==========================================
        # 결과 해석
        # ==========================================

        st.subheader("결과 해석")

        difference = (
            best_prediction - current_prediction
        )

        st.write(
            f"현재 배치의 예측 발전량과 비교했을 때 "
            f"가장 높은 예측 발전량은 {difference:,.4f}만큼 "
            f"차이가 납니다."
        )

        st.caption(
            "주의: 이 분석 결과는 학습된 선형 회귀 모델의 예측값을 "
            "비교한 것입니다. 실제 파력 발전 장치의 최적 배치나 "
            "실제 발전량을 보장하지 않습니다. "
            "또한 한 번에 하나의 좌표만 변화시키므로 "
            "다른 장치의 좌표와 함께 변화할 때의 영향은 반영하지 않습니다."
        )


# ==========================================
# 분석 방법 안내
# ==========================================

st.divider()

with st.expander("분석 방법 알아보기"):

    st.write(
        "이 페이지에서는 기존 파력 발전 데이터의 32개 배치 좌표를 "
        "입력값으로 사용하고 전체 발전량을 예측하는 선형 회귀 모델을 "
        "학습합니다."
    )

    st.write(
        "분석할 장치와 좌표 방향을 선택한 다음 좌표 변화 범위를 설정합니다. "
        "선택한 좌표만 일정한 간격으로 변화시키고 나머지 좌표는 "
        "기준 배치의 값으로 고정합니다."
    )

    st.write(
        "각 좌표 조합에 대한 예측 발전량을 계산하고 그래프로 나타내어 "
        "좌표 변화와 예측 발전량 사이의 관계를 비교합니다."
    )
