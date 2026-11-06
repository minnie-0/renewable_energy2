import json
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 기본 설정
# ==========================================

APP_TITLE = "파력 발전량 예측 실습실"

BASE_DIR = Path(__file__).parent
ZIP_PATH = BASE_DIR / "wave+energy+converters.zip"

RECORD_DIR = BASE_DIR / "내_기록"
RECORD_PATH = RECORD_DIR / "파력발전_데이터_기록.json"

X_COLS = [f"X{i}" for i in range(1, 17)]
Y_COLS = [f"Y{i}" for i in range(1, 17)]
P_COLS = [f"P{i}" for i in range(1, 17)]

FEATURE_COLS = X_COLS + Y_COLS

DATA_COLS = FEATURE_COLS + P_COLS + ["Total Power"]

REGION_FILES = {
    "Adelaide": "WECs_DataSet/Adelaide_Data.csv",
    "Perth": "WECs_DataSet/Perth_Data.csv",
    "Sydney": "WECs_DataSet/Sydney_Data.csv",
    "Tasmania": "WECs_DataSet/Tasmania_Data.csv",
}


# ==========================================
# 열 이름과 우리말 뜻
# ==========================================

KOREAN_MEANINGS = {
    **{
        f"X{i}": f"{i}번 장치의 X 좌표"
        for i in range(1, 17)
    },
    **{
        f"Y{i}": f"{i}번 장치의 Y 좌표"
        for i in range(1, 17)
    },
    **{
        f"P{i}": f"{i}번 장치의 개별 발전량"
        for i in range(1, 17)
    },
    "Total Power": "전체 발전량",
}


# ==========================================
# Streamlit 페이지 설정
# ==========================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌊",
    layout="wide",
)


# ==========================================
# 기록 파일 불러오기
# ==========================================

def load_records():

    if RECORD_PATH.exists():

        try:

            with RECORD_PATH.open(
                "r",
                encoding="utf-8"
            ) as file:

                saved = json.load(file)

            if isinstance(saved, dict):
                return saved

        except (json.JSONDecodeError, OSError):

            pass

    return {
        "source": "wave+energy+converters.zip",
        "meanings": KOREAN_MEANINGS.copy(),
    }


# ==========================================
# 기록 파일 저장하기
# ==========================================

def save_records(records):

    try:

        RECORD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with RECORD_PATH.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2
            )

        return True

    except OSError as error:

        st.error(
            f"기록 파일을 저장하지 못했습니다: {error}"
        )

        return False


# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data(show_spinner="파력 발전 데이터를 읽는 중...")
def load_wave_data(zip_path_string):

    zip_path = Path(zip_path_string)

    if not zip_path.exists():

        return (
            None,
            "데이터 ZIP 파일을 찾을 수 없습니다."
        )

    frames = []

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:

            archive_files = archive.namelist()

            for region, csv_path in REGION_FILES.items():

                if csv_path not in archive_files:

                    return (
                        None,
                        f"ZIP 파일 안에서 {csv_path} 파일을 찾을 수 없습니다."
                    )

                with archive.open(csv_path) as csv_file:

                    frame = pd.read_csv(
                        csv_file,
                        header=None
                    )

                if frame.shape[1] != 49:

                    return (
                        None,
                        f"{region} 데이터의 열 개수가 49개가 아닙니다. "
                        f"현재 열 개수: {frame.shape[1]}"
                    )

                frame.columns = DATA_COLS

                frame["지역"] = region

                frames.append(frame)

        combined = pd.concat(
            frames,
            ignore_index=True
        )

        return combined, None

    except (
        zipfile.BadZipFile,
        OSError,
        pd.errors.ParserError
    ) as error:

        return (
            None,
            f"데이터를 읽는 중 오류가 발생했습니다: {error}"
        )


# ==========================================
# 앱 제목
# ==========================================

st.title(
    "🌊 파력 발전 장치 배치 좌표를 이용한 전체 발전량 예측"
)

st.caption(
    "파력 발전 장치의 위치 정보와 발전량 데이터를 탐색하고 "
    "회귀 모델을 이용해 전체 발전량을 예측합니다."
)


# ==========================================
# 데이터 불러오기
# ==========================================

records = load_records()

data, data_error = load_wave_data(
    str(ZIP_PATH)
)


# ==========================================
# 데이터가 없을 때 안내
# ==========================================

if data is None:

    st.warning(data_error)

    st.info(
        "앱과 같은 폴더에 "
        "`wave+energy+converters.zip` 파일을 넣어 주세요."
    )

    st.stop()


# ==========================================
# 데이터 소개
# ==========================================

st.subheader("데이터 소개")

st.write(
    """
    이 데이터는 Adelaide, Perth, Sydney, Tasmania 지역의
    파력 발전 장치 배치와 발전량을 기록한 자료입니다.

    각 행에는 16개 장치의 X 좌표와 Y 좌표가 있으며
    각 장치의 개별 발전량과 전체 발전량도 포함되어 있습니다.

    이 앱에서는 장치의 배치 좌표를 입력 변수로 사용하고
    전체 발전량인 Total Power를 예측합니다.

    개별 발전량 P1부터 P16까지는 전체 발전량과 직접적인
    관계가 있으므로 예측 입력 변수에서 제외합니다.
    """
)


# ==========================================
# 데이터 요약
# ==========================================

total_rows = len(data)

total_columns = len(DATA_COLS)

missing_count = int(
    data[DATA_COLS].isna().sum().sum()
)

region_count = data["지역"].nunique()


# ==========================================
# 요약 카드
# ==========================================

st.subheader("데이터 요약")

metric_columns = st.columns(4)

metric_columns[0].metric(
    "전체 데이터 수",
    f"{total_rows:,}"
)

metric_columns[1].metric(
    "열 개수",
    f"{total_columns:,}"
)

metric_columns[2].metric(
    "결측값 수",
    f"{missing_count:,}"
)

metric_columns[3].metric(
    "지역 수",
    f"{region_count:,}"
)


st.divider()


# ==========================================
# 열 정보 표
# ==========================================

st.subheader("데이터 열 정보")

saved_meanings = records.get(
    "meanings",
    {}
)

metadata_rows = []

for column in DATA_COLS:

    series = data[column]

    if pd.api.types.is_numeric_dtype(series):

        value_type = "숫자형"

    else:

        value_type = "문자형"

    metadata_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": saved_meanings.get(
                column,
                KOREAN_MEANINGS.get(
                    column,
                    ""
                )
            ),
            "값의 종류": value_type,
            "빈 값 개수": int(
                series.isna().sum()
            ),
        }
    )


metadata_frame = pd.DataFrame(
    metadata_rows
)


# ==========================================
# 열 정보 수정
# ==========================================

edited_metadata = st.data_editor(
    metadata_frame,
    use_container_width=True,
    hide_index=True,
    column_config={
        "열 이름": st.column_config.TextColumn(
            disabled=True
        ),
        "우리말 뜻": st.column_config.TextColumn(
            help="열의 의미를 수정할 수 있습니다.",
            width="large"
        ),
        "값의 종류": st.column_config.TextColumn(
            disabled=True
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            disabled=True
        ),
    },
    key="metadata_editor"
)


# ==========================================
# 데이터 출처 입력
# ==========================================

source_text = st.text_input(
    "데이터 출처",
    value=str(
        records.get(
            "source",
            "wave+energy+converters.zip"
        )
    )
)


# ==========================================
# 기록 저장
# ==========================================

if st.button(
    "우리말 뜻과 출처 저장",
    type="primary"
):

    updated_records = {
        "source": source_text,
        "meanings": {
            row["열 이름"]: row["우리말 뜻"]
            for _, row in edited_metadata.iterrows()
        }
    }

    if save_records(updated_records):

        st.success(
            "내_기록 폴더의 JSON 파일에 저장했습니다."
        )

        st.rerun()


st.caption(
    f"현재 출처 기록: {records.get('source', '미입력')}"
)


# ==========================================
# 데이터 처음 다섯 줄
# ==========================================

st.subheader("데이터 처음 다섯 줄")

st.dataframe(
    data[DATA_COLS].head(5),
    use_container_width=True
)


# ==========================================
# 지역별 발전량 그래프
# ==========================================

st.subheader("지역별 전체 발전량 분포")

region_plot = px.box(
    data,
    x="지역",
    y="Total Power",
    points=False,
    title="지역별 Total Power 분포",
    labels={
        "지역": "지역",
        "Total Power": "전체 발전량"
    }
)

st.plotly_chart(
    region_plot,
    use_container_width=True
)


# ==========================================
# 회귀 모델
# ==========================================

st.divider()

st.header("전체 발전량 예측")

st.write(
    """
    16개 장치의 X 좌표와 Y 좌표를 입력해
    전체 발전량을 예측합니다.

    모든 지역의 데이터를 합쳐 학습하고
    테스트 데이터로 모델 성능을 평가합니다.
    """
)


# ==========================================
# 모델 학습 데이터 준비
# ==========================================

model_data = data[
    FEATURE_COLS + ["Total Power"]
].dropna()


if len(model_data) < 10:

    st.error(
        "회귀 모델을 학습하기 위한 데이터가 충분하지 않습니다."
    )

    st.stop()


X = model_data[FEATURE_COLS]

y = model_data["Total Power"]


# ==========================================
# 학습 데이터와 테스트 데이터 분리
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 선형 회귀 모델
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================
# 모델 평가
# ==========================================

test_predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    test_predictions
)

mse = mean_squared_error(
    y_test,
    test_predictions
)


# ==========================================
# 평가 지표 표시
# ==========================================

st.subheader("모델 평가")

score_columns = st.columns(2)

score_columns[0].metric(
    "MAE",
    f"{mae:,.4f}"
)

score_columns[1].metric(
    "MSE",
    f"{mse:,.4f}"
)


st.caption(
    f"학습 데이터 {len(X_train):,}행 · "
    f"테스트 데이터 {len(X_test):,}행"
)


# ==========================================
# 실제값과 예측값 비교
# ==========================================

with st.expander(
    "테스트 데이터 실제값과 예측값 비교"
):

    comparison = pd.DataFrame(
        {
            "실제 전체 발전량": y_test.iloc[:20].to_numpy(),
            "예측 전체 발전량": test_predictions[:20],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )


    comparison_plot = px.scatter(
        comparison,
        x="실제 전체 발전량",
        y="예측 전체 발전량",
        title="실제 발전량과 예측 발전량 비교",
        labels={
            "실제 전체 발전량": "실제 전체 발전량",
            "예측 전체 발전량": "예측 전체 발전량"
        }
    )

    st.plotly_chart(
        comparison_plot,
        use_container_width=True
    )


# ==========================================
# 장치 배치 좌표 입력
# ==========================================

st.subheader("장치 배치 좌표 입력")

st.write(
    """
    각 장치의 X 좌표와 Y 좌표를 입력하세요.

    기본값은 데이터의 첫 번째 행에 있는 좌표입니다.
    """
)


# ==========================================
# 기본 좌표 설정
# ==========================================

default_row = data.iloc[0]

coordinate_values = {}

for i in range(1, 17):

    coordinate_values[f"X{i}"] = float(
        default_row[f"X{i}"]
    )

    coordinate_values[f"Y{i}"] = float(
        default_row[f"Y{i}"]
    )


# ==========================================
# 좌표 입력 폼
# ==========================================

with st.form("coordinate_form"):

    st.markdown("**X 좌표**")

    x_columns = st.columns(4)

    for i in range(1, 17):

        with x_columns[(i - 1) % 4]:

            coordinate_values[f"X{i}"] = st.number_input(
                f"X{i}",
                value=coordinate_values[f"X{i}"],
                key=f"input_X{i}"
            )


    st.markdown("**Y 좌표**")

    y_columns = st.columns(4)

    for i in range(1, 17):

        with y_columns[(i - 1) % 4]:

            coordinate_values[f"Y{i}"] = st.number_input(
                f"Y{i}",
                value=coordinate_values[f"Y{i}"],
                key=f"input_Y{i}"
            )


    predict_clicked = st.form_submit_button(
        "전체 발전량 예측",
        type="primary"
    )


# ==========================================
# 발전량 예측
# ==========================================

if predict_clicked:

    input_frame = pd.DataFrame(
        [
            [
                coordinate_values[column]
                for column in FEATURE_COLS
            ]
        ],
        columns=FEATURE_COLS
    )


    predicted_power = float(
        model.predict(input_frame)[0]
    )


    st.success(
        f"예측 전체 발전량: {predicted_power:,.4f}"
    )


    # ==========================================
    # 입력 좌표 시각화
    # ==========================================

    coordinate_plot_data = pd.DataFrame(
        {
            "X 좌표": [
                coordinate_values[column]
                for column in X_COLS
            ],
            "Y 좌표": [
                coordinate_values[column]
                for column in Y_COLS
            ],
            "장치": [
                f"장치 {i}"
                for i in range(1, 17)
            ]
        }
    )


    layout_plot = px.scatter(
        coordinate_plot_data,
        x="X 좌표",
        y="Y 좌표",
        text="장치",
        title="입력한 파력 발전 장치 배치",
        labels={
            "X 좌표": "X 좌표",
            "Y 좌표": "Y 좌표"
        }
    )


    layout_plot.update_traces(
        textposition="top center"
    )


    st.plotly_chart(
        layout_plot,
        use_container_width=True
    )
