import json
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


APP_TITLE = "파력 발전량 예측 실습실"
BASE_DIR = Path(__file__).resolve().parent
ZIP_NAME = "wave+energy+converters.zip"
ZIP_PATH = BASE_DIR / ZIP_NAME
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_PATH = RECORD_DIR / "파력발전_데이터_기록.json"

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

MEANINGS = {
    **{f"X{i}": f"{i}번 장치의 X 좌표" for i in range(1, 17)},
    **{f"Y{i}": f"{i}번 장치의 Y 좌표" for i in range(1, 17)},
    **{f"P{i}": f"{i}번 장치의 개별 발전량" for i in range(1, 17)},
    "Total Power": "전체 발전량",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌊",
    layout="wide"
)


def load_records():

    default = {
        "source": ZIP_NAME,
        "meanings": MEANINGS.copy(),
    }

    if RECORD_PATH.exists():

        try:

            with RECORD_PATH.open(
                "r",
                encoding="utf-8"
            ) as f:

                saved = json.load(f)

            if isinstance(saved, dict):

                default["source"] = saved.get(
                    "source",
                    default["source"]
                )

                default["meanings"].update(
                    saved.get("meanings", {})
                )

        except (
            OSError,
            json.JSONDecodeError,
            TypeError
        ):

            pass

    return default


def save_records(records):

    try:

        RECORD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with RECORD_PATH.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2
            )

        return True

    except OSError as e:

        st.error(
            f"기록 파일 저장에 실패했습니다: {e}"
        )

        return False


def locate_zip():

    if ZIP_PATH.exists():

        return ZIP_PATH

    candidates = list(
        BASE_DIR.glob("*.zip")
    )

    for candidate in candidates:

        normalized = (
            candidate.name
            .lower()
            .replace(" ", "")
        )

        if (
            "wave" in normalized
            and "energy" in normalized
        ):

            return candidate

    return None


@st.cache_data(
    show_spinner="파력 발전 데이터를 불러오는 중..."
)
def load_data(zip_path, zip_mtime):

    del zip_mtime

    frames = []

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zf:

        names = zf.namelist()

        for region, expected_filename in REGION_FILES.items():

            matches = [
                name
                for name in names
                if Path(name).name.lower()
                == expected_filename.lower()
            ]

            if not matches:

                raise FileNotFoundError(
                    f"ZIP 내부에서 {expected_filename} 파일을 찾지 못했습니다."
                )

            with zf.open(matches[0]) as file:

                frame = pd.read_csv(
                    file,
                    header=None
                )

            if frame.shape[1] != 49:

                raise ValueError(
                    f"{region} CSV 열 개수가 49개가 아닙니다. "
                    f"현재 {frame.shape[1]}개입니다."
                )

            frame.columns = DATA_COLS

            frame["지역"] = region

            frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    return combined


# ==========================================
# 앱 제목
# ==========================================

st.title(
    "🌊 파력 발전 장치 배치 좌표를 이용한 전체 발전량 예측"
)

st.caption(
    "파력 발전 장치의 배치 좌표와 발전량 데이터를 탐색하고 "
    "선형 회귀로 전체 발전량을 예측합니다."
)


# ==========================================
# 데이터 ZIP 확인
# ==========================================

zip_file = locate_zip()

if zip_file is None:

    st.warning(
        "데이터 ZIP 파일을 찾을 수 없습니다."
    )

    st.info(
        f"`{ZIP_NAME}` 파일을 `main.py`와 같은 폴더에 넣고 "
        "GitHub 저장소에도 함께 올려 주세요."
    )

    st.stop()


# ==========================================
# 데이터 불러오기
# ==========================================

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
        f"데이터를 읽지 못했습니다: {e}"
    )

    st.info(
        "ZIP 파일 내부에 Adelaide, Perth, Sydney, "
        "Tasmania의 CSV 4개가 있는지 확인해 주세요."
    )

    st.stop()


records = load_records()


# ==========================================
# 데이터 소개
# ==========================================

st.subheader("데이터 소개")

st.write(
    "이 데이터는 Adelaide, Perth, Sydney, Tasmania 지역의 파력 발전 장치 배치와 "
    "발전량을 기록한 자료입니다. 각 행에는 16개 장치의 X·Y 좌표와 장치별 발전량 "
    "P1~P16, 전체 발전량 Total Power가 들어 있습니다. 이 앱은 X·Y 좌표 32개를 "
    "입력 변수로 사용해 Total Power를 예측합니다. P1~P16은 전체 발전량과 직접 "
    "연결되어 있어 예측 입력 변수에서 제외합니다."
)


# ==========================================
# 데이터 요약
# ==========================================

numeric_data = data[DATA_COLS]

missing_count = int(
    numeric_data.isna().sum().sum()
)

metric_cols = st.columns(4)

metric_cols[0].metric(
    "전체 데이터 수",
    f"{len(data):,}"
)

metric_cols[1].metric(
    "열 개수",
    f"{len(DATA_COLS):,}"
)

metric_cols[2].metric(
    "결측값 수",
    f"{missing_count:,}"
)

metric_cols[3].metric(
    "지역 수",
    f"{data['지역'].nunique():,}"
)


st.divider()


# ==========================================
# 열 정보
# ==========================================

st.subheader("열 정보 및 우리말 뜻")

metadata = []

saved_meanings = records.get(
    "meanings",
    {}
)

for col in DATA_COLS:

    s = data[col]

    metadata.append(
        {
            "열 이름": col,

            "우리말 뜻": saved_meanings.get(
                col,
                MEANINGS.get(col, "")
            ),

            "값의 종류": (
                "숫자형"
                if pd.api.types.is_numeric_dtype(s)
                else "문자형"
            ),

            "빈 값 개수": int(
                s.isna().sum()
            ),
        }
    )


metadata_df = pd.DataFrame(
    metadata
)


edited_metadata = st.data_editor(
    metadata_df,
    hide_index=True,
    use_container_width=True,

    column_config={
        "열 이름": st.column_config.TextColumn(
            disabled=True
        ),

        "우리말 뜻": st.column_config.TextColumn(
            help="열의 의미를 직접 수정할 수 있습니다.",
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
# 출처 입력
# ==========================================

source = st.text_input(
    "데이터 출처",

    value=str(
        records.get(
            "source",
            ZIP_NAME
        )
    )
)


# ==========================================
# JSON 저장
# ==========================================

if st.button(
    "우리말 뜻과 출처 저장",
    type="primary"
):

    updated = {

        "source": source,

        "meanings": {

            row["열 이름"]: row["우리말 뜻"]

            for _, row in edited_metadata.iterrows()

        }

    }

    if save_records(updated):

        st.success(
            "우리말 뜻과 출처를 내_기록 폴더의 JSON 파일에 저장했습니다."
        )

        st.rerun()


st.caption(
    f"현재 기록된 출처: {records.get('source', ZIP_NAME)}"
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
# 지역별 전체 발전량 분포
# ==========================================

st.subheader("지역별 전체 발전량 분포")

region_fig = px.box(
    data,

    x="지역",

    y="Total Power",

    title="지역별 Total Power 분포",

    labels={
        "지역": "지역",
        "Total Power": "전체 발전량"
    },
)


# 박스플롯의 세로 길이를 늘려 분포를 더 크게 표시

region_fig.update_layout(
    height=800,

    margin=dict(
        l=50,
        r=30,
        t=80,
        b=50
    ),
)


st.plotly_chart(
    region_fig,
    use_container_width=True
)


# ==========================================
# 회귀 모델
# ==========================================

st.divider()

st.header("전체 발전량 예측")

st.write(
    "16개 장치의 X 좌표와 Y 좌표를 입력하면 "
    "Total Power를 예측합니다."
)


# ==========================================
# 모델 데이터 준비
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
# 예측
# ==========================================

test_pred = model.predict(
    X_test
)


# ==========================================
# 모델 평가
# ==========================================

mae = mean_absolute_error(
    y_test,
    test_pred
)

mse = mean_squared_error(
    y_test,
    test_pred
)


st.subheader("모델 평가")

scores = st.columns(2)

scores[0].metric(
    "MAE",
    f"{mae:,.4f}"
)

scores[1].metric(
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

            "예측 전체 발전량": test_pred[:20],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    comparison_fig = px.scatter(
        comparison,

        x="실제 전체 발전량",

        y="예측 전체 발전량",

        title="실제 발전량과 예측 발전량 비교",
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )


# ==========================================
# 좌표 입력
# ==========================================

st.subheader("장치 배치 좌표 입력")

st.write(
    "기본값은 데이터의 첫 번째 행에 있는 장치 좌표입니다."
)


default_row = data.iloc[0]

coord = {
    col: float(default_row[col])
    for col in FEATURE_COLS
}


# ==========================================
# 좌표 입력 폼
# ==========================================

with st.form("coordinate_form"):

    st.markdown("**X 좌표**")

    x_cols = st.columns(4)

    for i, col in enumerate(X_COLS):

        with x_cols[i % 4]:

            coord[col] = st.number_input(
                col,

                value=coord[col],

                key=f"coord_{col}"
            )


    st.markdown("**Y 좌표**")

    y_cols = st.columns(4)

    for i, col in enumerate(Y_COLS):

        with y_cols[i % 4]:

            coord[col] = st.number_input(
                col,

                value=coord[col],

                key=f"coord_{col}"
            )


    predict = st.form_submit_button(
        "전체 발전량 예측",
        type="primary"
    )


# ==========================================
# 발전량 예측
# ==========================================

if predict:

    input_df = pd.DataFrame(
        [
            [
                coord[c]
                for c in FEATURE_COLS
            ]
        ],

        columns=FEATURE_COLS
    )


    prediction = float(
        model.predict(input_df)[0]
    )


    st.success(
        f"예측 전체 발전량: {prediction:,.4f}"
    )


    # ==========================================
    # 장치 배치 그래프
    # ==========================================

    layout_df = pd.DataFrame(
        {
            "X 좌표": [
                coord[c]
                for c in X_COLS
            ],

            "Y 좌표": [
                coord[c]
                for c in Y_COLS
            ],

            "장치": [
                f"장치 {i}"
                for i in range(1, 17)
            ],
        }
    )


    layout_fig = px.scatter(
        layout_df,

        x="X 좌표",

        y="Y 좌표",

        text="장치",

        title="입력한 파력 발전 장치 배치",
    )


    layout_fig.update_traces(
        textposition="top center"
    )


    st.plotly_chart(
        layout_fig,
        use_container_width=True
    )
