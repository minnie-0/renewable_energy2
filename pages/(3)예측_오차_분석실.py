import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="예측 오차 분석실",
    page_icon="📊",
    layout="wide"
)

st.title("📊 예측 오차 분석실")
st.caption("선형 회귀 모델의 예측 결과와 실제 발전량을 비교하고 성능을 평가합니다.")

DATA_PATH = Path(__file__).resolve().parent.parent / "wave_energy_reduced.zip"

REGIONS = ["Adelaide", "Perth", "Sydney", "Tasmania"]

FEATURE_COLUMNS = (
    [f"X{i}" for i in range(1, 17)]
    + [f"Y{i}" for i in range(1, 17)]
)

TARGET_COLUMN = "Total Power"

COLUMNS = (
    [f"X{i}" for i in range(1, 17)]
    + [f"Y{i}" for i in range(1, 17)]
    + [f"P{i}" for i in range(1, 17)]
    + ["Total Power"]
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}"
        )

    data_list = []

    with zipfile.ZipFile(DATA_PATH, "r") as zip_file:

        for region in REGIONS:

            file_path = f"WECs_DataSet/{region}_Data.csv"

            if file_path not in zip_file.namelist():
                raise FileNotFoundError(
                    f"ZIP 파일 내부에서 {file_path}를 찾을 수 없습니다."
                )

            csv_data = zip_file.read(file_path)

            df = pd.read_csv(
                BytesIO(csv_data),
                header=None,
                names=COLUMNS
            )

            df["지역"] = region

            data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    # 수치형 데이터 변환
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # 결측값 및 무한대 값 제거
    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    return data


try:
    data = load_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 데이터 분할 및 모델 학습
# --------------------------------------------------

# 동일한 결과를 재현하기 위해 random_state 고정
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

X_train = train_data[FEATURE_COLUMNS]
y_train = train_data[TARGET_COLUMN]

X_test = test_data[FEATURE_COLUMNS]
y_test = test_data[TARGET_COLUMN]

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터 예측
predictions = model.predict(X_test)


# --------------------------------------------------
# 예측 결과 및 오차 계산
# --------------------------------------------------

result = test_data[["지역", TARGET_COLUMN]].copy()

result = result.rename(
    columns={TARGET_COLUMN: "실제 발전량"}
)

result["예측 발전량"] = predictions

# 실제 발전량 - 예측 발전량
result["예측 오차"] = (
    result["실제 발전량"] - result["예측 발전량"]
)

result["절대 오차"] = result["예측 오차"].abs()


# --------------------------------------------------
# 전체 모델 평가 지표
# --------------------------------------------------

r2 = r2_score(y_test, predictions)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)


# --------------------------------------------------
# 데이터 기본 정보
# --------------------------------------------------

st.subheader("1. 데이터 및 모델 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 데이터", f"{len(data):,}개")

with col2:
    st.metric("학습 데이터", f"{len(train_data):,}개")

with col3:
    st.metric("테스트 데이터", f"{len(test_data):,}개")

with col4:
    st.metric("사용 독립 변수", "32개")

st.write(
    "독립 변수: 장치 16개의 X좌표와 Y좌표"
)

st.write(
    "종속 변수: 전체 발전량"
)

st.write(
    "모델: 선형 회귀"
)

st.write(
    "데이터 분할: 학습용 80% / 테스트용 20% "
    "(random_state=42)"
)

st.write(
    "※ 개별 장치의 발전량인 P1~P16은 독립 변수에서 제외했습니다."
)


# --------------------------------------------------
# 전체 성능 평가
# --------------------------------------------------

st.divider()

st.subheader("2. 모델 성능 평가")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "결정계수 (R²)",
        f"{r2:.4f}"
    )

with metric2:
    st.metric(
        "평균절대오차 (MAE)",
        f"{mae:,.2f}"
    )

with metric3:
    st.metric(
        "평균제곱근오차 (RMSE)",
        f"{rmse:,.2f}"
    )

with st.expander("평가 지표의 의미 알아보기"):

    st.markdown(
        """
        - **결정계수(R²)**: 실제 발전량의 변화를 모델이 얼마나 설명하는지 나타냅니다. 1에 가까울수록 설명력이 높습니다.
        - **평균절대오차(MAE)**: 실제 발전량과 예측 발전량의 차이를 절댓값으로 바꾼 뒤 평균한 값입니다. 작을수록 좋습니다.
        - **평균제곱근오차(RMSE)**: 예측 오차를 제곱해 평균한 뒤 제곱근을 구한 값입니다. 큰 오차에 더 큰 영향을 받으며 작을수록 좋습니다.
        """
    )


# --------------------------------------------------
# 실제 발전량과 예측 발전량 비교
# --------------------------------------------------

st.divider()

st.subheader("3. 실제 발전량과 예측 발전량 비교")

fig_scatter = go.Figure()

fig_scatter.add_trace(
    go.Scatter(
        x=result["실제 발전량"],
        y=result["예측 발전량"],
        mode="markers",
        name="예측 결과",
        marker=dict(
            size=6,
            opacity=0.5
        ),
        hovertemplate=(
            "실제 발전량: %{x:,.2f}<br>"
            "예측 발전량: %{y:,.2f}"
            "<extra></extra>"
        )
    )
)

# 실제값과 예측값이 같은 경우를 나타내는 기준선
min_value = min(
    result["실제 발전량"].min(),
    result["예측 발전량"].min()
)

max_value = max(
    result["실제 발전량"].max(),
    result["예측 발전량"].max()
)

fig_scatter.add_trace(
    go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="이상적인 예측 기준선",
        line=dict(
            dash="dash"
        )
    )
)

fig_scatter.update_layout(
    xaxis_title="실제 발전량",
    yaxis_title="예측 발전량",
    legend_title="구분",
    height=550,
    hovermode="closest"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

st.caption(
    "점들이 기준선에 가까울수록 실제 발전량과 예측 발전량의 차이가 작다는 의미입니다."
)


# --------------------------------------------------
# 예측 오차 분포
# --------------------------------------------------

st.divider()

st.subheader("4. 예측 오차 분포")

col1, col2 = st.columns(2)

with col1:

    fig_hist = px.histogram(
        result,
        x="예측 오차",
        nbins=40,
        title="예측 오차 분포",
        labels={
            "예측 오차": "예측 오차",
            "count": "데이터 개수"
        }
    )

    fig_hist.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="오차 0"
    )

    fig_hist.update_layout(
        xaxis_title="예측 오차 (실제 발전량 - 예측 발전량)",
        yaxis_title="데이터 개수",
        height=450
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

with col2:

    fig_residual = px.scatter(
        result,
        x="예측 발전량",
        y="예측 오차",
        title="예측 발전량에 따른 오차",
        color="지역",
        labels={
            "예측 발전량": "예측 발전량",
            "예측 오차": "예측 오차",
            "지역": "지역"
        },
        opacity=0.6
    )

    fig_residual.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_residual.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True
    )

st.caption(
    "예측 오차가 0에 가까울수록 정확한 예측이며 양수는 실제 발전량이 더 큰 경우, 음수는 예측 발전량이 더 큰 경우입니다."
)


# --------------------------------------------------
# 예측 오차가 큰 데이터 확인
# --------------------------------------------------

st.divider()

st.subheader("5. 예측 오차가 큰 데이터 사례")

st.write(
    "절대 오차가 큰 순서대로 테스트 데이터의 사례를 확인할 수 있습니다."
)

top_n = st.slider(
    "표시할 데이터 개수",
    min_value=5,
    max_value=30,
    value=10
)

largest_errors = result.sort_values(
    by="절대 오차",
    ascending=False
).head(top_n)

st.dataframe(
    largest_errors[
        [
            "지역",
            "실제 발전량",
            "예측 발전량",
            "예측 오차",
            "절대 오차"
        ]
    ].style.format({
        "실제 발전량": "{:,.2f}",
        "예측 발전량": "{:,.2f}",
        "예측 오차": "{:,.2f}",
        "절대 오차": "{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 지역별 예측 오차 비교
# --------------------------------------------------

st.divider()

st.subheader("6. 지역별 예측 오차 비교")

region_results = []

for region in REGIONS:

    region_data = result[
        result["지역"] == region
    ]

    if len(region_data) == 0:
        continue

    region_actual = region_data["실제 발전량"]
    region_predicted = region_data["예측 발전량"]

    region_mae = mean_absolute_error(
        region_actual,
        region_predicted
    )

    region_rmse = np.sqrt(
        mean_squared_error(
            region_actual,
            region_predicted
        )
    )

    if len(region_data) >= 2:
        region_r2 = r2_score(
            region_actual,
            region_predicted
        )
    else:
        region_r2 = np.nan

    region_results.append({
        "지역": region,
        "테스트 데이터 개수": len(region_data),
        "결정계수 (R²)": region_r2,
        "평균절대오차 (MAE)": region_mae,
        "평균제곱근오차 (RMSE)": region_rmse
    })


region_df = pd.DataFrame(region_results)

st.write("지역별 평가 지표")

st.dataframe(
    region_df.style.format({
        "결정계수 (R²)": "{:.4f}",
        "평균절대오차 (MAE)": "{:,.2f}",
        "평균제곱근오차 (RMSE)": "{:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# 지역별 MAE와 RMSE 비교 그래프
region_long = region_df.melt(
    id_vars="지역",
    value_vars=[
        "평균절대오차 (MAE)",
        "평균제곱근오차 (RMSE)"
    ],
    var_name="평가 지표",
    value_name="오차"
)

fig_region = px.bar(
    region_long,
    x="지역",
    y="오차",
    color="평가 지표",
    barmode="group",
    title="지역별 예측 오차 비교",
    labels={
        "지역": "지역",
        "오차": "오차 크기",
        "평가 지표": "평가 지표"
    },
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_region.update_layout(
    height=500,
    xaxis_title="지역",
    yaxis_title="오차 크기",
    legend_title="평가 지표"
)

st.plotly_chart(
    fig_region,
    use_container_width=True
)


# --------------------------------------------------
# 마무리
# --------------------------------------------------

st.divider()

st.subheader("7. 분석 정리")

st.write(
    f"테스트 데이터 {len(test_data):,}개를 이용해 모델의 예측 성능을 평가했습니다."
)

st.write(
    f"전체 데이터의 평균절대오차는 {mae:,.2f}이고 "
    f"평균제곱근오차는 {rmse:,.2f}입니다."
)

st.write(
    "실제 발전량과 예측 발전량의 산점도 및 오차 분포를 통해 "
    "모델의 예측 경향을 확인할 수 있습니다."
)

st.write(
    "지역별 평가 지표를 비교하면 지역에 따라 예측 오차가 달라지는지도 확인할 수 있습니다."
)

st.caption(
    "※ 이 페이지의 평가 결과는 학습에 사용하지 않은 테스트 데이터를 기준으로 계산했습니다."
)
