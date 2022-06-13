import math
import pickle
import calendar
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import psycopg2
import streamlit_authenticator as stauth
import plotly.graph_objects as go
from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster, grid_search_forecaster

from constants import (
    CATEGORY_CODES,
    CATEGORY_COLUMN,
    DATE_COLUMN,
    DISPLAY_COLUMNS,
    END_TRAIN,
    END_VALID,
    END_TEST,
    LAGS,
    SEED,
    TARGET_COLUMN,
    NAMES,
    USERNAMES,
)

HASHED_PWDS_PATH = 'hashed_pwd.pickle'

months_map = {
    11: 'Ноябрь',
    12: 'Декабрь',
    1: 'Январь',
    2: 'Февраль'
}


def get_db_connection():
    conn = psycopg2.connect(
        host='postgres', database='flask_db', user='estroev', password='estroev'
    )
    return conn


def get_db_col_values(col_name):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f'SELECT {col_name} FROM users;')
    result = cur.fetchall()
    return [i[0] for i in result]


for key in CATEGORY_CODES.keys():
    if key not in st.session_state:
        st.session_state[key] = None
if 'pressed_first_button' not in st.session_state:
    st.session_state.pressed_first_button = False
if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = None
if 'name' not in st.session_state:
    st.session_state.name = None
if 'username' not in st.session_state:
    st.session_state.username = None

base_models = {
    'Линейная регрессия (ElasticNet)': ElasticNet(random_state=SEED),
    'CatBoost': CatBoostRegressor(bootstrap_type='Bernoulli', random_state=SEED, silent=True, thread_count=-1),
    'Случайный лес': RandomForestRegressor(random_state=SEED, n_jobs=-1),
#    'LightGBM': LGBMRegressor(random_state=SEED, n_jobs=-1)
}

param_grid = {
    'Линейная регрессия (ElasticNet)': {
        'alpha': [
            0.25
            #0.5
            #0.75
            #1.0,
            #1.25,
            #1.5,
            #1.75,
            #2.0
        ],
        'l1_ratio': [
            0
            #0.15
            #0.3
            #0.45,
            #0.6,
            #0.75,
            #0.9,
            #1.0
        ]
    },
    'CatBoost': {
        'n_estimators': [
            100
            #200
            #300,
            #400
        ],
        'depth': [
            3
            #6
            #9
        ],
        'l2_leaf_reg': [
            1
            #3
            #5,
            #7
        ],
        'subsample': [
            0.1
            #0.3,
            #0.5
        ],
        'learning_rate': np.linspace(0.05, 0.2, num=5)
    },
#    'LightGBM' : {
#        'num_iterations': [200, 300, 400],
#        'max_depth': [6, 9, 13],
#        'lambda_l1': np.logspace(1e-8, 10, num=5),
#        'lambda_l2': np.logspace(1e-8, 10, num=5),
#        'learning_rate': np.linspace(0.05, 0.2, num=5)
#    },
    'Случайный лес': {
        'n_estimators': [
            100
            #200
            #300,
            #400
        ],
        'max_depth': [
            3
            #6,
            #9,
            #None
        ],
        'criterion':  [
            'squared_error',
            'absolute_error',
            'poisson'
        ]
    }
}

metrics = {
    'MAE': mean_absolute_error,
    'MAPE': mean_absolute_percentage_error,
    'R2': r2_score
}


@st.cache(show_spinner=False, ttl=300)
def load_passwords(path):
    with open(path, 'rb') as file:
        hashed_passwords = pickle.load(file)
    return hashed_passwords


def train_model(code):
    data_category = data[data[CATEGORY_COLUMN] == CATEGORY_CODES[code]].set_index(DATE_COLUMN)
    data_category = data_category.asfreq('d')
    X, y = data_category.drop(columns=[CATEGORY_COLUMN, TARGET_COLUMN]), data_category[TARGET_COLUMN]

    models = dict()
    scores = dict()
    predictions = dict()
    for model_label, model in base_models.items():

        forecaster = ForecasterAutoreg(
            regressor=model,
            lags=LAGS
        )
        _ = grid_search_forecaster(
            forecaster=forecaster,
            y=y.loc[:END_VALID],
            exog=X.loc[:END_VALID],
            param_grid=param_grid[model_label],
            fixed_train_size=False,
            steps=len(y.loc[END_TRAIN:END_VALID]) - 1,
            refit=True,
            metric='mean_absolute_percentage_error',
            initial_train_size=len(y.loc[:END_TRAIN]),
            return_best=True,
            verbose=False
        )

        def backtest(metric):
            return backtesting_forecaster(
                forecaster=forecaster,
                y=y.loc[:END_TEST],
                exog=X.loc[:END_TEST],
                initial_train_size=len(y.loc[:END_VALID]),
                steps=len(y.loc[END_VALID:END_TEST]) - 1,
                fixed_train_size=True,
                refit=True,
                metric=metric,
                verbose=False
            )

        scores[model_label] = dict()
        for scoring_label, scoring_function in metrics.items():
            if scoring_label != 'MAPE':
                scores[model_label][scoring_label] = backtest(scoring_function)[0]
            else:
                scores[model_label][scoring_label], predictions[model_label] = backtest(scoring_function)
                predictions[model_label] = predictions[model_label]['pred']
                predictions[model_label].index = y.loc[END_VALID:END_TEST].iloc[1:].index

        forecaster.fit(y=y.loc[:END_TEST], exog=X.loc[:END_TEST])
        models[model_label] = forecaster

    scores = pd.DataFrame.from_dict(scores).T

    return models, scores, predictions, X, y


if __name__ == '__main__':

    # hashed_passwords = load_passwords(HASHED_PWDS_PATH)
    names = get_db_col_values('name')
    surnames = get_db_col_values('surname')
    fullnames = [' '.join(i) for i in zip(names, surnames)]

    authenticator = stauth.Authenticate(
        fullnames,
        get_db_col_values('username'),
        get_db_col_values('pwd_hash'),
        'estroev_cookie_name',
        'estroev_signature_key',
        cookie_expiry_days=10
    )
    st.session_state['name'], st.session_state['authentication_status'], st.session_state['username'] = authenticator.login('Login', 'main')
    if st.session_state['authentication_status']:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome {st.session_state["name"]}')
    elif st.session_state['authentication_status'] == False:
        st.error('Username/password is incorrect')
        st.stop()
    elif st.session_state['authentication_status'] is None:
        st.warning('Please enter your username and password')
        st.stop()

    st.title('Прогнозирование продаж')

    _, col, _ = st.columns([1, 6, 1])
    with col:

        data = None
        uploader = st.file_uploader(label='Данные по продажам', key='uploader')

        if uploader is not None:
            data = pd.read_csv(uploader, parse_dates=['date'], sep='\t', index_col=0)
            uploader.seek(0)
            file_container = st.expander(f'Проверить содержимое файла')
            file_container.write(data[DISPLAY_COLUMNS])

        if data is None:
            st.info(f'👆 Для работы загрузите файл с данными продаж')
            st.stop()

    st.write('')
    st.header('Обучение модели')

    with st.form(key='train_form'):

        cat_code = st.selectbox(
            label='Выберите категорию',
            options=list(CATEGORY_CODES.keys()),
            index=0,
            format_func=lambda code: CATEGORY_CODES[code],
        )
        run_model_train = st.form_submit_button('Обучить модель')

    st.write('')
    if run_model_train or st.session_state.pressed_first_button:
        st.session_state.pressed_first_button = True
        if st.session_state[cat_code] is None:
            result = dict()
            with st.spinner('Обучение модели ... '):
                result['models'], result['scores'], result['predictions'], result['X'], result['y'] = train_model(cat_code)
            st.write('Тестовые показатели моделей')
            st.table(result['scores'])
            best_model_label = result['scores']['MAPE'].idxmin()
            st.write(f'Лучшая модель согласно показателю MAPE: `{best_model_label}`')

            result['best_model_label'] = best_model_label
            st.session_state[cat_code] = result
        else:
            st.write('Тестовые показатели моделей')
            st.table(st.session_state[cat_code]['scores'])
            st.write(f'Лучшая модель согласно показателю MAPE: `{st.session_state[cat_code]["best_model_label"]}`')

        model_options = list(st.session_state[cat_code]['models'].keys())
        selected_model_labels = st.multiselect(
            label='Модели',
            options=model_options,
            default=model_options[:3],
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state[cat_code]['y'].loc[:END_VALID].index,
            y=st.session_state[cat_code]['y'].loc[:END_VALID],
            name='Спрос (train)'
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state[cat_code]['y'].loc[END_VALID:END_TEST].index,
            y=st.session_state[cat_code]['y'].loc[END_VALID:END_TEST],
            name='Спрос (test)'
        ))
        for model_label in selected_model_labels:
            prediction = st.session_state[cat_code]['predictions'][model_label]
            fig.add_trace(go.Scatter(
                x=prediction.index,
                y=prediction,
                name=model_label
            ))
        fig.update_traces(hovertemplate='<br>'.join(['Дата: %{x}', 'Количество: %{y}']))
        fig.update_xaxes(dtick='M1', tickformat='%b\n%Y')
        fig.update_layout(
            title_text=CATEGORY_CODES[cat_code],
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title='Дата',
            yaxis_title='Количество',
        )
        st.plotly_chart(fig)

    else:
        st.stop()

    st.write('')
    st.header('Прогноз продаж при помощи модели')

    month = st.selectbox(
        label='Выберите месяц',
        options=list(months_map.keys()),
        index=0,
        format_func=lambda month: months_map[month],
    )

    year = 2015 if month in [11, 12] else 2016
    st.write(f'Месяц: `{months_map[month]}` Год: `{year}`')
    month_length = calendar.monthrange(year, month)[1]
    month_last_day = date(year, month, month_length)

    delta = month_last_day - datetime.strptime(END_TEST, '%Y-%m-%d').date()
    model = st.session_state[cat_code]['models'][st.session_state[cat_code]['best_model_label']]
    forecast = model.predict(
        delta.days,
        exog=st.session_state[cat_code]['X'].loc[END_TEST:].iloc[1:delta.days+1]
    )
    forecast.index = pd.date_range(
        (datetime.strptime(END_TEST, '%Y-%m-%d') + timedelta(days=1)).date(),
        month_last_day,
        freq='D',
    )
    month_sum = sum(forecast[-month_length:])
    st.write(f'Суммарное количество заказов за {months_map[month]}: `{math.ceil(month_sum)}`')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state[cat_code]['y'].loc[:END_TEST].index,
        y=st.session_state[cat_code]['y'].loc[:END_TEST],
        name='Исторический спрос'
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast,
        name=f'Предсказание спроса'
    ))
    fig.update_traces(hovertemplate='<br>'.join(['Дата: %{x}', 'Количество: %{y}']))
    fig.update_xaxes(dtick='M1', tickformat='%b\n%Y')
    fig.update_layout(
        title_text=f'{CATEGORY_CODES[cat_code]}. Предсказание спроса с помощью модели `{st.session_state[cat_code]["best_model_label"]}`',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Дата',
        yaxis_title='Количество',
    )
    st.plotly_chart(fig)
