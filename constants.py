CATEGORY_CODES = {
    'movies_dvd': 'Кино - DVD',
    'music_cd_local': 'Музыка - CD локального производства',
    'games_ps_3': 'Игры - PS3',
    'games_xbox_360': 'Игры - XBOX 360',
    'presents_softtoy': 'Подарки - Мягкие игрушки',
    'presents_boardgame': 'Подарки - Настольные игры (компактные)'
}
TARGET_COLUMN = 'item_cnt_day'
CATEGORY_COLUMN = 'category'
DATE_COLUMN = 'date'
DISPLAY_COLUMNS = [CATEGORY_COLUMN, DATE_COLUMN, TARGET_COLUMN]
EXOG_COLUMNS = [
    'day_cos',
    'day_sin',
    'month_cos',
    'month_sin',
    'day_of_week_cos',
    'day_of_week_sin'
]

TOTAL_COLUMNS = [
    *EXOG_COLUMNS,
    *DISPLAY_COLUMNS
]
FURTHER_DAYS = 121
SEED = 42

END_TRAIN = '2015-08-31'
END_VALID = '2015-09-30'
END_TEST = '2015-10-31'

LAGS = [1, 2, 3, 5, 6, 7, 14, 15, 16, 21, 22, 23, 28, 29, 30, 31, 118, 119, 120, 121, 122, 180, 181, 182, 183]

NAMES = ['Egor Stroev']
USERNAMES = ['estroev']
