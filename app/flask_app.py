import os
from datetime import datetime

import psycopg2
from flask import Flask, redirect, render_template, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, EqualTo
from streamlit_authenticator import Hasher


app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


def get_db_connection():
    conn = psycopg2.connect(
        host='postgres', port=5432, database='flask_db', user='estroev', password='estroev'
    )
    return conn


class RegistrationForm(FlaskForm):
    username = StringField('Пользователь', validators =[DataRequired()])
    name = StringField('Имя', validators=[DataRequired()])
    surname = StringField('Фамилия', validators=[DataRequired()])
    password1 = PasswordField('Пароль', validators = [DataRequired()])
    password2 = PasswordField('Подтверждение пароля', validators = [DataRequired(), EqualTo('password1')])
    submit = SubmitField('Регистрация')


@app.route('/')
@app.route('/index')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users;')
    users = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('index.html', users=users)


@app.route('/register', methods = ['POST','GET'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        conn = get_db_connection()
        cur = conn.cursor()
        username = form.username.data
        name = form.name.data
        surname = form.surname.data
        date_added = datetime.now().strftime('%Y-%m-%d')
        pwd_hash = Hasher([form.password1.data]).generate()[0]
        cur.execute(
            'INSERT INTO users (name, surname, username, pwd_hash, date_added) VALUES (%s, %s, %s, %s, %s);',
            (name, surname, username, pwd_hash, date_added)
        )
        conn.commit()
        cur.close()
        conn.close()
        return redirect(url_for('index'))
    return render_template('registration.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
