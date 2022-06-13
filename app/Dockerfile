FROM python:3.8-slim-buster
WORKDIR /app
ENV FLASK_APP=flask_app.py
ENV FLASK_RUN_HOST=0.0.0.0
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ADD . .
EXPOSE 5000 8501
CMD ["flask", "run"]