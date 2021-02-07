FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/usr/src/app

ENV DBT_PROFILES_DIR=/usr/src/app/dbt
ENV DBT_PROJECT_DIR=/usr/src/app/dbt

COPY . .

CMD [ "python"]
