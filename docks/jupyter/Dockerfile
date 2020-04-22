FROM python

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/usr/src/app

COPY . .

RUN pip install jupyterlab

CMD [ "python"]
