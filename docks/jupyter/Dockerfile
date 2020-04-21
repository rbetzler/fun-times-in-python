FROM python:3

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install jupyterlab

RUN pip install graphviz

ENV PYTHONPATH=/usr/src/app:$PYTHONPATH

WORKDIR /usr/src/app

# Set the default command to python3
CMD ["python"]
