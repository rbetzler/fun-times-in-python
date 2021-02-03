## Python
* Build python docker container, from `docks/python`
  ```
  sudo docker build . --tag py-dw-stocks
  ```
* Initialize container
  * Linux
    ```
    sudo docker run -it --name py-temp -v /media/nautilus/fun-times-in-python:/usr/src/app --network bridge py-dw-stocks
    ```
  * Mac
    ```
    docker run -it --name py-temp -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app --network local-network py-dw-stocks
    ```
* Run a python script from root
  ```
  docker exec py-temp python /home/utilities/test_dag_script.py
  ```
* Enter container
  ```
  docker run -it --network bridge -v /media/nautilus/fun-times-in-python:/usr/src/app py-dw-stocks bash
  ```

## Postgres
* Pull docker image, since gpg keys are a pain locally
  ```
  sudo docker pull postgres
  ```
* Initially run the db
  ```
  sudo docker run -d -p 5432:5432 --name dw-stocks --shm-size=1g -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data -v /media/nautilus/fun-times-in-python:/mnt -v /mnt:/media --network bridge postgres
  ```
* Edit configs
  ```
  sudo vim /media/nautilus/docks/postgres/postgresql.conf
  ```
* Create tablespace on SSD
  ```
  create tablespace ssd_tablespace location '/media/dw-stocks-tablespace';
  ```

## Airflow
* Build the airflow image, from `docks/airflow`
  ```
  docker build . --tag airflow
  ```
* Grant sock permissions
  ```
  sudo chmod 777 /var/run/docker.sock
  ```
* Run the airflow container
  ```
  docker run --name airflow-prod -p 8080:8080 -v /media/nautilus/fun-times-in-python/dags:/usr/local/airflow/dags -v /var/run/docker.sock:/var/run/docker.sock:ro -td --network bridge airflow
  ```
* Create airflow user on Postgres
  ```
  create user airflow password 'airflow';
  grant all privileges on all tables in schema public to airflow;
  ```
* Initialize airflow on Postgres
  ```
  airflow initdb
  ```

## Pytorch
* Build docker image
  ```
  sudo docker build . --tag pytorch
  ```
* Run the docker container
  ```
  docker run -it --name pytorch-gpu --runtime=nvidia -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all -v /media/nautilus/fun-times-in-python:/usr/src/app --network bridge pytorch
  ```
* Startup jupyter labs
  ```
  jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/usr/src/app
  ```
* In order to get Dockerized GPUs working in Airflow, update the default docker runtime -- [documentation here.](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html#:~:text=Use%20docker%20run%20with%20nvidia,file%20as%20the%20first%20entry.)

## Jupyter Sans GPU
* Build image
  ```
  sudo docker build . --tag jupyter
  ```
* Run the docker container
  ```
  docker run -it --name jupyter-explorer -p 8888:8888 -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app --network bridge jupyter
  ```
* Startup jupyter labs
  ```
  jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/usr/src/app --allow-root
  ```
