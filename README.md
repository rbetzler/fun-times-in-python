# fun-times-in-python

1. Scrape financial data
2. Process and store in a postgres instance
3. Run data science models

* Maintain everything in docker containers
* Execute tasks in airflow

### Misc Commands
#### Python
##### From dockerfile
```
sudo docker build . --tag py-dw-stocks
sudo docker run -it --name py-temp -v /media/nautilus/fun-times-in-python:/usr/src/app -v /media/nautilus/raw-files:/mnt --network bridge py-dw-stocks
docker run -it --name py-temp -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app -v /Users/rickbetzler/personal/raw_files:/mnt --network local-network py-dw-stocks
```
##### Run a python script
```
docker exec py-temp python /home/utilities/test_dag_script.py
docker run -it --network bridge -v /media/nautilus/fun-times-in-python:/usr/src/app py-dw-stocks bash
```

#### Postgres
##### Docker build is too painful
```
sudo docker pull postgres
sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data -v /media/nautilus/fun-times-in-python:/mnt -v /mnt:/media --network bridge postgres
docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /Users/rickbetzler/personal/docks/postgres:/var/lib/postgresql/data --network local-network postgres

also run: create tablespace ssd_tablespace location '/media/dw-stocks-tablespace';
```


#### Airflow
##### From dockerfile
```
docker build . --tag airflow
docker run --name airflow-prod -p 8080:8080 -v /media/nautilus/fun-times-in-python/dags:/usr/local/airflow/dags -v /var/run/docker.sock:/var/run/docker.sock:ro -td --network bridge airflow
```
Manual airflow setup:
```
# Airflow host permissions
sudo chmod 777 /var/run/docker.sock

# Create airflow user in postgres
create user airflow password 'airflow';
grant all privileges on all tables in schema public to airflow;

# Run airflow init
airflow initdb
```

##### Start container, access terminal
```
sudo docker start airflow-prod
sudo docker exec -it airflow-prod bash
```

#### Pytorch
```
sudo docker build . --tag pytorch
docker run -it --name pytorch-gpu --runtime=nvidia -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all -v /media/nautilus/fun-times-in-python:/usr/src/app --network bridge pytorch
jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/usr/src/app
```

#### Jupyter Sans GPU
```
sudo docker build . --tag jupyter
docker run -it --name jupyter-explorer -p 8888:8888 -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app --network bridge jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/usr/src/app --allow-root
```
