### Airflow

##### Setup

* Build the airflow image
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

##### Useful commands
* Startup airflow container
```
sudo docker start airflow-prod
```
* Enter container
```
sudo docker exec -it airflow-prod bash
```
