# Commands

### Docker
```
sudo docker images
sudo docker ps -a
sudo docker tag 3be5dc25d0fa python2
sudo docker run -it --rm python2
sudo docker network create local-network
```


#### Python
`sudo docker build . --tag python3`
`sudo docker run -it --name py-temp --network local-network python3`


#### Postgres
##### Docker build is too painful
`sudo docker pull postgres --tag postgres`
##### Setup container
`sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data postgres`
##### Access terminal and db
```
sudo docker exec -it dw-stocks bash
psql -U postgres
```

#### Airflow
##### From dockerfile
`sudo docker build . --tag airflow`
##### Setup container
`sudo docker run --name airflow -p 8080:8080 -v /home/nautilus/development/fun-times-in-python/dags:/usr/local/airflow/dags &&
-v /home/nautilus/development/fun-times-in-python/py-scripts:/usr/local/airflow_home -td &&
--network local-network airflow`
##### Startup container
`sudo docker start airflow-prod`
##### Access terminal
`sudo docker exec -it airflow-prod bash`
