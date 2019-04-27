# Commands

### Docker
```
sudo docker images
sudo docker ps -a
sudo docker network create local-network
```


#### Python
##### From dockerfile
```
sudo docker build . --tag python3`
sudo docker run -it --name py-temp --network local-network python3
```
##### Start container, access terminal
`sudo docker start -i py-temp`


#### Postgres
##### Docker build is too painful
```
sudo docker pull postgres
sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data --network local-network postgres
```
##### Start postgres, access terminal and db
`sudo docker start dw-stocks`
`sudo docker exec -it dw-stocks bash`
`psql -U postgres`


#### Airflow
##### From dockerfile
```
sudo docker build . --tag airflow
sudo docker run --name airflow-prod -p 8080:8080 -v /home/nautilus/development/fun-times-in-python/dags:/usr/local/airflow/dags -v /home/nautilus/development/fun-times-in-python/py-scripts:/usr/local/airflow_home -td --network local-network airflow
```
##### Start container, access terminal
`sudo docker start airflow-prod`
`sudo docker exec -it airflow-prod bash`
##### Test and trigger dag
`airflow run test_dag print_test 2019-01-01`
`airflow trigger_dag test_dag`
##### Start scheduler
 `airflow scheduler`
