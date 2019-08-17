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
sudo docker build . --tag py-dw-stocks
sudo docker run -it --name py-temp -v /home/nautilus/development/fun-times-in-python:/home -v /media/nautilus/raw-files:/mnt --network local-network py-dw-stocks
docker run -it --name py-temp -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app -v /Users/rickbetzler/personal/raw_files:/mnt --network local-network py-dw-stocks
```
##### Start container, access terminal
```
sudo docker start py-temp
sudo docker exec -it py-temp bash
```
##### Run a python script
`sudo docker exec py-temp python /home/utilities/test_dag_script.py`


#### Postgres
##### Docker build is too painful
```
sudo docker pull postgres
sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data --network local-network postgres
docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /Users/rickbetzler/personal/docks/postgres:/var/lib/postgresql/data --network local-network postgres
```
##### Start postgres, access terminal and db
```
sudo docker start dw-stocks
sudo docker exec -it dw-stocks bash
psql -U postgres
```
connect from host: `psql -h 172.17.0.1 -U postgres`


#### Airflow
##### From dockerfile
```
sudo docker build . --tag airflow
sudo docker run --name airflow-prod -p 8080:8080 -v /home/nautilus/development/fun-times-in-python/dags:/usr/local/airflow/dags -v /home/nautilus/development/fun-times-in-python/py-scripts:/usr/local/airflow_home -v /var/run/docker.sock:/var/run/docker.sock:ro -v /requirements.txt:/requirements.txt -td --network local-network airflow
```
##### Start container, access terminal
```
sudo docker start airflow-prod
sudo docker exec -it airflow-prod bash
```
##### Test and trigger dag
```
airflow run test_dag print_test 2019-01-01
airflow trigger_dag test_dag
```
##### Start scheduler
 `airflow scheduler`
