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
sudo docker run -it --name py-temp -v /media/nautilus/fun-times-in-python:/usr/src/app -v /media/nautilus/raw-files:/mnt --network bridge py-dw-stocks
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
sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data -v /media/nautilus/fun-times-in-python:/mnt --network bridge postgres
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
sudo docker run --name airflow-prod -p 8080:8080 -v /media/nautilus/fun-times-in-python/dags:/usr/local/airflow/dags -v /media/nautilus/development/fun-times-in-python/py-scripts:/usr/local/airflow_home -v /var/run/docker.sock:/var/run/docker.sock:ro -v /requirements.txt:/requirements.txt -td --network bridge airflow

sudo chmod 777 /var/run/docker.sock
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

#### Pytorch
`sudo docker build . --tag pytorch` or `docker pull anibali/pytorch:cuda-8.0`
`docker run -it --name pytorch-gpu --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -v /media/nautilus/fun-times-in-python:/app --network bridge pytorch`

##### Start local, undockerized jupyter
`jupyter notebook --notebook-dir=/Users/rickbetzler/personal/fun-times-in-python/`
`jupyter notebook --notebook-dir=/media/nautilus/fun-times-in-python/`

#### Tensorflow + Jupyter
`sudo docker pull tensorflow/tensorflow:latest-gpu-jupyter --tag tensorflow-stocks`
`sudo docker run -d --name tensor --gpus all -v /media/nautilus/fun-times-in-python:/tf --network bridge tensorflow-stocks`
`jupyter notebook list`