# Commands

## Kubernetes

### Microk8s
```
microk8s.start &&
microk8s.status &&
microk8s.kubectl get nodes &&
microk8s.kubectl get deployment
```

### Create kube services - postgres
```
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres/postgres-config.yaml &&
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres/postgres-storage.yaml &&
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres/postgres-deployment.yaml &&
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres/postgres-service.yaml
```

### Get service info
`microk8s.kubectl get svc postgres`

### Access kube postgres
`psql -h localhost -U dbadmin --password -p 31801 dw_stocks`

### Docker
```
sudo docker images
sudo docker ps -a
sudo docker tag 3be5dc25d0fa python2
sudo docker run -it --rm python2
```

##### Postgres
```
sudo docker pull postgres
sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data postgres
sudo docker exec -it dw-stocks bash
psql -U postgres
```

##### Airflow
```
sudo docker pull puckel/docker-airflow
sudo docker tag 6e247b3efc89 airflow
sudo docker run --name airflow-prod -p 8080:8080 -v /home/nautilus/development/fun-times-in-python/dags:/usr/local/airflow/dags -td airflow
sudo docker start airflow-prod
sudo docker exec -ti airflow-prod bash
```
