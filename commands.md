# Commands

## Kubernetes

### Microk8s
```
microk8s.start &&
microk8s.status &&
microk8s.kubectl get nodes &&
microk8s.kubectl get deployment
```

### Create kube services
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
sudo docker run --rm -d -p 5432:5432 --name temp-postgres -e POSTGRES_PASSWORD=password postgres
sudo docker exec -it temp-postgres bash
psql -U postgres
```

##### Airflow
```
sudo docker pull puckel/docker-airflow
sudo docker run --name airflow-prod -p 8080:8080 -td airflow
```
