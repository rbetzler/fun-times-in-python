# Commands

## Kubernetes

### Microk8s
```
microk8s.start
microk8s.status
microk8s.kubectl get nodes
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
`sudo docker ps -a`
`sudo docker run -it --rm 105fbac90ce5`
