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
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres-config.yaml
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres-storage.yaml
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres-deployment.yaml
microk8s.kubectl create -f /home/nautilus/development/fun-times-in-python/kubes/postgres-service.yaml
```

### Get service info
`microk8s.kubectl get svc postgres`

### Access kube postgres
`psql -h localhost -U postgresadmin --password -p 31801 postgresdb`
