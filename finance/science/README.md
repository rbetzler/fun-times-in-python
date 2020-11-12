### Data

Herein are two frameworks:
* Predictor, which runs an LSTM to generate predictions
* Decisioner, which selects trades to make

Both frameworks write files for ingestion into Postgres.

##### Pytorch
* Build docker image
```
sudo docker build . --tag pytorch
```
* Run the docker container
```
docker run -it --name pytorch-gpu --runtime=nvidia -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all -v /media/nautilus/fun-times-in-python:/usr/src/app --network bridge pytorch
```
* Startup jupyter labs
```
jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/usr/src/app
```
* In order to get Dockerized GPUs working in Airflow, update the default docker runtime -- [documentation here.](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html#:~:text=Use%20docker%20run%20with%20nvidia,file%20as%20the%20first%20entry.)

##### Jupyter Sans GPU
* Build image
```
sudo docker build . --tag jupyter
```
* Run the docker container
```
docker run -it --name jupyter-explorer -p 8888:8888 -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app --network bridge jupyter
```
* Startup jupyter labs
```
jupyter lab --ip=0.0.0.0 --port=8888 --notebook-dir=/usr/src/app --allow-root
```
