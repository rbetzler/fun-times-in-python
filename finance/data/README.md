### Data

Herein are four different frameworks:
* Loader
  * Load data into Postgres
* Reporter
  * Generate reports, which may get sent via email
* Scraper
  * Scrape data from the internet and store it locally
* SQL Runner
  * Run queries on Postgres: DDLs, DMLs, and Administration like VACUUMs


##### Setup & Useful Commands
* Build python docker container
```
sudo docker build . --tag py-dw-stocks
```
* Initially run container
```
sudo docker run -it --name py-temp -v /media/nautilus/fun-times-in-python:/usr/src/app -v /media/nautilus/raw-files:/mnt --network bridge py-dw-stocks
docker run -it --name py-temp -v /Users/rickbetzler/personal/fun-times-in-python:/usr/src/app -v /Users/rickbetzler/personal/raw_files:/mnt --network local-network py-dw-stocks
```
* Run a python script from root
```
docker exec py-temp python /home/utilities/test_dag_script.py
```
* Enter container
```
docker run -it --network bridge -v /media/nautilus/fun-times-in-python:/usr/src/app py-dw-stocks bash
```

##### Postgres
* Pull docker image, since gpg keys are a pain locally
```
sudo docker pull postgres
```
* Initially run the db
```
sudo docker run -d -p 5432:5432 --name dw-stocks -e POSTGRES_PASSWORD=password -v /media/nautilus/docks/postgres:/var/lib/postgresql/data -v /media/nautilus/fun-times-in-python:/mnt -v /mnt:/media --network bridge postgres
```
* Create tablespace on SSD
```
create tablespace ssd_tablespace location '/media/dw-stocks-tablespace';
```
