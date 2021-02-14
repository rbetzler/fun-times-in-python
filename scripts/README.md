


## Pi
* Update pi
  ```
  sudo apt-get update && sudo-apt-get upgrade
  ```
* Install python packages (no venv)
  ```
  sudo apt-get install python3-numpy python3-pandas python3-requests python3-psycopg2 -y
  ```

## Sync
* Map network to find pi
  ```
  nmap -sn 192.168.1.0/24
  ```
* Ssh in
  ```
  ssh pi@000.0.0.0
  ```
* Run startup to mount hdd
  ```
  sh pi.sh random_arg
  ```
