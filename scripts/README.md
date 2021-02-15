## Miscellaneous bash scripts

## Startup nautilus
* `nautilus.sh` is a symlinked file to mount hdd and startup docker containers

## Pi setup
* Update pi
  ```
  sudo apt-get update && sudo-apt-get upgrade
  ```
* Install python packages (no venv)
  ```
  sudo apt-get install python3-numpy python3-pandas python3-requests python3-psycopg2 -y
  ```
* Enable ssh via pi: [instructions here](https://www.raspberrypi.org/documentation/remote-access/ssh/)
* Setup ssh keys
  ```
  cat ~/.ssh/id_rsa.pub | ssh pi@IP 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'
  ```

## Sync
* Run `sh scripts/sync.sh`
* To sync data from nautilus to pi

## Nasdaq
* Run `sh nasdaq.sh` and click `Download CSV` on the webpage that opens
* Semi-manual process to download a csv from the nasdaq website and run some jobs
* TODO: Replace bash with selenium
