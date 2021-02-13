#!/bin/sh

if grep -qs '/dev/sda1 ' /proc/mounts; then
    echo "HDD already mounted."
else
    sudo mount /dev/sda1 /home/pi/fun-times-in-python/audit/
    echo "Mounting HDD."
fi

export PYTHONPATH=/home/pi/fun-times-in-python
cd $PYTHONPATH

echo "Moved to dir: $(pwd)"

if [ $# -eq 0 ]; then
	echo "Running python scripts"
	Jobs="options income_statements balance_sheet cash_flow"
	for j in $Jobs; do
		echo "Running $j"
		/usr/bin/python3 $PYTHONPATH/data/yahoo/$j/scrape.py
	done
	echo "Jobs completed."
else
	echo "Argument passed. Not running python scripts."
fi
