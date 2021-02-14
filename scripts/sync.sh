#!/bin/sh

cd /media/nautilus/fun-times-in-python
echo "In dir: $(pwd)"

PiIp=$(nmap -sn 192.168.1.0/24 | grep "pi" | awk -F"[()]" '{print $2}')
echo "Pi Ip identified as $PiIp"

echo "Running pi startup script"
ssh pi@$PiIp 'sh /home/pi/fun-times-in-python/scripts/pi.sh 0'

echo "Running td data syncs to pi"
Jobs="options fundamentals quotes equities"
for j in $Jobs; do
  echo "Syncing $j"
  rsync -rv --ignore-existing audit/td_ameritrade/$j/* pi@$PiIp:/home/pi/fun-times-in-python/audit/$j/
done
echo "Jobs completed."

echo "Running yahoo data syncs to pi"
Jobs="balance_sheet_annual balance_sheet_quarterly cash_flow_annual cash_flow_quarterly income_statement_quarterly income_statements_annual income_statements_quarterly options"
for j in $Jobs; do
  echo "Syncing $j"
  rsync -rv --ignore-existing audit/yahoo/$j/* pi@$PiIp:/home/pi/fun-times-in-python/audit/yahoo/$j/
done
echo "Jobs completed."
