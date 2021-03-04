
if [[ $(uname -s) = "Darwin" ]]; then
  cd /Users/rickbetzler/Downloads
else
  cd /home/nautilus/Downloads
fi
echo "In dir: $(pwd)"

url="https://www.nasdaq.com/market-activity/stocks/screener"
if [[ $(uname -s) = "Darwin" ]]; then
  open -a "Google Chrome" $url
else
  chromium-browser $url
fi

echo "Sleeping while the file downloads"
sleep 20s

file=$(ls -lt | grep "nasdaq" | awk '{print $9}' | head -1)
if [[ $(uname -s) = "Darwin" ]]; then
  prefix="/Users/rickbetzler/personal"
else
  prefix="/media/nautilus"
fi
target="$prefix/fun-times-in-python/audit/nasdaq/listed_stocks/nasdaq_screener_$(date +"%Y%m%d%H%M%S").csv"
echo "Copying $file to $target"
cp $file $target

echo "Running nasdaq loading, updating dbt tables, writing report"
docker exec py-temp bash -c "python data/nasdaq/load.py; cd dbt; dbt run -m tickers; cd ..; python data/nasdaq/report.py"
echo "Docker jobs complete"

if [ $# -eq 0 ]; then
  echo "Writing report to pi"
  PiIp="$(nmap -sn 192.168.1.0/24 | grep "pi" | awk -F"[()]" '{print $2}')"
  echo "Pi Ip identified as $PiIp"

  echo "Running pi startup script"
  ssh pi@$PiIp 'sh /home/pi/fun-times-in-python/scripts/pi.sh 0'

  report="fun-times-in-python/audit/reports/listed_stocks/listed_stocks.csv"
  echo "Copying report to pi: $report"
  scp $prefix/$report pi@$PiIp:/home/pi/$report

else
  echo "Skipping pi sync"

fi

echo "Script finished"
