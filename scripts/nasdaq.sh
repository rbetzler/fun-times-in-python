
cd /home/nautilus/Downloads
echo "In dir: $(pwd)"

chromium-browser "https://www.nasdaq.com/market-activity/stocks/screener"

echo "Sleeping while the file downloads"
sleep 20s

file=$(ls -lt | grep "nasdaq" | awk '{print $9}' | head -1)
target="/media/nautilus/fun-times-in-python/audit/nasdaq/listed_stocks/nasdaq_screener_$(date +"%Y%m%d%H%M%S").csv"
echo "Copying $file to $target"
cp $file $target

echo "Running nasdaq loading, updating dbt tables, writing report"
docker exec py-temp bash -c "python data/nasdaq/load.py; cd dbt; dbt run -m tickers; cd ..; python data/nasdaq/report.py"
echo "Docker jobs complete"

echo "Writing report to pi"
PiIp="$(nmap -sn 192.168.1.0/24 | grep "pi" | awk -F"[()]" '{print $2}')"
echo "Pi Ip identified as $PiIp"

echo "Running pi startup script"
ssh pi@$PiIp 'sh /home/pi/fun-times-in-python/scripts/pi.sh 0'

report="fun-times-in-python/audit/reports/listed_stocks/listed_stocks.csv"
echo "Copying report to pi: $report"
scp /media/nautilus/$report pi@$PiIp:/home/pi/$report

echo "Script finished"
