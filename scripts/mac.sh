
Jobs="options fundamentals quotes"

for j in $Jobs; do
  echo "Scraping $j"
  docker exec py-temp bash -c "python data/td_ameritrade/$j/scrape.py"
done
echo "Scraping jobs complete"
