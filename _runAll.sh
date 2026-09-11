cd /home/app
rm -rf file/output/*/
> log/document_scanner.log
pkill -f "^python3 .*server\.py"
sleep 2
nohup python3 onnx/document_scanner/server.py >> log/document_scanner.log 2>&1 &
for a in $(seq 1 60); do
  if grep -qi "Ready on" log/document_scanner.log 2>/dev/null; then break; fi
  sleep 2
done
sleep 3
IFS=$'\n'
for pathFile in $(ls file/test/); do
  nameFile="${pathFile%.*}"
  curl -s -o /dev/null -X POST -H "Content-Type: application/json" \
    -d "{\"pathInput\":\"file/test/$pathFile\",\"pathOutput\":\"file/output/$nameFile/\",\"searchText\":\"\"}" \
    http://127.0.0.1:1114/engine
done
