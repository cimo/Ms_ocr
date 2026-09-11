cd /home/app
rm -rf file/output/*/
nohup python3 onnx/document_scanner/test_server.py > log/document_scanner.log 2>&1 &
for a in $(seq 1 60); do
  if grep -qi "Ready on" log/document_scanner.log 2>/dev/null; then break; fi
  sleep 2
done
sleep 3
IFS=$'\n'
for pathFile in $(ls file/test/); do
  nameFile="${pathFile%.*}"
  echo "=== $pathFile"
  curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"pathInput\":\"file/test/$pathFile\",\"pathOutput\":\"file/output/$nameFile/\",\"searchText\":\"\"}" \
    http://127.0.0.1:1115/engine
  echo
done
pkill -f test_server.py
grep -iv Provider log/document_scanner.log | grep -iE "error|traceback|exception" | head -20
echo "--- fine"
