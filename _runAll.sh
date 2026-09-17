cd /home/app
rm -rf file/output/*/
> log/onnx.log
pkill -f "onnx/server\.py"
sleep 2
nohup python3 onnx/server.py >> log/onnx.log 2>&1 &
for a in $(seq 1 60); do
  if grep -qi "Ready on" log/onnx.log 2>/dev/null; then break; fi
  sleep 2
done
sleep 3
IFS=$'\n'
for pathFile in $(ls file/test/); do
  nameFile="${pathFile%.*}"
  curl -s -o /dev/null -X POST -H "Content-Type: application/json" \
    -d "{\"pathInput\":\"file/test/$pathFile\",\"pathOutput\":\"file/output/$nameFile/\",\"password\":\"\"}" \
    http://127.0.0.1:1114/engine
done
