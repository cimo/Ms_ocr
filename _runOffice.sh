cd /home/app
rm -rf file/output/*/
IFS=$'\n'
for pathFile in $(ls file/test/ | grep -iE "\.(docx|xlsx|pptx)$"); do
  nameFile="${pathFile%.*}"
  echo "=== $pathFile"
  curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"pathInput\":\"file/test/$pathFile\",\"pathOutput\":\"file/output/$nameFile/\",\"searchText\":\"\"}" \
    http://127.0.0.1:1114/engine
  echo
done
echo "--- fine"
