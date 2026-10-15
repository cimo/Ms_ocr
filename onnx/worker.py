import sys
import os
import json
import concurrent.futures

sys.dont_write_bytecode = True

os.sched_setaffinity(0, range(max(1, os.cpu_count() // 2)))

# Source
import engine

fileOutput = os.fdopen(os.dup(1), "w", encoding="utf-8")

os.dup2(2, 1)
sys.stdout = sys.stderr

def lineWrite(dataObject):
    fileOutput.write(f"{json.dumps(dataObject, ensure_ascii=False)}\n")
    fileOutput.flush()

engineObject = engine.Engine()

executorThread = concurrent.futures.ThreadPoolExecutor(1)

print("Onnx - Ready")

for line in sys.stdin:
    payload = json.loads(line)

    future = executorThread.submit(engineObject.execute, payload["pathInput"], payload["pathOutput"], payload["password"])

    if future.exception() is not None:
        print(f"Error: {future.exception()}")

        lineWrite({"uniqueId": payload["uniqueId"], "error": str(future.exception())})

        continue

    resultObject = future.result()
    resultObject["uniqueId"] = payload["uniqueId"]

    lineWrite(resultObject)
