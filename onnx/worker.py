import sys
import os
import json
import concurrent.futures
import functools

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

executorThread = concurrent.futures.ThreadPoolExecutor(int(os.environ["MS_FDE_QUEUE"]))

jobObject = {}

def jobDone(uniqueId, future):
    del jobObject[uniqueId]

    if future.exception() is not None:
        print(f"Error: {future.exception()}")

        lineWrite({"uniqueId": uniqueId, "error": str(future.exception())})

        return

    resultObject = future.result()
    resultObject["uniqueId"] = uniqueId

    lineWrite(resultObject)

print("Onnx - Ready")

for line in sys.stdin:
    payload = json.loads(line)

    if payload["action"] == "cancel":
        if payload["uniqueId"] in jobObject:
            jobObject[payload["uniqueId"]]["isCancel"] = True

        continue

    jobObject[payload["uniqueId"]] = {"isCancel": False}

    future = executorThread.submit(engineObject.execute, payload["pathInput"], payload["pathOutput"], jobObject[payload["uniqueId"]], payload["password"])
    future.add_done_callback(functools.partial(jobDone, payload["uniqueId"]))
