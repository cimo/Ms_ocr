import sys
sys.dont_write_bytecode = True

import os
import json
import time
import signal
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Source
import layout
import engine

class HandlerHttpRequest(BaseHTTPRequestHandler):
    layoutImage = layout.Image()
    layoutOfficeDocx = layout.Office.Docx()
    layoutOfficeXlsx = layout.Office.Xlsx()
    layoutOfficePptx = layout.Office.Pptx()

    engineProcessor = engine.Processor()

    def _routeLayout(self, text):
        payload = json.loads(text)

        pathInput = payload.get("pathInput")
        pathOutput = payload.get("pathOutput")

        extension = os.path.splitext(pathInput)[1].lower()
        fileName = os.path.basename(pathInput)

        result = {}

        if extension in self.engineProcessor.extensionImageList:
            self.engineProcessor.pageImageGenerate("single", pathInput, pathOutput)

            result = self.layoutImage.execute(f"{pathOutput}page/", pathOutput, fileName)
        elif extension == ".pdf":
            self.engineProcessor.pageImageGenerate("multiple", pathInput, pathOutput)

            result = self.layoutImage.execute(f"{pathOutput}page/", pathOutput, fileName)
        elif extension == ".docx":
            result = self.layoutOfficeDocx.execute(pathInput, pathOutput, fileName)
        elif extension == ".xlsx":
            result = self.layoutOfficeXlsx.execute(pathInput, pathOutput, fileName)
        elif extension == ".pptx":
            result = self.layoutOfficePptx.execute(pathInput, pathOutput, fileName)

        return result

    def _routeEngine(self, text):
        payload = json.loads(text)

        pathInput = payload.get("pathInput")
        pathOutput = payload.get("pathOutput")
        searchText = payload.get("searchText")

        fileName = os.path.basename(pathInput)

        return self.engineProcessor.execute(pathInput, pathOutput, fileName, searchText)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))

        text = self.rfile.read(length).decode("utf-8")

        result = {}

        if self.path == "/layout":
            result = self._routeLayout(text)
        elif self.path == "/engine":
            result = self._routeEngine(text)

        body = json.dumps(result, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        self.wfile.write(body)

    def log_message(self, format, *argumentList):
        return

class ServerHttp(ThreadingHTTPServer):
    def handle_error(self, request, clientAddress):
        errorText = str(sys.exc_info()[1])

        bodyByte = json.dumps({"error": errorText}, ensure_ascii=False).encode("utf-8")

        headerText = f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {len(bodyByte)}\r\nConnection: close\r\n\r\n"

        request.sendall(headerText.encode("utf-8") + bodyByte)

        print(f"Error: {errorText}")

urlSplit = os.environ["MS_O_URL_API_ONNX_DS"].replace("http://", "").split(":")
host = urlSplit[0]
port = int(urlSplit[1])

checkSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
isRunning = checkSocket.connect_ex((host, port)) == 0
checkSocket.close()

if isRunning:
    pathScript = os.path.dirname(os.path.abspath(__file__))
    pgrepRun = subprocess.run(["pgrep", "-f", f"{pathScript}/server.py"], capture_output=True, text=True)
    pidSplit = pgrepRun.stdout.split()

    for a in range(len(pidSplit)):
        if int(pidSplit[a]) != os.getpid():
            os.kill(int(pidSplit[a]), signal.SIGTERM)

    while isRunning:
        time.sleep(0.1)

        checkSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        isRunning = checkSocket.connect_ex((host, port)) == 0
        checkSocket.close()

serverHttp = ServerHttp((host, port), HandlerHttpRequest)

print(f"Onnx - document_scanner - Ready on => {host}:{port}")

serverHttp.serve_forever()
