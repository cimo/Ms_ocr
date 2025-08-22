import Express, { Request, Response } from "express";
import { execFile } from "child_process";
import { Ca } from "@cimo/authentication/dist/src/Main";

// Source
import * as helperSrc from "../HelperSrc";
import ControllerUpload from "./Upload";

export default class ControllerOcr {
    // Variable
    private app: Express.Express;
    private controllerUpload: ControllerUpload;

    // Method
    constructor(app: Express.Express) {
        this.app = app;
        this.controllerUpload = new ControllerUpload();
    }

    api = (): void => {
        this.app.post("/api/extract", Ca.authenticationMiddleware, (request: Request, response: Response) => {
            void (async () => {
                await this.controllerUpload
                    .execute(request, true)
                    .then((resultControllerUploadList) => {
                        let fileName = "";
                        let language = "";
                        let result = "";
                        let debug = "";

                        for (const resultControllerUpload of resultControllerUploadList) {
                            if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                                fileName = resultControllerUpload.fileName;
                            } else if (resultControllerUpload.name === "language" && resultControllerUpload.buffer) {
                                language = resultControllerUpload.buffer.toString().match("^(jp|jp_vert|en)$")
                                    ? resultControllerUpload.buffer.toString()
                                    : "";
                            } else if (resultControllerUpload.name === "result" && resultControllerUpload.buffer) {
                                result = resultControllerUpload.buffer.toString().match("^(txt|hocr|tsv|pdf)$")
                                    ? resultControllerUpload.buffer.toString()
                                    : "";
                            } else if (resultControllerUpload.name === "debug" && resultControllerUpload.buffer) {
                                debug = resultControllerUpload.buffer.toString().match("^(true|false)$")
                                    ? resultControllerUpload.buffer.toString()
                                    : "";
                            }
                        }

                        const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_INPUT}${fileName}`;
                        const outputResult = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${fileName}_result.png`;
                        const output = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${fileName}.${result}`;

                        const execCommand = `. ${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_SCRIPT}command1.sh`;
                        const execArgumentList = [`"${fileName}"`, `"${language}"`, `"${result}"`, `"${debug}"`];

                        execFile(execCommand, execArgumentList, { shell: "/bin/bash", encoding: "utf8" }, (_, stdout, stderr) => {
                            // Tesseract use always stderr for the output
                            if (stdout === "" || stderr !== "" || (stdout !== "" && stderr !== "")) {
                                helperSrc.fileReadStream(output, (resultFileReadStream) => {
                                    if (Buffer.isBuffer(resultFileReadStream)) {
                                        helperSrc.responseBody(resultFileReadStream.toString("base64"), "", response, 200);
                                    } else {
                                        helperSrc.writeLog(
                                            "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream()",
                                            resultFileReadStream.toString()
                                        );

                                        helperSrc.responseBody("", resultFileReadStream.toString(), response, 500);
                                    }

                                    helperSrc.fileRemove(input, (resultFileRemove) => {
                                        if (typeof resultFileRemove !== "boolean") {
                                            helperSrc.writeLog(
                                                "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - fileRemove(input)",
                                                resultFileRemove.toString()
                                            );

                                            helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                        }
                                    });

                                    helperSrc.fileRemove(outputResult, (resultFileRemove) => {
                                        if (typeof resultFileRemove !== "boolean") {
                                            helperSrc.writeLog(
                                                "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - fileRemove(outputResult)",
                                                resultFileRemove.toString()
                                            );

                                            helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                        }
                                    });

                                    helperSrc.fileRemove(output, (resultFileRemove) => {
                                        if (typeof resultFileRemove !== "boolean") {
                                            helperSrc.writeLog(
                                                "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - fileRemove(output)",
                                                resultFileRemove.toString()
                                            );

                                            helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                        }
                                    });
                                });
                            } else if (stdout !== "" || stderr === "") {
                                helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - execFile() - stdout", stdout);

                                helperSrc.fileRemove(input, (resultFileRemove) => {
                                    if (typeof resultFileRemove !== "boolean") {
                                        stderr += resultFileRemove;

                                        helperSrc.writeLog(
                                            "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileRemove(input)",
                                            resultFileRemove.toString()
                                        );
                                    }
                                });

                                helperSrc.responseBody(stdout, "", response, 500);
                            }
                        });
                    })
                    .catch((error: Error) => {
                        helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - catch()", error);

                        helperSrc.responseBody("", error, response, 500);
                    });
            })();
        });
    };
}
