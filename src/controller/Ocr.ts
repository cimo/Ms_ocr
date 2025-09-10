/* eslint-disable @typescript-eslint/no-unused-vars */
import Express, { Request, Response } from "express";
import { RateLimitRequestHandler } from "express-rate-limit";
import { execFile } from "child_process";
import Path from "path";
import Fs from "fs";
import { Ca } from "@cimo/authentication/dist/src/Main";

// Source
import * as helperSrc from "../HelperSrc";
import ControllerUpload from "./Upload";

export default class ControllerOcr {
    // Variable
    private app: Express.Express;
    private limiter: RateLimitRequestHandler;
    private controllerUpload: ControllerUpload;

    // Method
    constructor(app: Express.Express, limiter: RateLimitRequestHandler) {
        this.app = app;
        this.limiter = limiter;
        this.controllerUpload = new ControllerUpload();
    }

    api = (): void => {
        this.app.post("/api/extract", this.limiter, Ca.authenticationMiddleware, (request: Request, response: Response) => {
            this.controllerUpload
                .execute(request, true)
                .then((resultControllerUploadList) => {
                    let fileName = "";
                    let language = "";
                    let isCuda = "";
                    let isDebug = "";

                    for (const resultControllerUpload of resultControllerUploadList) {
                        if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                            fileName = resultControllerUpload.fileName;
                        } else if (resultControllerUpload.name === "language" && resultControllerUpload.buffer) {
                            language = resultControllerUpload.buffer.toString().match("^(jp|jp_vert|en)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        } else if (resultControllerUpload.name === "isCuda" && resultControllerUpload.buffer) {
                            isCuda = resultControllerUpload.buffer.toString().match("^(true|false)$") ? resultControllerUpload.buffer.toString() : "";
                        } else if (resultControllerUpload.name === "isDebug" && resultControllerUpload.buffer) {
                            isDebug = resultControllerUpload.buffer.toString().match("^(true|false)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        }
                    }

                    const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_INPUT}${fileName}`;
                    const outputCraftResult = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}craft/`;
                    const outputTesseractResult = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}tesseract/`;
                    const fileNameParse = Path.parse(fileName).name;

                    const execCommand = `. ${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_SCRIPT}command1.sh`;
                    const execArgumentList = [`"${fileName}"`, `"${language}"`, `"${isCuda}"`, `"${isDebug}"`];

                    execFile(execCommand, execArgumentList, { shell: "/bin/bash", encoding: "utf8" }, (_, stdout, stderr) => {
                        if ((stdout !== "" && stderr === "") || (stdout !== "" && stderr !== "")) {
                            helperSrc.fileReadStream(`${outputTesseractResult}${fileNameParse}.pdf`, (resultFileReadStream) => {
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

                                /*Fs.readdir(outputCraftResult, (error, fileNameList) => {
                                    if (error) {
                                        helperSrc.writeLog(
                                            "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - Fs.readdir(outputCraftResult)",
                                            resultFileReadStream.toString()
                                        );

                                        return;
                                    }

                                    for (const fileName of fileNameList) {
                                        if (fileName.startsWith(fileNameParse)) {
                                            helperSrc.fileRemove(`${outputCraftResult}${fileName}`, (resultFileRemove) => {
                                                if (typeof resultFileRemove !== "boolean") {
                                                    helperSrc.writeLog(
                                                        "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - Fs.readdir(outputCraftResult) - fileRemove(outputCraftResult)",
                                                        resultFileRemove.toString()
                                                    );

                                                    helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                                }
                                            });
                                        }
                                    }
                                });

                                Fs.readdir(outputTesseractResult, (error, fileNameList) => {
                                    if (error) {
                                        helperSrc.writeLog(
                                            "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - Fs.readdir(outputTesseractResult)",
                                            resultFileReadStream.toString()
                                        );

                                        return;
                                    }

                                    for (const fileName of fileNameList) {
                                        if (fileName.startsWith(fileNameParse)) {
                                            helperSrc.fileRemove(`${outputTesseractResult}${fileName}`, (resultFileRemove) => {
                                                if (typeof resultFileRemove !== "boolean") {
                                                    helperSrc.writeLog(
                                                        "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - Fs.readdir(outputTesseractResult) - fileRemove(outputTesseractResult)",
                                                        resultFileRemove.toString()
                                                    );

                                                    helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                                }
                                            });
                                        }
                                    }
                                });*/
                            });
                        } else if (stdout === "" && stderr !== "") {
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
        });
    };
}
