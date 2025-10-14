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
                    let engine = "";

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
                        } else if (resultControllerUpload.name === "engine" && resultControllerUpload.buffer) {
                            engine = resultControllerUpload.buffer.toString().match("^(paddle|tesseract)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        }
                    }

                    const uniqueId = helperSrc.generateUniqueId();

                    const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_INPUT}${fileName}`;
                    const outputCraftResult = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}craft/${uniqueId}/`;
                    const outputEngineResult = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${engine}/${uniqueId}/`;
                    const fileNameParse = Path.parse(fileName).name;

                    const execCommand = `. ${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_SCRIPT}command1.sh`;
                    const execArgumentList = [`"${fileName}"`, `"${language}"`, `"${isCuda}"`, `"${isDebug}"`, `"${engine}"`, `"${uniqueId}"`];

                    execFile(execCommand, execArgumentList, { shell: "/bin/bash", encoding: "utf8" }, (_, stdout) => {
                        helperSrc.fileRemove(input, (resultFileRemove) => {
                            if (typeof resultFileRemove !== "boolean") {
                                helperSrc.writeLog(
                                    "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileRemove(input)",
                                    resultFileRemove.toString()
                                );

                                helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                            }
                        });

                        if (stdout.trim() === "ok") {
                            helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - execFile() - stdout", stdout);

                            helperSrc.fileReadStream(`${outputEngineResult}export/${fileNameParse}.pdf`, async (resultFileReadStream) => {
                                if (Buffer.isBuffer(resultFileReadStream)) {
                                    const dataList = await helperSrc.findFileInDirectoryRecursive(
                                        `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}paddle/${uniqueId}/`,
                                        ".xlsx"
                                    );

                                    const excelList: string[] = [];

                                    for (const data of dataList) {
                                        excelList.push(data.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${engine}/${uniqueId}`, ""));
                                    }

                                    const responseJson = {
                                        uniqueId,
                                        pdf: resultFileReadStream.toString("base64"),
                                        excelList
                                    };

                                    helperSrc.responseBody(JSON.stringify(responseJson), "", response, 200);
                                } else {
                                    helperSrc.writeLog(
                                        "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream()",
                                        resultFileReadStream.toString()
                                    );

                                    helperSrc.responseBody("", resultFileReadStream.toString(), response, 500);
                                }

                                if (engine !== "paddle") {
                                    Fs.readdir(outputCraftResult, (error) => {
                                        if (error) {
                                            helperSrc.writeLog(
                                                "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - Fs.readdir(outputCraftResult)",
                                                error.toString()
                                            );

                                            return;
                                        }

                                        helperSrc.fileRemove(outputCraftResult, (resultFileRemove) => {
                                            if (typeof resultFileRemove !== "boolean") {
                                                helperSrc.writeLog(
                                                    "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileReadStream() - Fs.readdir(outputCraftResult) - fileRemove(resultFileRemove)",
                                                    resultFileRemove.toString()
                                                );

                                                helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                            }
                                        });
                                    });
                                }
                            });
                        } else {
                            helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - execFile() - stdout", stdout);

                            helperSrc.responseBody("", stdout.toString(), response, 500);
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
