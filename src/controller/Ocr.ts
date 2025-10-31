import Express, { Request, Response } from "express";
import { RateLimitRequestHandler } from "express-rate-limit";
import { execFile } from "child_process";
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
                        if (resultControllerUpload.name === "language" && resultControllerUpload.buffer) {
                            language = resultControllerUpload.buffer.toString().match("^(jp|jp_vert|en)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        } else if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                            fileName = resultControllerUpload.fileName;
                        } else if (resultControllerUpload.name === "isCuda" && resultControllerUpload.buffer) {
                            isCuda = resultControllerUpload.buffer.toString().match("^(true|false)$") ? resultControllerUpload.buffer.toString() : "";
                        } else if (resultControllerUpload.name === "isDebug" && resultControllerUpload.buffer) {
                            isDebug = resultControllerUpload.buffer.toString().match("^(true|false)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        }
                    }

                    const uniqueId = helperSrc.generateUniqueId();

                    const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_INPUT}${fileName}`;

                    const execCommand = `. ${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_SCRIPT}command1.sh`;
                    const execArgumentList = [`"${language}"`, `"${fileName}"`, `"${isCuda}"`, `"${isDebug}"`, `"${uniqueId}"`];

                    execFile(execCommand, execArgumentList, { shell: "/bin/bash", encoding: "utf8" }, async (_, stdout) => {
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

                            const dataJsonList = await helperSrc.findFileInDirectoryRecursive(
                                `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/export/`,
                                ".json"
                            );

                            const dataPdfList = await helperSrc.findFileInDirectoryRecursive(
                                `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/export/`,
                                ".pdf"
                            );

                            const dataXlsxList = await helperSrc.findFileInDirectoryRecursive(
                                `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/table/`,
                                ".xlsx"
                            );

                            const dataHtmlList = await helperSrc.findFileInDirectoryRecursive(
                                `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/table/`,
                                ".html"
                            );

                            const jsonList: string[] = [];

                            for (const dataJson of dataJsonList) {
                                jsonList.push(
                                    dataJson.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/`, "")
                                );
                            }

                            const pdfList: string[] = [];

                            for (const dataPdf of dataPdfList) {
                                pdfList.push(
                                    dataPdf.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/`, "")
                                );
                            }

                            const excelList: string[] = [];

                            for (const dataXlsx of dataXlsxList) {
                                excelList.push(
                                    dataXlsx.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/`, "")
                                );
                            }

                            const htmlList: string[] = [];

                            for (const dataHtml of dataHtmlList) {
                                htmlList.push(
                                    dataHtml.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/`, "")
                                );
                            }

                            const responseJson = {
                                uniqueId,
                                jsonList,
                                pdfList,
                                excelList,
                                htmlList
                            };

                            helperSrc.responseBody(JSON.stringify(responseJson), "", response, 200);
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

        this.app.post("/api/download", this.limiter, Ca.authenticationMiddleware, (request: Request, response: Response) => {
            const requestBody = request.body;

            const uniqueId = requestBody.uniqueId;
            const pathFile = requestBody.pathFile;

            const path = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE_OUTPUT}${helperSrc.ENGINE}/${uniqueId}/${pathFile}`;

            helperSrc.fileReadStream(path, async (resultFileReadStream) => {
                if (Buffer.isBuffer(resultFileReadStream)) {
                    helperSrc.responseBody(resultFileReadStream.toString("base64"), "", response, 200);
                } else {
                    helperSrc.writeLog("Ocr.ts - api() - post(/api/download) - fileReadStream()", resultFileReadStream.toString());

                    helperSrc.responseBody("", resultFileReadStream.toString(), response, 500);
                }
            });
        });
    };
}
