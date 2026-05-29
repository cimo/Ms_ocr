import Express, { Request, Response } from "express";
import { RateLimitRequestHandler } from "express-rate-limit";
import { execFile } from "child_process";
import { Ca } from "@cimo/authentication/dist/src/Main.js";

// Source
import * as helperSrc from "../HelperSrc.js";
import ControllerUpload from "./Upload.js";

export default class Ocr {
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
                .execute(request, true, false, `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/`)
                .then((resultControllerUploadList) => {
                    let fileName = "";
                    let language = "";
                    let searchText = "";
                    let mode = "";

                    for (let a = 0; a < resultControllerUploadList.length; a++) {
                        const resultControllerUpload = resultControllerUploadList[a];

                        if (resultControllerUpload.name === "language" && resultControllerUpload.buffer) {
                            language = resultControllerUpload.buffer.toString().match("^(-|ja|ja_vert|en)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        } else if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                            fileName = resultControllerUpload.fileName;
                        } else if (resultControllerUpload.name === "searchText" && resultControllerUpload.buffer) {
                            searchText = resultControllerUpload.buffer.toString();
                        } else if (resultControllerUpload.name === "mode" && resultControllerUpload.buffer) {
                            mode = resultControllerUpload.buffer.toString();
                        }
                    }

                    const fileDetail = helperSrc.fileDetail(fileName);

                    const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${fileDetail.baseName}/`;

                    const uniqueId = helperSrc.generateUniqueId();

                    const execCommand = `${helperSrc.PATH_ROOT}${helperSrc.PATH_SCRIPT}command1.sh`;
                    const execArgumentList = [execCommand, language, `${fileDetail.baseName}/${fileDetail.fileName}`, uniqueId, searchText, mode];

                    execFile("/bin/bash", execArgumentList, { encoding: "utf8" }, (error, stdout) => {
                        if (error) {
                            helperSrc.writeLog(`Ocr.ts - api() - post(/api/extract) - execFile() - error`, error.message);

                            helperSrc.responseBody("", error.message, response, 500);

                            return;
                        }

                        const output = stdout.trim();

                        if (output.startsWith("file")) {
                            helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - execFile() - stdout", stdout);

                            const baseExport = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/export/`;
                            const baseTable = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/table/`;

                            let dataJsonList: string[] | undefined;
                            let dataPdfList: string[] | undefined;
                            let dataXlsxList: string[] | undefined;
                            let dataHtmlList: string[] | undefined;

                            let isCompleted = false;

                            const finalizeResponse = () => {
                                if (isCompleted) {
                                    return;
                                }

                                const isAllReady =
                                    Array.isArray(dataJsonList) &&
                                    Array.isArray(dataPdfList) &&
                                    Array.isArray(dataXlsxList) &&
                                    Array.isArray(dataHtmlList);

                                if (!isAllReady) {
                                    return;
                                }

                                const jsonList: string[] = [];
                                for (let a = 0; a < dataJsonList!.length; a++) {
                                    const dataJson = dataJsonList![a];

                                    jsonList.push(
                                        dataJson.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const pdfList: string[] = [];
                                for (let a = 0; a < dataPdfList!.length; a++) {
                                    const dataPdf = dataPdfList![a];

                                    pdfList.push(
                                        dataPdf.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const excelList: string[] = [];
                                for (let a = 0; a < dataXlsxList!.length; a++) {
                                    const dataXlsx = dataXlsxList![a];

                                    excelList.push(
                                        dataXlsx.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const htmlList: string[] = [];
                                for (let a = 0; a < dataHtmlList!.length; a++) {
                                    const dataHtml = dataHtmlList![a];

                                    htmlList.push(
                                        dataHtml.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const responseJson = {
                                    uniqueId,
                                    jsonList,
                                    pdfList,
                                    excelList,
                                    htmlList
                                };

                                isCompleted = true;

                                helperSrc.responseBody(JSON.stringify(responseJson), "", response, 200);
                            };

                            helperSrc.findInDirectoryRecursive(baseExport, ".json", (list) => {
                                dataJsonList = list || [];

                                finalizeResponse();
                            });

                            helperSrc.findInDirectoryRecursive(baseExport, ".pdf", (list) => {
                                dataPdfList = list || [];

                                finalizeResponse();
                            });

                            helperSrc.findInDirectoryRecursive(baseTable, ".xlsx", (list) => {
                                dataXlsxList = list || [];

                                finalizeResponse();
                            });

                            helperSrc.findInDirectoryRecursive(baseTable, ".html", (list) => {
                                dataHtmlList = list || [];

                                finalizeResponse();
                            });
                        } else if (output.startsWith("data")) {
                            const outputSlice = output.slice("data".length).trim();

                            helperSrc.responseBody(outputSlice, "", response, 200);
                        } else {
                            helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - execFile() - Error", "Problem with the output.");

                            helperSrc.responseBody("", "ko", response, 500);
                        }

                        helperSrc.fileOrFolderDelete(input, (resultFileDelete) => {
                            if (typeof resultFileDelete !== "boolean") {
                                helperSrc.writeLog(
                                    "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileOrFolderDelete()",
                                    resultFileDelete.toString()
                                );
                            }
                        });
                    });
                })
                .catch((error: Error) => {
                    helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - catch()", error.message);

                    helperSrc.responseBody("", "ko", response, 500);
                });
        });

        this.app.post("/api/download", this.limiter, Ca.authenticationMiddleware, (request: Request, response: Response) => {
            const requestBody = request.body;

            const uniqueId = requestBody.uniqueId;
            const pathFile = requestBody.pathFile;

            const path = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/${pathFile}`;

            helperSrc.fileReadStream(path, (resultFileReadStream) => {
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
