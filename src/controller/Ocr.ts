import Express, { Request, Response } from "express";
import { RateLimitRequestHandler } from "express-rate-limit";
import { execFile } from "child_process";
import { Ca } from "@cimo/authentication/dist/src/Main.js";

// Source
import * as helperSrc from "../HelperSrc.js";
import ControllerUpload from "./Upload.js";

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

    header = (request: Request, response: Response): void => {
        const sessionId = request.headers["x-session-id"] as string;
        const endpoint = request.headers["x-endpoint"] as string;

        response.setHeader("X-Session-Id", sessionId);
        response.setHeader("X-Endpoint", endpoint);
    };

    api = (): void => {
        this.app.post("/api/extract", this.limiter, Ca.authenticationMiddleware, (request: Request, response: Response) => {
            this.header(request, response);

            this.controllerUpload
                .execute(request, true)
                .then((resultControllerUploadList) => {
                    let fileName = "";
                    let language = "";
                    let searchText = "";
                    let dataType = "";

                    for (const resultControllerUpload of resultControllerUploadList) {
                        if (resultControllerUpload.name === "language" && resultControllerUpload.buffer) {
                            language = resultControllerUpload.buffer.toString().match("^(ja|ja_vert|en)$")
                                ? resultControllerUpload.buffer.toString()
                                : "";
                        } else if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                            fileName = resultControllerUpload.fileName;
                        } else if (resultControllerUpload.name === "searchText" && resultControllerUpload.buffer) {
                            searchText = resultControllerUpload.buffer.toString();
                        } else if (resultControllerUpload.name === "dataType" && resultControllerUpload.buffer) {
                            dataType = resultControllerUpload.buffer.toString();
                        }
                    }

                    const uniqueId = helperSrc.generateUniqueId();

                    const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${fileName}`;

                    const execCommand = `. ${helperSrc.PATH_ROOT}${helperSrc.PATH_SCRIPT}command1.sh`;
                    const execArgumentList = [`"${language}"`, `"${fileName}"`, `"${uniqueId}"`, `"${searchText}"`, `"${dataType}"`];

                    execFile(execCommand, execArgumentList, { shell: "/bin/bash", encoding: "utf8" }, (_, stdout) => {
                        helperSrc.fileOrFolderRemove(input, (resultFileRemove) => {
                            if (typeof resultFileRemove !== "boolean") {
                                helperSrc.writeLog(
                                    "Ocr.ts - api() - post(/api/extract) - execute() - execFile() - fileOrFolderRemove(input)",
                                    resultFileRemove.toString()
                                );

                                helperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                            }
                        });

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
                                for (const dataJson of dataJsonList!) {
                                    jsonList.push(
                                        dataJson.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const pdfList: string[] = [];
                                for (const dataPdf of dataPdfList!) {
                                    pdfList.push(
                                        dataPdf.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const excelList: string[] = [];
                                for (const dataXlsx of dataXlsxList!) {
                                    excelList.push(
                                        dataXlsx.replace(`${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${helperSrc.RUNTIME}/${uniqueId}/`, "")
                                    );
                                }

                                const htmlList: string[] = [];
                                for (const dataHtml of dataHtmlList!) {
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

                            helperSrc.findFileInDirectoryRecursive(baseExport, ".json", (list) => {
                                dataJsonList = list || [];

                                finalizeResponse();
                            });

                            helperSrc.findFileInDirectoryRecursive(baseExport, ".pdf", (list) => {
                                dataPdfList = list || [];

                                finalizeResponse();
                            });

                            helperSrc.findFileInDirectoryRecursive(baseTable, ".xlsx", (list) => {
                                dataXlsxList = list || [];

                                finalizeResponse();
                            });

                            helperSrc.findFileInDirectoryRecursive(baseTable, ".html", (list) => {
                                dataHtmlList = list || [];

                                finalizeResponse();
                            });
                        } else if (output.startsWith("polygon")) {
                            const outputSlice = output.slice("polygon".length).trim();
                            const polygon = JSON.parse(outputSlice);

                            const responseJson = { polygon };

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
            this.header(request, response);

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
