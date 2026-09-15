import Express, { Request, Response } from "express";
import { RateLimitRequestHandler } from "express-rate-limit";
import { Ca } from "@cimo/authentication/dist/src/Main.js";

// Source
import * as helperSrc from "../HelperSrc.js";
import * as instance from "../Instance.js";
import * as modelService from "../model/Service.js";
import ControllerUpload from "./Upload.js";

export default class Service {
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
                .then(async (resultControllerUploadList) => {
                    let fileName = "";
                    let password = "";
                    let searchText = "";

                    for (let a = 0; a < resultControllerUploadList.length; a++) {
                        const resultControllerUpload = resultControllerUploadList[a];

                        if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                            fileName = resultControllerUpload.fileName;
                        } else if (resultControllerUpload.name === "password" && resultControllerUpload.buffer) {
                            password = resultControllerUpload.buffer.toString();
                        } else if (resultControllerUpload.name === "searchText" && resultControllerUpload.buffer) {
                            searchText = resultControllerUpload.buffer.toString();
                        }
                    }

                    const fileDetail = await helperSrc.fileDetail(fileName);

                    const uniqueId = helperSrc.generateUniqueId();

                    const pathInput = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${fileDetail.baseName}/${fileDetail.name}`;
                    const pathInputBasename = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${fileDetail.baseName}/`;
                    const pathOutput = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${uniqueId}/`;

                    instance.api
                        .post<modelService.IapiResponse>(
                            "/engine",
                            {
                                headers: {
                                    "Content-Type": "application/json"
                                }
                            },
                            { pathInput, pathOutput, password, searchText }
                        )
                        .then(async (resultApi) => {
                            const data = resultApi.data;

                            if (data.response.state === "ok") {
                                const fileReadStream = await helperSrc.fileReadStream(`${pathOutput}result.md`);

                                if (!Buffer.isBuffer(fileReadStream)) {
                                    helperSrc.writeLog(
                                        "Service.ts - api() - post(/api/extract) - post(/engine) - fileReadStream()",
                                        fileReadStream.toString()
                                    );

                                    helperSrc.responseBody({ state: "ko", message: fileReadStream.toString() }, response, 500);
                                } else {
                                    helperSrc.responseBody({ state: "ok", message: "", data: fileReadStream.toString("base64") }, response, 200);
                                }
                            } else {
                                helperSrc.responseBody({ state: data.response.state, message: data.response.message }, response, 200);
                            }

                            const fileOrFolderDeleteInput = await helperSrc.fileOrFolderDelete(pathInputBasename);

                            if (typeof fileOrFolderDeleteInput !== "boolean") {
                                helperSrc.writeLog(
                                    "Service.ts - api() - post(/api/extract) - post(/engine) - fileOrFolderDelete()",
                                    fileOrFolderDeleteInput.toString()
                                );
                            }

                            const fileOrFolderDeleteOutput = await helperSrc.fileOrFolderDelete(pathOutput);

                            if (typeof fileOrFolderDeleteOutput !== "boolean") {
                                helperSrc.writeLog(
                                    "Service.ts - api() - post(/api/extract) - post(/engine) - fileOrFolderDelete()",
                                    fileOrFolderDeleteOutput.toString()
                                );
                            }
                        })
                        .catch(async (error: Error) => {
                            helperSrc.writeLog("Service.ts - api() - post(/api/extract) - post(/engine) - catch()", error.message);

                            helperSrc.responseBody({ state: "ko", message: error.message }, response, 500);

                            const fileOrFolderDelete = await helperSrc.fileOrFolderDelete(pathInputBasename);

                            if (typeof fileOrFolderDelete !== "boolean") {
                                helperSrc.writeLog(
                                    "Service.ts - api() - post(/api/extract) - post(/engine) - catch() - fileOrFolderDelete()",
                                    fileOrFolderDelete.toString()
                                );
                            }
                        });
                })
                .catch((error: Error) => {
                    helperSrc.writeLog("Service.ts - api() - post(/api/extract) - execute() - catch()", error.message);

                    helperSrc.responseBody({ state: "ko", message: error.message }, response, 500);
                });
        });
    };
}
