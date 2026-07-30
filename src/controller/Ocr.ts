import Express, { Request, Response } from "express";
import { RateLimitRequestHandler } from "express-rate-limit";
import { Ca } from "@cimo/authentication/dist/src/Main.js";

// Source
import * as helperSrc from "../HelperSrc.js";
import * as instance from "../Instance.js";
import * as modelOcr from "../model/Ocr.js";
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
                    let searchText = "";

                    for (let a = 0; a < resultControllerUploadList.length; a++) {
                        const resultControllerUpload = resultControllerUploadList[a];

                        if (resultControllerUpload.name === "file" && resultControllerUpload.fileName) {
                            fileName = resultControllerUpload.fileName;
                        } else if (resultControllerUpload.name === "searchText" && resultControllerUpload.buffer) {
                            searchText = resultControllerUpload.buffer.toString();
                        }
                    }

                    const fileDetail = helperSrc.fileDetail(fileName);

                    const uniqueId = helperSrc.generateUniqueId();

                    const pathInput = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${fileDetail.baseName}/${fileDetail.fileName}`;
                    const pathInputBasename = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${fileDetail.baseName}/`;
                    const pathOutput = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${uniqueId}/`;

                    instance.api
                        .post<modelOcr.IapiOnnxResponse>(
                            "/scanner",
                            {
                                headers: {
                                    "Content-Type": "application/json"
                                }
                            },
                            { pathInput, pathOutput, searchText }
                        )
                        .then(async (resultApi) => {
                            const data = resultApi.data;

                            helperSrc.responseBody(
                                JSON.stringify({ uniqueId, layoutList: data.layoutList, itemList: data.itemList }),
                                "",
                                response,
                                200
                            );

                            const fileOrFolderDelete = await helperSrc.fileOrFolderDelete(pathInputBasename);

                            if (typeof fileOrFolderDelete !== "boolean") {
                                helperSrc.writeLog(
                                    "Ocr.ts - api() - post(/api/extract) - post(/scanner) - fileOrFolderDelete()",
                                    fileOrFolderDelete.toString()
                                );
                            }
                        })
                        .catch((error: Error) => {
                            helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - post(/scanner) - catch()", error.message);

                            helperSrc.responseBody("", "ko", response, 500);
                        });
                })
                .catch((error: Error) => {
                    helperSrc.writeLog("Ocr.ts - api() - post(/api/extract) - execute() - catch()", error.message);

                    helperSrc.responseBody("", "ko", response, 500);
                });
        });

        this.app.post("/api/download", this.limiter, Ca.authenticationMiddleware, (request: Request, response: Response) => {
            const body = request.body as modelOcr.IapiDownloadBody;

            const uniqueId = body.uniqueId;
            const pathFile = body.pathFile;

            const path = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}output/${uniqueId}/${pathFile}`;

            helperSrc.fileReadStream(path).then((resultFileReadStream) => {
                if (Buffer.isBuffer(resultFileReadStream)) {
                    helperSrc.responseBody(resultFileReadStream.toString("base64"), "", response, 200);
                } else {
                    helperSrc.writeLog("Ocr.ts - api() - post(/api/download) - fileReadStream()", resultFileReadStream.toString());

                    helperSrc.responseBody("", "ko", response, 500);
                }
            });
        });
    };
}
