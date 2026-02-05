import { Request } from "express";
import Fs from "fs";
import { Cfdp, CfdpModel } from "@cimo/form-data_parser/dist/src/Main.js";

// Source
import * as helperSrc from "../HelperSrc.js";

export default class Upload {
    // Variable

    // Method
    constructor() {
        //...
    }

    execute = (request: Request, isFileExists: boolean): Promise<CfdpModel.Iinput[]> => {
        return new Promise((resolve, reject) => {
            const chunkList: Buffer[] = [];

            request.on("data", (data: Buffer) => {
                chunkList.push(data);
            });

            request.on("end", () => {
                const buffer = Buffer.concat(chunkList);
                const formDataList = Cfdp.readInput(buffer, request.headers["content-type"]);

                const resultCheckRequest = this.checkRequest(formDataList);

                if (resultCheckRequest === "") {
                    for (const formData of formDataList) {
                        if (formData.name === "file" && formData.fileName && formData.buffer) {
                            const input = `${helperSrc.PATH_ROOT}${helperSrc.PATH_FILE}input/${formData.fileName}`;

                            if (isFileExists) {
                                Fs.access(input, Fs.constants.F_OK, (error) => {
                                    if (!error) {
                                        reject("File exists.");

                                        return;
                                    } else {
                                        helperSrc.fileWriteStream(input, formData.buffer, (resultFileWriteStream) => {
                                            if (resultFileWriteStream) {
                                                resolve(formDataList);

                                                return;
                                            } else {
                                                reject(resultFileWriteStream);

                                                return;
                                            }
                                        });
                                    }
                                });
                            } else {
                                helperSrc.fileWriteStream(input, formData.buffer, (resultFileWriteStream) => {
                                    if (resultFileWriteStream) {
                                        resolve(formDataList);

                                        return;
                                    } else {
                                        reject(resultFileWriteStream);

                                        return;
                                    }
                                });
                            }

                            break;
                        }
                    }
                } else {
                    reject(resultCheckRequest);

                    return;
                }
            });

            request.on("error", (error: Error) => {
                reject(error);

                return;
            });
        });
    };

    private checkRequest = (formDataList: CfdpModel.Iinput[]): string => {
        let result = "";

        const parameterList: string[] = [];

        for (const formData of formDataList) {
            parameterList.push(formData.name);

            if (formData.name === "file") {
                if (formData.fileName === "" || formData.mimeType === "" || formData.size === "") {
                    result += "File input empty.";
                } else if (!helperSrc.fileCheckMimeType(formData.mimeType)) {
                    result += "Mime type are not allowed.";
                } else if (!helperSrc.fileCheckSize(parseInt(formData.size))) {
                    result += "File size exceeds limit.";
                }
            }
        }

        if (!parameterList.includes("file")) {
            result += "Parameter 'file' is missing.";
        }

        return result;
    };
}
