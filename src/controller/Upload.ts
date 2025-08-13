import { Request } from "express";
import Fs from "fs";
import { Cfdp, CfdpModel } from "@cimo/form-data_parser/dist/src/Main";

// Source
import * as HelperSrc from "../HelperSrc";

export default class ControllerUpload {
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
                        if (formData.name === "file" && formData.filename && formData.buffer) {
                            const input = `${HelperSrc.PATH_ROOT}${HelperSrc.PATH_FILE_INPUT}${formData.filename}`;

                            if (isFileExists && Fs.existsSync(input)) {
                                reject("File exists.");
                            } else {
                                HelperSrc.fileWriteStream(input, formData.buffer, (resultFileWriteStream) => {
                                    if (resultFileWriteStream) {
                                        resolve(formDataList);
                                    } else {
                                        reject(resultFileWriteStream);
                                    }
                                });
                            }

                            break;
                        }
                    }
                } else {
                    reject(resultCheckRequest);
                }
            });

            request.on("error", (error: Error) => {
                reject(error);
            });
        });
    };

    private checkRequest = (formDataList: CfdpModel.Iinput[]): string => {
        let result = "";

        const parameterList: string[] = [];

        for (const formData of formDataList) {
            parameterList.push(formData.name);

            if (formData.name === "file") {
                if (formData.filename === "" || formData.mimeType === "" || formData.size === "") {
                    result += "File input empty.";
                } else if (!HelperSrc.fileCheckMimeType(formData.mimeType)) {
                    result += "Mime type are not allowed.";
                } else if (!HelperSrc.fileCheckSize(formData.size)) {
                    result += "File size exceeds limit.";
                }
            }
        }

        if (!parameterList.includes("filename")) {
            result += "Filename parameter missing.";
        } else if (!parameterList.includes("file")) {
            result += "File parameter missing.";
        }

        return result;
    };
}
