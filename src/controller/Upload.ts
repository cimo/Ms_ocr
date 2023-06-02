import Express from "express";
import * as FormDataParser from "@cimo/form-data_parser";

// Source
import * as ControllerHelper from "../controller/Helper";

const checkRequest = (formDataList: FormDataParser.Iinput[]): boolean => {
    const parameterList: string[] = [];
    let tokenWrong = false;
    let fileProblem = "";
    let parameterNotFound = "";

    for (const value of formDataList) {
        if (value.name === "token_api") {
            if (!ControllerHelper.checkToken(value.buffer.toString())) {
                tokenWrong = true;
            }
        }

        if (value.name === "file") {
            if (value.filename === "" || value.mimeType === "" || value.size === "") {
                fileProblem = "empty";
            } else if (!ControllerHelper.checkMymeType(value.mimeType)) {
                fileProblem = "mimeType";
            } else if (!ControllerHelper.checkFileSize(value.size)) {
                fileProblem = "size";
            }
        }

        parameterList.push(value.name);
    }

    if (!parameterList.includes("token_api")) {
        parameterNotFound = "token_api";
    }

    if (!parameterList.includes("file_name")) {
        parameterNotFound = "file_name";
    }

    if (!parameterList.includes("file")) {
        parameterNotFound = "file";
    }

    // Result
    const result = tokenWrong === false && fileProblem === "" && parameterNotFound === "" ? true : false;

    if (!result) {
        ControllerHelper.writeLog(
            "Upload.ts - checkRequest",
            `tokenWrong: ${tokenWrong.toString()} - fileProblem: ${fileProblem.toString()} - parameterNotFound: ${parameterNotFound}`
        );
    }

    return result;
};

export const execute = (request: Express.Request): Promise<string> => {
    return new Promise((resolve, reject) => {
        const chunkList: Buffer[] = [];

        request.on("data", (data: Buffer) => {
            chunkList.push(data);
        });

        request.on("end", () => {
            void (async () => {
                const buffer = Buffer.concat(chunkList);
                const formDataList = FormDataParser.readInput(buffer, request.headers["content-type"]);

                const check = checkRequest(formDataList);

                if (check) {
                    for (const value of formDataList) {
                        if (value.name === "file" && value.filename && value.buffer) {
                            const input = `${ControllerHelper.PATH_FILE_INPUT}${value.filename}`;

                            await ControllerHelper.fileWriteStream(input, value.buffer)
                                .then(() => {
                                    resolve(input);
                                })
                                .catch((error: Error) => {
                                    ControllerHelper.writeLog(
                                        "Upload.ts - ControllerHelper.fileWriteStream() - catch error: ",
                                        ControllerHelper.objectOutput(error)
                                    );
                                });

                            break;
                        }
                    }
                } else {
                    reject(check);
                }
            })();
        });

        request.on("error", (error: Error) => {
            ControllerHelper.writeLog("Upload.ts - execute() - error: ", ControllerHelper.objectOutput(error));

            reject(error);
        });
    });
};
