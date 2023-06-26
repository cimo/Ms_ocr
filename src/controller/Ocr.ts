import Express from "express";
import { exec } from "child_process";

// Source
import * as ControllerHelper from "../controller/Helper";
import * as ControllerUpload from "../controller/Upload";

const removeFile = (input: string, output: string) => {
    return new Promise((resolve, reject) => {
        void (async () => {
            if (input !== "") {
                await ControllerHelper.fileRemove(input)
                    .then(() => {
                        resolve("");
                    })
                    .catch((error: Error) => {
                        ControllerHelper.writeLog(
                            "Ocr.ts - ControllerHelper.fileRemove() - input catch error: ",
                            ControllerHelper.objectOutput(error)
                        );

                        reject(error);
                    });
            }

            await ControllerHelper.fileRemove(output)
                .then(() => {
                    resolve("");
                })
                .catch((error: Error) => {
                    ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove() - output catch error: ", ControllerHelper.objectOutput(error));

                    reject(error);
                });
        })();
    });
};

export const execute = (app: Express.Express): void => {
    app.post("/msocr/extract", (request: Express.Request, response: Express.Response) => {
        void (async () => {
            await ControllerUpload.execute(request)
                .then((resultList) => {
                    let fileName = "";
                    let language = "";

                    for (const value of resultList) {
                        if (value.name === "file" && value.filename) {
                            fileName = value.filename;
                        } else if (value.name === "language" && value.buffer) {
                            language = value.buffer.toString().match("^(jp|jp_vert|en)$") ? value.buffer.toString() : "";
                        }
                    }

                    const input = `${ControllerHelper.PATH_FILE_INPUT}${fileName}`;
                    const output1 = `${ControllerHelper.PATH_FILE_OUTPUT}${fileName}`;
                    const output2 = `${ControllerHelper.PATH_FILE_OUTPUT}${fileName}.txt`;

                    exec(`python3 '/home/root/src/library/preprocess.py' '${fileName}' '${language}'`, (error, stdout, stderr) => {
                        void (async () => {
                            if (stdout !== "" && stderr === "") {
                                await removeFile(input, output1).catch((error: Error) => {
                                    ControllerHelper.responseBody("", error, response, 500);
                                });

                                await removeFile("", output2).catch((error: Error) => {
                                    ControllerHelper.responseBody("", error, response, 500);
                                });
                            } else if (stdout === "" && stderr !== "") {
                                ControllerHelper.writeLog("Ocr.ts - python3 /home/root/... - stderr", stderr);

                                ControllerHelper.responseBody(stdout, stderr, response, 200);

                                // Commented for do the test
                                /*await removeFile(input, output1)
                                    .then(() => {
                                        ControllerHelper.responseBody(stdout, stderr, response, 200);
                                    })
                                    .catch((error: Error) => {
                                        ControllerHelper.responseBody("", error, response, 500);
                                    });

                                await removeFile("", output2).catch((error: Error) => {
                                    ControllerHelper.responseBody("", error, response, 500);
                                });*/
                            } else {
                                await removeFile(input, output1).catch((error: Error) => {
                                    ControllerHelper.responseBody("", error, response, 500);
                                });

                                await removeFile("", output2).catch((error: Error) => {
                                    ControllerHelper.responseBody("", error, response, 500);
                                });
                            }
                        })();
                    });
                })
                .catch((error: Error) => {
                    ControllerHelper.writeLog("Ocr.ts - msocr/extract - catch error: ", ControllerHelper.objectOutput(error));

                    ControllerHelper.responseBody("", error, response, 500);
                });
        })();
    });
};
