import Express from "express";
import { exec } from "child_process";

// Source
import * as ControllerHelper from "../controller/Helper";
import * as ControllerUpload from "../controller/Upload";

export const execute = (app: Express.Express): void => {
    app.post("/msocr/extract", (request: Express.Request, response: Express.Response) => {
        void (async () => {
            await ControllerUpload.execute(request, true)
                .then((resultList) => {
                    let fileName = "";
                    let language = "";
                    let result = "";

                    for (const value of resultList) {
                        if (value.name === "file" && value.filename) {
                            fileName = value.filename;
                        } else if (value.name === "language" && value.buffer) {
                            language = value.buffer.toString().match("^(jp|jp_vert|en)$") ? value.buffer.toString() : "";
                        } else if (value.name === "result" && value.buffer) {
                            result = value.buffer.toString().match("^(txt|hocr|tsv|pdf)$") ? value.buffer.toString() : "";
                        }
                    }

                    const input = `${ControllerHelper.PATH_FILE_INPUT}${fileName}`;
                    const output1 = `${ControllerHelper.PATH_FILE_OUTPUT}${fileName}`;
                    const output2 = `${ControllerHelper.PATH_FILE_OUTPUT}${fileName}.${result}`;

                    exec(
                        `python3 '/home/root/src/library/preprocess.py' '${fileName}' '${language}' '${result}' '${ControllerHelper.DEBUG}'`,
                        (error, stdout, stderr) => {
                            void (async () => {
                                if ((stdout !== "" && stderr === "") || (stdout !== "" && stderr !== "")) {
                                    await ControllerHelper.fileRemove(input)
                                        .then()
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileRemove(input) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );
                                        });

                                    await ControllerHelper.fileRemove(output1)
                                        .then()
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileRemove(output1) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );
                                        });

                                    await ControllerHelper.fileRemove(output2)
                                        .then()
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileRemove(output2) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );
                                        });

                                    ControllerHelper.responseBody(stdout, stderr, response, 500);
                                } else if (stdout === "" && stderr !== "") {
                                    ControllerHelper.writeLog("Ocr.ts - python3 /home/root/... - stderr", stderr);

                                    await ControllerHelper.fileReadStream(output2)
                                        .then((buffer) => {
                                            ControllerHelper.responseBody("", buffer.toString("base64"), response, 200);
                                        })
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileReadStream(output2) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );

                                            ControllerHelper.responseBody("", error, response, 500);
                                        });

                                    await ControllerHelper.fileRemove(input)
                                        .then()
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileRemove(input) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );
                                        });

                                    await ControllerHelper.fileRemove(output1)
                                        .then()
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileRemove(output1) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );
                                        });

                                    await ControllerHelper.fileRemove(output2)
                                        .then()
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileRemove(output2) - catch error: ",
                                                ControllerHelper.objectOutput(error)
                                            );
                                        });
                                }
                            })();
                        }
                    );
                })
                .catch((error: Error) => {
                    ControllerHelper.writeLog("Ocr.ts - ControllerUpload.execute() - catch error: ", ControllerHelper.objectOutput(error));

                    ControllerHelper.responseBody("", error, response, 500);
                });
        })();
    });
};
