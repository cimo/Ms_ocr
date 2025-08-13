import Express, { Request, Response } from "express";
import { execFile } from "child_process";
import { Ca } from "@cimo/authentication/dist/src/Main";

// Source
import * as HelperSrc from "../HelperSrc";
import ControllerUpload from "./Upload";

export default class ControllerAntivirus {
    // Variable
    private app: Express.Express;
    private controllerUpload: ControllerUpload;

    // Method
    constructor(app: Express.Express) {
        this.app = app;
        this.controllerUpload = new ControllerUpload();
    }

    api = (): void => {
        this.app.post("/api/extract", Ca.authenticationMiddleware, (request: Request, response: Response) => {
            void (async () => {
                await this.controllerUpload
                    .execute(request, true)
                    .then((resultControllerUploadList) => {
                        let filename = "";
                        let language = "";
                        let result = "";
                        let debug = "";

                        for (const resultControllerUpload of resultControllerUploadList) {
                            if (resultControllerUpload.name === "file" && resultControllerUpload.filename) {
                                filename = resultControllerUpload.filename;
                            } else if (resultControllerUpload.name === "language" && resultControllerUpload.buffer) {
                                language = resultControllerUpload.buffer.toString().match("^(jp|jp_vert|en)$")
                                    ? resultControllerUpload.buffer.toString()
                                    : "";
                            } else if (resultControllerUpload.name === "result" && resultControllerUpload.buffer) {
                                result = resultControllerUpload.buffer.toString().match("^(txt|hocr|tsv|pdf)$")
                                    ? resultControllerUpload.buffer.toString()
                                    : "";
                            } else if (resultControllerUpload.name === "debug" && resultControllerUpload.buffer) {
                                debug = resultControllerUpload.buffer.toString().match("^(true|false)$")
                                    ? resultControllerUpload.buffer.toString()
                                    : "";
                            }
                        }

                        const input = `${HelperSrc.PATH_ROOT}${HelperSrc.PATH_FILE_INPUT}${filename}`;
                        const outputResult = `${HelperSrc.PATH_ROOT}${HelperSrc.PATH_FILE_OUTPUT}${filename}_result.png`;
                        const output = `${HelperSrc.PATH_ROOT}${HelperSrc.PATH_FILE_OUTPUT}${filename}.${result}`;

                        const execCommand = `. ${HelperSrc.PATH_ROOT}${HelperSrc.PATH_FILE_SCRIPT}command1.sh`;
                        const execArgumentList = [`"${filename}"`, `"${language}"`, `"${result}"`, `"${debug}"`];

                        execFile(execCommand, execArgumentList, { shell: "/bin/bash", encoding: "utf8" }, (_, stdout, stderr) => {
                            // Tesseract use always stderr for the output
                            if (stdout === "" || stderr !== "" || (stdout !== "" && stderr !== "")) {
                                HelperSrc.fileReadStream(output, (resultFileReadStream) => {
                                    if (Buffer.isBuffer(resultFileReadStream)) {
                                        HelperSrc.responseBody(resultFileReadStream.toString("base64"), "", response, 200);
                                    } else {
                                        HelperSrc.writeLog(
                                            "Ocr.ts - api() => post(/api/extract) => execute() => execFile(python3) => fileReadStream()",
                                            resultFileReadStream.toString()
                                        );

                                        HelperSrc.responseBody("", resultFileReadStream.toString(), response, 500);
                                    }

                                    HelperSrc.fileRemove(input, (resultFileRemove) => {
                                        if (typeof resultFileRemove !== "boolean") {
                                            HelperSrc.writeLog(
                                                "Ocr.ts - api() => post(/api/extract) => execute() => execFile(python3) => fileReadStream() => fileRemove(input)",
                                                resultFileRemove.toString()
                                            );

                                            HelperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                        }
                                    });

                                    HelperSrc.fileRemove(outputResult, (resultFileRemove) => {
                                        if (typeof resultFileRemove !== "boolean") {
                                            HelperSrc.writeLog(
                                                "Ocr.ts - api() => post(/api/extract) => execute() => execFile(python3) => fileReadStream() => fileRemove(outputResult)",
                                                resultFileRemove.toString()
                                            );

                                            HelperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                        }
                                    });

                                    HelperSrc.fileRemove(output, (resultFileRemove) => {
                                        if (typeof resultFileRemove !== "boolean") {
                                            HelperSrc.writeLog(
                                                "Ocr.ts - api() => post(/api/extract) => execute() => execFile(python3) => fileReadStream() => fileRemove(output)",
                                                resultFileRemove.toString()
                                            );

                                            HelperSrc.responseBody("", resultFileRemove.toString(), response, 500);
                                        }
                                    });
                                });
                            } else if (stdout !== "" || stderr === "") {
                                HelperSrc.writeLog("Ocr.ts - api() => post(/api/extract) => execute() => execFile(python3) => stdout", stdout);

                                HelperSrc.fileRemove(input, (resultFileRemove) => {
                                    if (typeof resultFileRemove !== "boolean") {
                                        stderr += resultFileRemove;

                                        HelperSrc.writeLog(
                                            "Ocr.ts - api() => post(/api/extract) => execute() => execFile(python3) => fileRemove(input)",
                                            resultFileRemove.toString()
                                        );
                                    }
                                });

                                HelperSrc.responseBody(stdout, "", response, 500);
                            }
                        });
                    })
                    .catch((error: Error) => {
                        HelperSrc.writeLog("Ocr.ts - api() => post(/api/extract) => execute() => catch()", error);

                        HelperSrc.responseBody("", error, response, 500);
                    });
            })();
        });
    };
}
