import Express from "express";
import Path from "path";
import { exec } from "child_process";

// Source
import * as ControllerHelper from "../controller/Helper";
import * as ControllerUpload from "../controller/Upload";

const removeFile = (input: string, output: string, response: Express.Response) => {
    void (async () => {
        await ControllerHelper.fileRemove(input)
            .then()
            .catch((error: Error) => {
                ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove() - input catch error: ", ControllerHelper.objectOutput(error));

                ControllerHelper.responseBody("", error, response, 500);
            });

        await ControllerHelper.fileRemove(output)
            .then()
            .catch((error: Error) => {
                ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove() - output catch error: ", ControllerHelper.objectOutput(error));

                ControllerHelper.responseBody("", error, response, 500);
            });
    })();
};

const outputFile = (input: string, output: string, response: Express.Response) => {
    void (async () => {
        await ControllerHelper.fileReadStream(output)
            .then((buffer) => {
                removeFile(input, output, response);

                ControllerHelper.responseBody(buffer.toString("base64"), "", response, 200);
            })
            .catch((error: Error) => {
                ControllerHelper.writeLog("Converter.ts - ControllerHelper.fileReadStream - catch error", ControllerHelper.objectOutput(error));

                removeFile(input, output, response);

                ControllerHelper.responseBody("", error, response, 500);
            });
    })();
};

export const execute = (app: Express.Express): void => {
    app.post("/msocr/extract", (request: Express.Request, response: Express.Response) => {
        void (async () => {
            await ControllerUpload.execute(request)
                .then((result) => {
                    const fileName = Path.parse(result).name;
                    const output = `${ControllerHelper.PATH_FILE_OUTPUT}${fileName}.pdf`;

                    exec(
                        `tesseract "${result}" "${output.split(".")[0]}" -l eng+jpn+lat --psm 3 --oem 1 --dpi 600 -c preserve_interword_spaces=1 txt`,
                        (error, stdout, stderr) => {
                            if (stdout !== "" && stderr === "") {
                                outputFile(result, output, response);
                            } else if (stdout === "" && stderr !== "") {
                                ControllerHelper.writeLog("Ocr.ts - exec('tesseract... - stderr", stderr);

                                removeFile(result, output, response);
                            } else {
                                outputFile(result, output, response);
                            }
                        }
                    );
                })
                .catch((error: Error) => {
                    ControllerHelper.writeLog("Ocr.ts - msocr/extract - catch error: ", ControllerHelper.objectOutput(error));

                    ControllerHelper.responseBody("", error, response, 500);
                });
        })();
    });
};