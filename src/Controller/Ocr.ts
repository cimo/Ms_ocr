import Express from "express";
import { exec } from "child_process";

// Source
import * as ControllerHelper from "../Controller/Helper";
import * as ControllerUpload from "../Controller/Upload";

const removeFile = (input: string | undefined, output: string | undefined) => {
    if (input) {
        ControllerHelper.fileRemove(input)
            .then(() => {
                ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove - input", input);
            })
            .catch((error: Error) => {
                ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove - error", ControllerHelper.objectOutput(error));
            });
    }

    if (output) {
        ControllerHelper.fileRemove(output)
            .then(() => {
                ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove - output", output);
            })
            .catch((error: Error) => {
                ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileRemove - error", ControllerHelper.objectOutput(error));
            });
    }
};

export const execute = (app: Express.Express): void => {
    app.post("/msocr/extract", (request: Express.Request, response: Express.Response) => {
        void (async () => {
            await ControllerUpload.execute(request)
                .then((result) => {
                    const input = result.input;
                    const output = result.output;

                    exec(
                        `tesseract "${input}" "${output.split(".")[0]}" -l jpn+eng+lat --psm 3 --oem 1 --dpi 600 -c preserve_interword_spaces=1 txt`,
                        (error, stdout, stderr) => {
                            void (async () => {
                                if (stderr.startsWith("Tesseract Open Source OCR Engine")) {
                                    await ControllerHelper.fileReadStream(result.output)
                                        .then((buffer) => {
                                            ControllerHelper.writeLog("Ocr.ts - ControllerHelper.fileReadStream - stdout", stderr);

                                            response.status(200).send(buffer.toString("base64"));

                                            //removeFile(input, output);
                                        })
                                        .catch((error: Error) => {
                                            ControllerHelper.writeLog(
                                                "Ocr.ts - ControllerHelper.fileReadStream - catch error",
                                                ControllerHelper.objectOutput(error)
                                            );

                                            removeFile(input, output);

                                            response.status(500).send({ Error: stderr });
                                        });
                                } else {
                                    ControllerHelper.writeLog("Ocr.ts - exec('tesseract... - stderr", stderr);

                                    removeFile(input, output);

                                    response.status(500).send({ Error: stderr });
                                }
                            })();
                        }
                    );
                })
                .catch(() => {
                    response.status(500).send({ Error: "Upload failed." });
                });
        })();
    });
};
