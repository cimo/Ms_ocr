import Express from "express";
import { exec } from "child_process";

// Source
import * as ControllerHelper from "../Controller/Helper";
import * as ControllerUpload from "../Controller/Upload";
import { IrequestBody } from "../Model/Helper";

const removeFile = (input: string | undefined, output: string | undefined) => {
    if (input) {
        ControllerHelper.fileRemove(input)
            .then(() => {
                ControllerHelper.writeLog("Converter.ts - ControllerHelper.fileRemove - input", input);
            })
            .catch((error: Error) => {
                ControllerHelper.writeLog("Converter.ts - ControllerHelper.fileRemove - error", ControllerHelper.objectOutput(error));
            });
    }

    if (output) {
        ControllerHelper.fileRemove(output)
            .then(() => {
                ControllerHelper.writeLog("Converter.ts - ControllerHelper.fileRemove - output", output);
            })
            .catch((error: Error) => {
                ControllerHelper.writeLog("Converter.ts - ControllerHelper.fileRemove - error", ControllerHelper.objectOutput(error));
            });
    }
};

export const execute = (app: Express.Express): void => {
    app.post("/msantivirus/check", (request: Express.Request, response: Express.Response) => {
        void (async () => {
            await ControllerUpload.execute(request)
                .then((result) => {
                    const input = result.input;

                    exec(`clamdscan "${input}"`, (error, stdout, stderr) => {
                        removeFile(input, undefined);

                        if (stdout !== "" && stderr === "") {
                            ControllerHelper.writeLog("Antivirus.ts - exec('clamdscan... - stdout", stdout);

                            response.status(200).send({ Response: stdout });
                        } else if (stdout === "" && stderr !== "") {
                            ControllerHelper.writeLog("Antivirus.ts - exec('clamdscan... - stderr", stderr);

                            response.status(500).send({ Error: stderr });
                        }
                    });
                })
                .catch(() => {
                    response.status(500).send({ Error: "Upload failed." });
                });
        })();
    });

    app.post("/msantivirus/update", (request: Express.Request, response: Express.Response) => {
        const requestBody = request.body as IrequestBody;

        const check = ControllerHelper.checkToken(requestBody.token_api);

        if (check) {
            exec("freshclam", (error, stdout, stderr) => {
                if (stdout !== "" && stderr === "") {
                    ControllerHelper.writeLog("Antivirus.ts - exec('freshclam --quiet... - stdout", stdout);

                    response.status(200).send({ Response: stdout });
                } else if (stdout === "" && stderr !== "") {
                    ControllerHelper.writeLog("Antivirus.ts - exec('freshclam --quiet... - stderr", stderr);

                    response.status(500).send({ Error: stderr });
                }
            });
        } else {
            ControllerHelper.writeLog("Antivirus.ts - app.post('/msantivirus/update' - error token_api", requestBody.token_api);

            response.status(500).send({ Error: "token_api not valid!" });
        }
    });
};
