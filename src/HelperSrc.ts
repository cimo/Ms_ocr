import Fs from "fs";
import { Response } from "express";
import { Ce } from "@cimo/environment";

// Source
import * as ModelHelperSrc from "./model/HelperSrc";

export const ENV_NAME = Ce.checkVariable("ENV_NAME") || (process.env.ENV_NAME as string);

Ce.loadFile(`./env/${ENV_NAME}.env`);

export const DOMAIN = Ce.checkVariable("DOMAIN") || (process.env.DOMAIN as string);
export const TIME_ZONE = Ce.checkVariable("TIME_ZONE") || (process.env.TIME_ZONE as string);
export const LANG = Ce.checkVariable("LANG") || (process.env.LANG as string);
export const SERVER_PORT = Ce.checkVariable("SERVER_PORT") || (process.env.SERVER_PORT as string);
export const PATH_ROOT = Ce.checkVariable("PATH_ROOT");
export const NAME = Ce.checkVariable("MS_O_NAME") || (process.env.MS_O_NAME as string);
export const LABEL = Ce.checkVariable("MS_O_LABEL") || (process.env.MS_O_LABEL as string);
export const DEBUG = Ce.checkVariable("MS_O_DEBUG") || (process.env.MS_O_DEBUG as string);
export const NODE_ENV = Ce.checkVariable("MS_O_NODE_ENV") || (process.env.MS_O_NODE_ENV as string);
export const URL_ROOT = Ce.checkVariable("MS_O_URL_ROOT") || (process.env.MS_O_URL_ROOT as string);
export const URL_CORS_ORIGIN = Ce.checkVariable("MS_O_URL_CORS_ORIGIN") || (process.env.MS_O_URL_CORS_ORIGIN as string);
export const PATH_CERTIFICATE_KEY = Ce.checkVariable("MS_O_PATH_CERTIFICATE_KEY");
export const PATH_CERTIFICATE_CRT = Ce.checkVariable("MS_O_PATH_CERTIFICATE_CRT");
export const PATH_PUBLIC = Ce.checkVariable("MS_O_PATH_PUBLIC");
export const PATH_LOG = Ce.checkVariable("MS_O_PATH_LOG");
export const PATH_FILE_INPUT = Ce.checkVariable("MS_O_PATH_FILE_INPUT");
export const PATH_FILE_OUTPUT = Ce.checkVariable("MS_O_PATH_FILE_OUTPUT");
export const PATH_FILE_SCRIPT = Ce.checkVariable("MS_O_PATH_FILE_SCRIPT");
export const MIME_TYPE = Ce.checkVariable("MS_O_MIME_TYPE") || (process.env.MS_O_MIME_TYPE as string);
export const FILE_SIZE_MB = Ce.checkVariable("MS_O_FILE_SIZE_MB") || (process.env.MS_O_FILE_SIZE_MB as string);

export const writeLog = (tag: string, value: string | Record<string, unknown> | Error): void => {
    if (DEBUG === "true") {
        if (typeof process !== "undefined") {
            Fs.appendFile(`${PATH_ROOT}${PATH_LOG}debug.log`, `${tag}: ${value.toString()}\n`, () => {
                // eslint-disable-next-line no-console
                console.log(`WriteLog => ${tag}: `, value);
            });
        } else {
            // eslint-disable-next-line no-console
            console.log(`WriteLog => ${tag}: `, value);
        }
    }
};

export const serverTime = (): string => {
    const currentDate = new Date();

    const month = currentDate.getMonth() + 1;
    const monthOut = month < 10 ? `0${month}` : `${month}`;

    const day = currentDate.getDate();
    const dayOut = day < 10 ? `0${day}` : `${day}`;

    const date = `${currentDate.getFullYear()}/${monthOut}/${dayOut}`;

    const minute = currentDate.getMinutes();
    const minuteOut = minute < 10 ? `0${minute}` : `${minute}`;

    const second = currentDate.getSeconds();
    const secondOut = second < 10 ? `0${second}` : `${second}`;

    const time = `${currentDate.getHours()}:${minuteOut}:${secondOut}`;

    return `${date} ${time}`;
};

export const fileWriteStream = (filePath: string, buffer: Buffer, callback: (result: NodeJS.ErrnoException | boolean) => void): void => {
    const writeStream = Fs.createWriteStream(filePath);

    writeStream.on("open", () => {
        writeStream.write(buffer);
        writeStream.end();
    });

    writeStream.on("finish", () => {
        callback(true);
    });

    writeStream.on("error", (error) => {
        callback(error);
    });
};

export const fileReadStream = (filePath: string, callback: (result: NodeJS.ErrnoException | Buffer) => void): void => {
    const chunkList: Buffer[] = [];

    const readStream = Fs.createReadStream(filePath);

    readStream.on("data", (chunk: Buffer) => {
        chunkList.push(chunk);
    });

    readStream.on("end", () => {
        callback(Buffer.concat(chunkList));
    });

    readStream.on("error", (error) => {
        callback(error);
    });
};

export const fileRemove = (path: string, callback: (result: NodeJS.ErrnoException | boolean) => void): void => {
    Fs.unlink(path, (error) => {
        if (!error) {
            callback(true);
        } else {
            callback(error);
        }
    });
};

export const fileCheckMimeType = (value: string): boolean => {
    if (MIME_TYPE && MIME_TYPE.includes(value)) {
        return true;
    }

    return false;
};

export const fileCheckSize = (value: string): boolean => {
    const fileSizeMb = parseInt(FILE_SIZE_MB ? FILE_SIZE_MB : "0") * 1024 * 1024;

    if (fileSizeMb >= parseInt(value)) {
        return true;
    }

    return false;
};

export const responseBody = (stdoutValue: string, stderrValue: string | Error, response: Response, mode: number): void => {
    const responseBody: ModelHelperSrc.IresponseBody = { response: { stdout: stdoutValue, stderr: stderrValue } };

    response.status(mode).send(responseBody);
};

export const keepProcess = (): void => {
    for (const event of ["uncaughtException", "unhandledRejection"]) {
        process.on(event, (error: Error) => {
            writeLog("HelperSrc.ts => keepProcess()", `Event: ${event} - Error: ${error.toString()}`);
        });
    }
};

export const isJson = (value: string): boolean => {
    return /^[\],:{}\s]*$/.test(
        value
            .replace(/\\["\\/bfnrtu]/g, "@")
            .replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?/g, "]")
            .replace(/(?:^|:|,)(?:\s*\[)+/g, "")
    );
};

export const removeAnsiEscape = (text: string) => {
    const regex = new RegExp(["\x1b", "[", "[0-9;]*", "[a-zA-Z]"].join(""), "g");

    return text.replace(regex, "");
};

export const locationFromEnvName = () => {
    let result = ENV_NAME.split("_").pop();

    if (result === "local") {
        result = "jp";
    }

    return result;
};
