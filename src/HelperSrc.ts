import { Response } from "express";
import Fs from "fs";

// Source
import * as ModelHelperSrc from "./model/HelperSrc";

export const writeLog = (tag: string, value: string | Record<string, unknown> | Error): void => {
    if (process.env.MS_O_DEBUG === "true") {
        if (typeof process !== "undefined" && process.env.MS_O_PATH_LOG) {
            Fs.appendFile(`${process.env.MS_O_PATH_LOG}debug.log`, `${tag}: ${value.toString()}\n`, () => {
                // eslint-disable-next-line no-console
                console.log(`WriteLog => ${tag}: `, value);
            });
        } else {
            // eslint-disable-next-line no-console
            console.log(`WriteLog => ${tag}: `, value);
        }
    }
};

const checkEnv = (key: string, value: string | undefined): string => {
    if (typeof process !== "undefined" && value === undefined) {
        writeLog("HelperSrc.ts => checkEnv()", `${key} is not defined!`);
    }

    return value ? value : "";
};

export const ENV_NAME = checkEnv("ENV_NAME", process.env.ENV_NAME);
export const DOMAIN = checkEnv("DOMAIN", process.env.DOMAIN);
export const TIMEZONE = checkEnv("TIMEZONE", process.env.TIMEZONE);
export const SERVER_PORT = checkEnv("SERVER_PORT", process.env.SERVER_PORT);
export const SERVER_LOCATION = checkEnv("SERVER_LOCATION", process.env.SERVER_LOCATION);
export const PATH_ROOT = checkEnv("PATH_ROOT", process.env.PATH_ROOT);
export const NAME = checkEnv("MS_O_NAME", process.env.MS_O_NAME);
export const LABEL = checkEnv("MS_O_LABEL", process.env.MS_O_LABEL);
export const DEBUG = checkEnv("MS_O_DEBUG", process.env.MS_O_DEBUG);
export const NODE_ENV = checkEnv("MS_O_NODE_ENV", process.env.MS_O_NODE_ENV);
export const URL_ROOT = checkEnv("MS_O_URL_ROOT", process.env.MS_O_URL_ROOT);
export const URL_CORS_ORIGIN = checkEnv("MS_O_URL_CORS_ORIGIN", process.env.MS_O_URL_CORS_ORIGIN);
export const PATH_CERTIFICATE_KEY = checkEnv("MS_O_PATH_CERTIFICATE_KEY", process.env.MS_O_PATH_CERTIFICATE_KEY);
export const PATH_CERTIFICATE_CRT = checkEnv("MS_O_PATH_CERTIFICATE_CRT", process.env.MS_O_PATH_CERTIFICATE_CRT);
export const PATH_PUBLIC = checkEnv("MS_O_PATH_PUBLIC", process.env.MS_O_PATH_PUBLIC);
export const PATH_LOG = checkEnv("MS_O_PATH_LOG", process.env.MS_O_PATH_LOG);
export const PATH_FILE_INPUT = checkEnv("MS_O_PATH_FILE_INPUT", process.env.MS_O_PATH_FILE_INPUT);
export const PATH_FILE_OUTPUT = checkEnv("MS_O_PATH_FILE_OUTPUT", process.env.MS_O_PATH_FILE_OUTPUT);
export const PATH_FILE_SCRIPT = checkEnv("MS_O_PATH_FILE_SCRIPT", process.env.MS_O_PATH_FILE_SCRIPT);
export const MIME_TYPE = checkEnv("MS_O_MIME_TYPE", process.env.MS_O_MIME_TYPE);
export const FILE_SIZE_MB = checkEnv("MS_O_FILE_SIZE_MB", process.env.MS_O_FILE_SIZE_MB);

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
