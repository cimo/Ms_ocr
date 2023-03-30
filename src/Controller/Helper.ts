import Fs from "fs";

// Source
import * as ModelHelper from "../Model/Helper";

export const checkEnv = (key: string, value: string | undefined): string => {
    if (value === undefined) {
        console.log("Helper.ts - checkEnv - error:", `${key} is not defined!`);
    }

    return value as string;
};

export const ENV_NAME = checkEnv("ENV_NAME", process.env.ENV_NAME);
export const DOMAIN = checkEnv("DOMAIN", process.env.DOMAIN);
export const DEBUG = checkEnv("MS_A_DEBUG", process.env.MS_A_DEBUG);
export const CORS_ORIGIN_URL = checkEnv("MS_A_CORS_ORIGIN_URL", process.env.MS_A_CORS_ORIGIN_URL);
export const SERVER_PORT = checkEnv("MS_A_SERVER_PORT", process.env.MS_A_SERVER_PORT);
export const MIME_TYPE = checkEnv("MS_A_MIME_TYPE", process.env.MS_A_MIME_TYPE);
export const FILE_SIZE_MB = checkEnv("MS_A_FILE_SIZE_MB", process.env.MS_A_FILE_SIZE_MB);
export const TOKEN = checkEnv("MS_A_TOKEN", process.env.MS_A_TOKEN);
export const PATH_STATIC = checkEnv("MS_A_PATH_STATIC", process.env.MS_A_PATH_STATIC);
export const PATH_LOG = checkEnv("MS_A_PATH_LOG", process.env.MS_A_PATH_LOG);
export const PATH_FILE_INPUT = checkEnv("MS_A_PATH_FILE_INPUT", process.env.MS_A_PATH_FILE_INPUT);
export const PATH_CERTIFICATE_FILE_KEY = checkEnv("MS_A_PATH_CERTIFICATE_FILE_KEY", process.env.MS_A_PATH_CERTIFICATE_FILE_KEY);
export const PATH_CERTIFICATE_FILE_CRT = checkEnv("MS_A_PATH_CERTIFICATE_FILE_CRT", process.env.MS_A_PATH_CERTIFICATE_FILE_CRT);

const circularReplacer = (): ModelHelper.circularReplacer => {
    const seen = new WeakSet();

    return (key: string, value: string): string | null => {
        if (value !== null && typeof value === "object") {
            if (seen.has(value)) {
                return null;
            }

            seen.add(value);
        }

        return value;
    };
};

export const objectOutput = (obj: unknown): string => {
    return JSON.stringify(obj, circularReplacer(), 2);
};

export const writeLog = (tag: string, value: string): void => {
    if (DEBUG === "true" && PATH_LOG) {
        Fs.appendFile(`${PATH_LOG}debug.log`, `${tag}: ${value}\n`, () => {
            console.log(`WriteLog => ${tag}: `, value);
        });
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

    const result = `${date} ${time}`;

    return result;
};

export const fileWriteStream = (filePath: string, buffer: Buffer): Promise<void> => {
    return new Promise((resolve, reject) => {
        const writeStream = Fs.createWriteStream(filePath);

        writeStream.on("open", () => {
            writeStream.write(buffer);
            writeStream.end();
        });

        writeStream.on("finish", () => {
            resolve();
        });

        writeStream.on("error", (error: Error) => {
            reject(error);
        });
    });
};

export const fileReadStream = (filePath: string): Promise<Buffer> => {
    return new Promise((resolve, reject) => {
        const chunkList: Buffer[] = [];

        const readStream = Fs.createReadStream(filePath);

        readStream.on("data", (chunk: Buffer) => {
            chunkList.push(chunk);
        });

        readStream.on("end", () => {
            const result = Buffer.concat(chunkList);

            resolve(result);
        });

        readStream.on("error", (error: Error) => {
            reject(error);
        });
    });
};

export const fileRemove = (path: string): Promise<NodeJS.ErrnoException | boolean> => {
    return new Promise((resolve, reject) => {
        Fs.unlink(path, (error: NodeJS.ErrnoException | null) => {
            if (error) {
                reject(error);
            } else {
                resolve(true);
            }
        });
    });
};

export const checkToken = (value: string): boolean => {
    if (TOKEN && TOKEN === value) {
        return true;
    }

    return false;
};

export const checkMymeType = (value: string): boolean => {
    if (MIME_TYPE && MIME_TYPE.includes(value)) {
        return true;
    }

    return false;
};

export const checkFileSize = (value: string): boolean => {
    const fileSizeMb = parseInt(FILE_SIZE_MB ? FILE_SIZE_MB : "0") * 1024 * 1024;

    if (fileSizeMb >= parseInt(value)) {
        return true;
    }

    return false;
};
