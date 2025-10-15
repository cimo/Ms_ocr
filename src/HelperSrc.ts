import Fs from "fs";
import { Response } from "express";
import { Ce } from "@cimo/environment/dist/src/Main";

// Source
import * as modelHelperSrc from "./model/HelperSrc";

const localeConfiguration: Record<string, { locale: string; currency: string }> = {
    // Europe
    it: { locale: "it-IT", currency: "EUR" },
    fr: { locale: "fr-FR", currency: "EUR" },
    de: { locale: "de-DE", currency: "EUR" },
    es: { locale: "es-ES", currency: "EUR" },
    pt: { locale: "pt-PT", currency: "EUR" },
    nl: { locale: "nl-NL", currency: "EUR" },
    ru: { locale: "ru-RU", currency: "RUB" },
    pl: { locale: "pl-PL", currency: "PLN" },
    sv: { locale: "sv-SE", currency: "SEK" },
    // Asia
    jp: { locale: "ja-JP", currency: "JPY" },
    cn: { locale: "zh-CN", currency: "CNY" },
    tw: { locale: "zh-TW", currency: "TWD" },
    kr: { locale: "ko-KR", currency: "KRW" },
    in: { locale: "hi-IN", currency: "INR" },
    th: { locale: "th-TH", currency: "THB" },
    // America
    us: { locale: "en-US", currency: "USD" },
    mx: { locale: "es-MX", currency: "MXN" },
    br: { locale: "pt-BR", currency: "BRL" },
    ca: { locale: "fr-CA", currency: "CAD" },
    // Africa
    ke: { locale: "sw-KE", currency: "KES" },
    za: { locale: "af-ZA", currency: "ZAR" },
    eg: { locale: "ar-EG", currency: "EGP" },
    // Oceania
    au: { locale: "en-AU", currency: "AUD" },
    nz: { locale: "mi-NZ", currency: "NZD" }
};

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

export const localeFromEnvName = (): string => {
    let result = ENV_NAME.split("_").pop();

    if (!result || result === "local") {
        result = "jp";
    }

    return result;
};

export const LOCALE = localeFromEnvName();

export const localeFormat = (value: number | Date, isTime = true): string | undefined => {
    if (typeof value === "number") {
        const formatOption: Intl.NumberFormatOptions = {
            style: "decimal",
            currency: localeConfiguration[LOCALE].currency
        };

        return new Intl.NumberFormat(localeConfiguration[LOCALE].locale, formatOption).format(value);
    } else if (value instanceof Date) {
        let formatOption: Intl.DateTimeFormatOptions = {
            year: "numeric",
            month: "numeric",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        };

        if (!isTime) {
            formatOption = {
                year: "numeric",
                month: "numeric",
                day: "numeric"
            };
        }

        return new Intl.DateTimeFormat(localeConfiguration[LOCALE].locale, formatOption).format(value);
    }

    return undefined;
};

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
    Fs.stat(path, (error, stats) => {
        if (error) {
            return callback(error);
        }

        if (stats.isDirectory()) {
            Fs.rm(path, { recursive: true, force: true }, (error) => {
                if (error) {
                    return callback(error);
                }

                callback(true);
            });
        } else {
            Fs.unlink(path, (error) => {
                if (error) {
                    return callback(error);
                }

                callback(true);
            });
        }
    });
};

export const fileCheckMimeType = (value: string): boolean => {
    if (MIME_TYPE && MIME_TYPE.includes(value)) {
        return true;
    }

    return false;
};

export const fileCheckSize = (byte: number): boolean => {
    const maxSizeByte = parseInt(FILE_SIZE_MB) * 1024 * 1024;

    if (byte > maxSizeByte) {
        return false;
    }

    return true;
};

export const responseBody = (stdoutValue: string, stderrValue: string | Error, response: Response, mode: number): void => {
    const responseBody: modelHelperSrc.IresponseBody = { response: { stdout: stdoutValue, stderr: stderrValue } };

    response.status(mode).send(responseBody);
};

export const keepProcess = (): void => {
    for (const event of ["uncaughtException", "unhandledRejection"]) {
        process.on(event, (error: Error) => {
            writeLog("HelperSrc.ts - keepProcess()", `Event: ${event} - Error: ${error.toString()}`);
        });
    }
};

export const isJson = (value: string): boolean => {
    try {
        JSON.parse(value);

        return true;
    } catch {
        return false;
    }
};

export const removeAnsiEscape = (text: string): string => {
    const regex = new RegExp(["\x1b", "[", "[0-9;]*", "[a-zA-Z]"].join(""), "g");

    return text.replace(regex, "");
};

export const generateUniqueId = (): string => {
    const timestamp = Date.now().toString(36);
    const randomPart = crypto.getRandomValues(new Uint32Array(1))[0].toString(36);

    return `${timestamp}-${randomPart}`;
};

export const findFileInDirectoryRecursive = async (path: string, extension: string): Promise<string[]> => {
    const resultList: string[] = [];

    const dataList = await Fs.promises.readdir(path);

    for (const data of dataList) {
        const pathData = `${path}${data}`;
        const statData = await Fs.promises.stat(pathData);

        if (statData.isDirectory()) {
            const dataSubList = await findFileInDirectoryRecursive(`${data}/`, extension);

            for (const dataSub of dataSubList) {
                resultList.push(dataSub);
            }
        } else if (statData.isFile() && data.endsWith(extension)) {
            resultList.push(data);
        }
    }

    return resultList;
};
