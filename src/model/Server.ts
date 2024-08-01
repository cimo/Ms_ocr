import { Request } from "express";

export interface Icors {
    originList: string | string[] | undefined;
    methodList: string[];
    preflightContinue: boolean;
    optionsSuccessStatus: number;
}

export interface Ilimiter {
    windowMs: number;
    limit: number;
}

export interface Irequest extends Request {
    clientIp?: string | undefined;
}
