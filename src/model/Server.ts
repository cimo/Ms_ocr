import { Request } from "express";

export interface Icors {
    originList: string[];
    methodList: string[];
    preflightContinue: boolean;
    optionsSuccessStatus: number;
}

export interface Irequest extends Request {
    clientIp?: string | undefined;
}
