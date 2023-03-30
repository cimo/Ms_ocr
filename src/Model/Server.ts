export interface Cors {
    originList: string | string[] | undefined;
    methodList: string[];
    preflightContinue: boolean;
    optionsSuccessStatus: number;
}
