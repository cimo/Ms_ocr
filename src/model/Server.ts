export interface Icors {
    originList: string | string[] | undefined;
    methodList: string[];
    preflightContinue: boolean;
    optionsSuccessStatus: number;
}
