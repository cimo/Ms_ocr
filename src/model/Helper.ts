export interface IcircularReplacer {
    (key: string, value: string): string | null;
}

export interface IrequestBody {
    token_api: string;
}

export interface IresponseBody {
    response: {
        stdout: string;
        stderr: string | Error;
    };
}
