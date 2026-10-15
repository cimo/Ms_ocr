export interface IengineLine {
    uniqueId: string;
    error?: string;
    response?: {
        state: string;
        message: string;
    };
}

export interface Ijob {
    state: string;
    message: string;
    isError: boolean;
}

export interface IapiCheckBody {
    uniqueId: string;
}
