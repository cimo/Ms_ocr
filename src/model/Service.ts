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
    isCancel: boolean;
}

export interface IapiCheckBody {
    uniqueId: string;
}

export interface IapiCancelBody {
    uniqueId: string;
}
