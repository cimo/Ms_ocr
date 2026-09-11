export interface IapiDownloadBody {
    uniqueId: string;
    pathFile: string;
}

export interface IapiScannerResponse {
    response: {
        state: string;
        message: string;
    };
}
