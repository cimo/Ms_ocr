interface Ilayout {
    label: string;
    score: number;
    centerPoint: {
        x: number;
        y: number;
    };
}

interface Iitem {
    id: number;
    text: string;
    centerPoint: {
        x: number;
        y: number;
    };
    isMatch: boolean;
}

export interface IapiDownloadBody {
    uniqueId: string;
    pathFile: string;
}

export interface IapiScannerResponse {
    layoutList: Ilayout[];
    itemList: Iitem[];
}
