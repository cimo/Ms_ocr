interface IonnxItem {
    id: number;
    polygon: number[][];
    text: string;
    isMatch: boolean;
}

interface IonnxLayout {
    label: string;
    score: number;
    coordinate: number[];
}

export interface IapiDownloadBody {
    uniqueId: string;
    pathFile: string;
}

export interface IapiOnnxResponse {
    layoutList: IonnxLayout[];
    itemList: IonnxItem[];
}
