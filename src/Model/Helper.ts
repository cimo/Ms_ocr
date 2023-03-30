export interface circularReplacer {
    (key: string, value: string): string | null;
}

export interface IrequestBody {
    token_api: string;
}
