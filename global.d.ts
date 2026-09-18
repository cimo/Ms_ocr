declare module "http-compression" {
    import { RequestHandler } from "express";

    const compression: (option?: {
        threshold?: number;
        level?: { brotli?: number; gzip?: number };
        brotli?: boolean;
        gzip?: boolean;
        mimes?: RegExp;
    }) => RequestHandler;

    export default compression;
}

declare namespace JSX {
    interface IntrinsicElements {
        [elemName: string]: unknown;
    }
}
