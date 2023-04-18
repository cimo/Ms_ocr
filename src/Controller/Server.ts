import Express from "express";
import * as Fs from "fs";
import * as Https from "https";
import CookieParser from "cookie-parser";
import Cors from "cors";

// Source
import * as ControllerHelper from "../Controller/Helper";
import * as ControllerOcr from "../Controller/Ocr";
import * as ModelServer from "../Model/Server";

const corsOption: ModelServer.Cors = {
    originList: ControllerHelper.CORS_ORIGIN_URL,
    methodList: ["GET", "HEAD", "PUT", "PATCH", "POST", "DELETE"],
    preflightContinue: false,
    optionsSuccessStatus: 200
};

const app = Express();
app.use(Express.json());
app.use(Express.urlencoded({ extended: true }));
app.use(Express.static(ControllerHelper.PATH_STATIC));
app.use(CookieParser());
app.use(
    Cors({
        origin: corsOption.originList,
        methods: corsOption.methodList,
        optionsSuccessStatus: corsOption.optionsSuccessStatus
    })
);

const server = Https.createServer(
    {
        key: Fs.readFileSync(ControllerHelper.PATH_CERTIFICATE_FILE_KEY),
        cert: Fs.readFileSync(ControllerHelper.PATH_CERTIFICATE_FILE_CRT)
    },
    app
);

server.listen(ControllerHelper.SERVER_PORT, () => {
    const serverTime = ControllerHelper.serverTime();

    ControllerHelper.writeLog("Server.ts - server.listen", `Port ${ControllerHelper.SERVER_PORT || ""} - Time: ${serverTime}`);

    app.get("/", (request, response) => {
        response.status(200).send("ms_ocr");
    });

    ControllerOcr.execute(app);
});
