import Express, { Request, Response, NextFunction } from "express";
import { rateLimit } from "express-rate-limit";
import CookieParser from "cookie-parser";
import Cors from "cors";
import * as Http from "http";
import * as Https from "https";
import Fs from "fs";
import { Ca } from "@cimo/authentication/dist/src/Main";

// Source
import * as HelperSrc from "../HelperSrc";
import * as ModelServer from "../model/Server";
import ControllerOcr from "./Ocr";

export default class ControllerServer {
    // Variable
    private corsOption: ModelServer.Icors;
    private limiterOption: ModelServer.Ilimiter;
    private app: Express.Express;

    // Method
    constructor() {
        this.corsOption = {
            originList: [HelperSrc.URL_CORS_ORIGIN],
            methodList: ["GET", "HEAD", "PUT", "PATCH", "POST", "DELETE"],
            preflightContinue: false,
            optionsSuccessStatus: 200
        };

        this.limiterOption = {
            windowMs: 15 * 60 * 1000,
            limit: 100
        };

        this.app = Express();
    }

    createSetting = (): void => {
        this.app.use(Express.json());
        this.app.use(Express.urlencoded({ extended: true }));
        this.app.use(Express.static(`${HelperSrc.PATH_ROOT}${HelperSrc.PATH_PUBLIC}`));
        this.app.use(CookieParser());
        this.app.use(
            Cors({
                origin: this.corsOption.originList,
                methods: this.corsOption.methodList,
                optionsSuccessStatus: this.corsOption.optionsSuccessStatus
            })
        );
        this.app.use((request: ModelServer.Irequest, _, next: NextFunction) => {
            const headerForwarded = request.headers["x-forwarded-for"] ? request.headers["x-forwarded-for"][0] : "";
            const removeAddress = request.socket.remoteAddress ? request.socket.remoteAddress : "";

            request.clientIp = headerForwarded || removeAddress;

            next();
        });
        this.app.use(
            rateLimit({
                windowMs: this.limiterOption.windowMs,
                limit: this.limiterOption.limit
            })
        );
    };

    createServer = (): void => {
        let creation: Http.Server | Https.Server;

        if (HelperSrc.locationFromEnvName() === "jp") {
            creation = Https.createServer(
                {
                    key: Fs.readFileSync(`${HelperSrc.PATH_ROOT}${HelperSrc.PATH_CERTIFICATE_KEY}`),
                    cert: Fs.readFileSync(`${HelperSrc.PATH_ROOT}${HelperSrc.PATH_CERTIFICATE_CRT}`)
                },
                this.app
            );
        } else {
            creation = Http.createServer(this.app);
        }

        const server = creation;

        server.listen(HelperSrc.SERVER_PORT, () => {
            const controllerOcr = new ControllerOcr(this.app);
            controllerOcr.api();

            const serverTime = HelperSrc.serverTime();

            HelperSrc.writeLog("Server.ts => createServer() => listen()", `Port: ${HelperSrc.SERVER_PORT} - Time: ${serverTime}`);

            this.app.get("/info", (request: ModelServer.Irequest, response: Response) => {
                HelperSrc.responseBody(`Client ip: ${request.clientIp || ""}`, "", response, 200);
            });

            this.app.get("/login", (_request: Request, response: Response) => {
                Ca.writeCookie(`${HelperSrc.LABEL}_authentication`, response);

                HelperSrc.responseBody("Logged.", "", response, 200);
            });

            this.app.get("/logout", Ca.authenticationMiddleware, (request: Request, response: Response) => {
                Ca.removeCookie(`${HelperSrc.LABEL}_authentication`, request, response);

                response.redirect("info");
            });
        });
    };
}

const controllerServer = new ControllerServer();
controllerServer.createSetting();
controllerServer.createServer();

HelperSrc.keepProcess();
