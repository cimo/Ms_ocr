import Express, { Request, Response, NextFunction } from "express";
import { rateLimit } from "express-rate-limit";
import CookieParser from "cookie-parser";
import Cors from "cors";
import * as Http from "http";
import * as Https from "https";
import Fs from "fs";
import { Ca } from "@cimo/authentication/dist/src/Main";

// Source
import * as helperSrc from "../HelperSrc";
import * as modelServer from "../model/Server";
import ControllerOcr from "./Ocr";

export default class Server {
    // Variable
    private corsOption: modelServer.Icors;
    private limiterOption: modelServer.Ilimiter;
    private app: Express.Express;

    // Method
    constructor() {
        this.corsOption = {
            originList: [helperSrc.URL_CORS_ORIGIN],
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
        this.app.use("/asset", Express.static(`${helperSrc.PATH_ROOT}${helperSrc.PATH_PUBLIC}asset/`));
        this.app.use(CookieParser());
        this.app.use(
            Cors({
                origin: this.corsOption.originList,
                methods: this.corsOption.methodList,
                optionsSuccessStatus: this.corsOption.optionsSuccessStatus
            })
        );
        this.app.use(
            rateLimit({
                windowMs: this.limiterOption.windowMs,
                limit: this.limiterOption.limit
            })
        );
        this.app.use((_request: modelServer.Irequest, response: Response, next: NextFunction) => {
            response.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, private");
            response.setHeader("Pragma", "no-cache");
            response.setHeader("Expires", "0");

            next();
        });
        this.app.use((request: modelServer.Irequest, _, next: NextFunction) => {
            const headerForwarded = request.headers["x-forwarded-for"] ? request.headers["x-forwarded-for"][0] : "";
            const removeAddress = request.socket.remoteAddress ? request.socket.remoteAddress : "";

            request.clientIp = headerForwarded || removeAddress;

            next();
        });
    };

    createServer = (): void => {
        let creation: Http.Server | Https.Server;

        if (helperSrc.locationFromEnvName() === "jp") {
            creation = Https.createServer(
                {
                    key: Fs.readFileSync(`${helperSrc.PATH_ROOT}${helperSrc.PATH_CERTIFICATE_KEY}`),
                    cert: Fs.readFileSync(`${helperSrc.PATH_ROOT}${helperSrc.PATH_CERTIFICATE_CRT}`)
                },
                this.app
            );
        } else {
            creation = Http.createServer(this.app);
        }

        const server = creation;

        server.listen(helperSrc.SERVER_PORT, () => {
            const controllerOcr = new ControllerOcr(this.app);
            controllerOcr.api();

            const serverTime = helperSrc.serverTime();

            helperSrc.writeLog("Server.ts - createServer() - listen()", `Port: ${helperSrc.SERVER_PORT} - Time: ${serverTime}`);

            this.app.get("/login", (_request: Request, response: Response) => {
                Ca.writeCookie(`${helperSrc.LABEL}_authentication`, response);

                helperSrc.responseBody("Logged.", "", response, 200);
            });

            this.app.get("/logout", Ca.authenticationMiddleware, (request: Request, response: Response) => {
                Ca.removeCookie(`${helperSrc.LABEL}_authentication`, request, response);

                response.redirect("/info");
            });

            this.app.get("/info", (request: modelServer.Irequest, response: Response) => {
                helperSrc.responseBody(`Client ip: ${request.clientIp || ""}`, "", response, 200);
            });

            this.app.get("/file/*", Ca.authenticationMiddleware, (request: Request, response: Response) => {
                const filePath = `${helperSrc.PATH_ROOT}${helperSrc.PATH_PUBLIC}${request.path}`;

                if (Fs.existsSync(filePath)) {
                    response.sendFile(filePath);
                } else {
                    response.status(404).send("File not found!");
                }
            });

            this.app.get("*", Ca.authenticationMiddleware, (_request: Request, response: Response) => {
                response.sendFile(`${helperSrc.PATH_ROOT}${helperSrc.PATH_PUBLIC}index.html`);
            });
        });
    };
}

const controllerServer = new Server();
controllerServer.createSetting();
controllerServer.createServer();

helperSrc.keepProcess();
