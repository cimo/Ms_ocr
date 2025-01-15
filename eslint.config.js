const globals = require("globals");
const tsParser = require("@typescript-eslint/parser");
const tsPlugin = require("@typescript-eslint/eslint-plugin");
const prettierPlugin = require("eslint-plugin-prettier");

const configCommon = {
    ignores: ["./**/*", "!./file/**/*", "!./src/**/*", "!./webpack.build.js"]
};

const configBase = {
    languageOptions: {
        globals: Object.assign({}, globals.browser, globals.node),
        parser: tsParser,
        sourceType: "module",
        parserOptions: {
            ecmaVersion: 2022
        }
    },
    plugins: {
        "@typescript-eslint": tsPlugin,
        prettier: prettierPlugin
    },
    rules: {
        "no-console": "error",
        "no-debugger": "error",
        "@typescript-eslint/no-unused-vars": "error",
        "prettier/prettier": [
            "error",
            {
                proseWrap: "always",
                printWidth: 150,
                arrowParens: "always",
                bracketSpacing: true,
                embeddedLanguageFormatting: "auto",
                htmlWhitespaceSensitivity: "css",
                quoteProps: "as-needed",
                semicolons: true,
                singleQuote: false,
                trailingComma: "none",
                endOfLine: "lf"
            }
        ]
    }
};

const configFile = {
    ...configBase,
    files: ["**/*.{ts,js}"],
    ignores: [".*/**/*", "public/**/*", "dist/**/*", "eslint.config.js", "webpack.build.js"],
    languageOptions: {
        ...configBase.languageOptions,
        parserOptions: {
            ...configBase.languageOptions.parserOptions,
            project: "./tsconfig.json",
            tsconfigRootDir: "./"
        }
    }
};

const configFileIgnored = {
    ...configBase,
    files: ["eslint.config.js", "webpack.build.js"]
};

module.exports = [configCommon, configFile, configFileIgnored];
