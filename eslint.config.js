const global = require("globals");
const prettierPlugin = require("eslint-plugin-prettier");
const typescriptParser = require("@typescript-eslint/parser");
const typescriptPlugin = require("@typescript-eslint/eslint-plugin");
const customRule = require("./dist/eslint.customRule");

const configIgnore = {
    ignores: ["dist", "node_modules", "public", ".cache", ".config", ".local", ".ms_cronjob-volume", ".npm", ".pki"]
};

const configBase = {
    languageOptions: {
        globals: Object.assign({}, global.browser, global.node),
        sourceType: "module",
        parserOptions: {
            ecmaVersion: 2022
        }
    },
    plugins: {
        prettier: prettierPlugin
    },
    rules: {
        "no-console": "error",
        "no-debugger": "error",
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

const configTypescript = {
    files: ["eslint.customRule.ts", "global.d.ts", "src/**/*.{ts,tsx}", "file/input/**/*.{ts,tsx}"],
    languageOptions: {
        ...configBase.languageOptions,
        parser: typescriptParser,
        parserOptions: {
            ...configBase.languageOptions.parserOptions,
            tsconfigRootDir: "./",
            project: "./tsconfig.json"
        }
    },
    plugins: {
        ...configBase.plugins,
        "@typescript-eslint": typescriptPlugin,
        "custom-rule": customRule
    },
    rules: {
        ...configBase.rules,
        "@typescript-eslint/no-explicit-any": "error",
        "@typescript-eslint/no-unused-vars": [
            "error",
            {
                varsIgnorePattern: "^jsxFactory$"
            }
        ],
        "custom-rule/disallow-array-for-object-type": "error"
    }
};

const configJavascript = {
    files: ["esbuild.build.js", "eslint.config.js", "webpack.build.js", "src/**/*.{js,jsx}", "file/input/**/*.{js,jsx}"],
    languageOptions: {
        ...configBase.languageOptions
    },
    plugins: {
        ...configBase.plugins
    },
    rules: {
        ...configBase.rules
    }
};

module.exports = [configIgnore, configTypescript, configJavascript];
