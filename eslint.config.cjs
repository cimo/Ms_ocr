const global = require("globals");
const prettierPlugin = require("eslint-plugin-prettier");
const customRule = require("./dist/eslint.customRule");
const typescriptParser = require("@typescript-eslint/parser");
const typescriptPlugin = require("@typescript-eslint/eslint-plugin");

const configIgnore = {
    ignores: ["dist", "node_modules", "public", ".cache", ".config", ".dbus", ".local", ".npm", ".nv", ".pki"]
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
        "custom-rule": customRule,
        "@typescript-eslint": typescriptPlugin
    },
    rules: {
        ...configBase.rules,
        "custom-rule/no-array-assignment-for-object-type": "error",
        "@typescript-eslint/no-explicit-any": "error",
        "@typescript-eslint/no-unused-vars": [
            "error",
            {
                varsIgnorePattern: "^jsxFactory$"
            }
        ]
    }
};

const configJavascript = {
    files: ["esbuild.build.js", "eslint.config.cjs", "webpack.build.js", "src/**/*.{js,jsx}", "file/input/**/*.{js,jsx}"],
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
