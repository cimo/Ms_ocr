const globals = require("globals");
const tsParser = require("@typescript-eslint/parser");
const tsPlugin = require("@typescript-eslint/eslint-plugin");
const prettierPlugin = require('eslint-plugin-prettier');

const configCommon = {
    ignores: ["./**/*", "!./file/**/*", "!./src/**/*", "!./webpack.build.js"]
}

const configFile = {
    files: ["**/*.{ts, js}"],
    languageOptions: {
        globals: Object.assign({}, globals.browser, globals.node),
        parser: tsParser,
        sourceType: "module",
        parserOptions: {
            ecmaVersion: 2015,
            project: "./tsconfig.json",
            tsconfigRootDir: "./"
        }
    },
    plugins: {
        '@typescript-eslint': tsPlugin,
        'prettier': prettierPlugin
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
}

module.exports = [
	configCommon,
	configFile
]
