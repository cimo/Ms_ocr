module.exports = {
    extends: [
        "eslint:recommended",
        "plugin:@typescript-eslint/recommended",
        "plugin:@typescript-eslint/recommended-requiring-type-checking",
        "plugin:prettier/recommended",
    ],
    parser: "@typescript-eslint/parser",
    plugins: ["@typescript-eslint", "prettier"],
    ignorePatterns: ["**/*.js", "dist", "node_modules", "public"],
    root: true,
    parserOptions: {
        project: true,
        tsconfigRootDir: __dirname,
    },
    rules: {
        "no-console": process.env.NODE_ENV === "production" ? "warn" : "off",
        "no-debugger": process.env.NODE_ENV === "production" ? "warn" : "off",
        "prettier/prettier": [
            "error",
            {
                proseWrap: "always",
                printWidth: 300,
                arrowParens: "always",
                bracketSpacing: true,
                embeddedLanguageFormatting: "auto",
                htmlWhitespaceSensitivity: "css",
                quoteProps: "as-needed",
                semicolons: true,
                singleQuote: false,
                trailingComma: "none",
                endOfLine: "lf",
            },
        ],
    },
};
