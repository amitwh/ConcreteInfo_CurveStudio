import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules", "src/api/schema.d.ts"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // The service worker runs outside the DOM: it needs the worker globals, not window.
    files: ["public/**/*.js"],
    languageOptions: {
      globals: {
        self: "readonly",
        caches: "readonly",
        fetch: "readonly",
        clients: "readonly",
        Promise: "readonly",
        URL: "readonly",
      },
    },
  },
  {
    files: ["**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
);
