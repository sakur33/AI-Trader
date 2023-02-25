module.exports = {
  env: {
    browser: true,
    commonjs: true,
    es2021: true,
    node: true,
  },
  extends: [
    "airbnb-base",
    "eslint:recommended",
    "plugin:vue/vue3-essential",
    "plugin:@typescript-eslint/recommended",
    "@vue/typescript",
    "plugin:vue/essential",
    "eslint:recommended",
    "plugin:import/recommended",
    "plugin:import/typescript",
    "prettier",
  ],
  overrides: [],
  parser: "vue-eslint-parser",
  parserOptions: {
    ecmaVersion: "latest",
  },
  plugins: ["vue", "@typescript-eslint", "eslint-plugin-import", "prettier"],
  ignorePatterns: ["dist/**/*.js", "node_modules/**/*.js"],
  rules: {
    semi: ["error", "always"],
    quotes: 0,
    "class-methods-use-this": 0,
    "no-nested-ternary": 0,
    "no-confusing-arrow": 0,
    "implicit-arrow-linebreak": 0,
    "@typescript-eslint/no-var-requires": 0,
    "@typescript-eslint/no-explicit-any": 0,
    "vue/no-deprecated-v-on-native-modifier": 0,
    "arrow-body-style": 0,
    "prefer-template": 0,
    "comma-dangle": 0,
    "linebreak-style": ["error", "unix"],
    "import/extensions": [
      "error",
      "ignorePackages",
      {
        js: "never",
        jsx: "never",
        ts: "never",
        tsx: "never",
      },
    ],
    "prettier/prettier": [
      "error",
      {
        endOfLine: "auto",
      },
    ],
  },
  settings: {
    "import/resolver": {
      alias: {
        map: [
          ["@", "./src"],
          ["@vue", "../node_modules"],
        ],
        extensions: [".js", ".jsx", ".ts", ".tsx"],
      },
      node: {
        extensions: [".js", ".jsx", ".json", ".ts", ".tsx"],
      },
    },
  },
}
