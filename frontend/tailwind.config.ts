import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: "#0E1117",
        cardBg: "#161B22",
        borderDark: "#21262D",
        accentGreen: "#10B981",
      },
    },
  },
  plugins: [],
};
export default config;