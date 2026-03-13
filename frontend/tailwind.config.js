/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef2ff",
          100: "#e0e7ff",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
        },
        surface: {
          800: "#1e1e2e",
          900: "#13131f",
          950: "#0d0d17",
        },
      },
      animation: {
        "bounce-dot": "bounce 0.8s infinite",
      },
    },
  },
  plugins: [],
};
