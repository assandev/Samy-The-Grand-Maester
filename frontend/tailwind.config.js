/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        panel: "0 22px 60px rgba(15, 23, 42, 0.08)",
      },
      colors: {
        cloud: "#eef4fb",
        mist: "#f7faff",
        line: "#dbe6f2",
        text: "#1e293b",
        muted: "#64748b",
        brand: "#2563eb",
        brandSoft: "#dbeafe",
        success: "#0f766e",
        successSoft: "#ecfdf5",
        warning: "#ea9b2d",
        warningSoft: "#fff7ed",
      },
      fontFamily: {
        sans: ["Manrope", "\"Segoe UI Variable\"", "\"Segoe UI\"", "sans-serif"],
      },
    },
  },
  plugins: [],
};
