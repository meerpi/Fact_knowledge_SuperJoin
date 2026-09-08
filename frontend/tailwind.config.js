/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          '"SF Mono"',
          'Menlo',
          'Monaco',
          'Consolas',
          '"Liberation Mono"',
          '"Courier New"',
          'monospace',
        ],
      },
      colors: {
        verdict: {
          corroborate: {
            DEFAULT: '#0f766e',
            light: '#f0fdfa',
            border: '#99f6e4',
            dark: '#115e59',
            text: '#134e4a',
          },
          contradict: {
            DEFAULT: '#b91c1c',
            light: '#fef2f2',
            border: '#fecaca',
            dark: '#991b1b',
            text: '#7f1d1d',
          },
          reconcile: {
            DEFAULT: '#1d4ed8',
            light: '#eff6ff',
            border: '#bfdbfe',
            dark: '#1e40af',
            text: '#1e3a8a',
          },
        },
      },
    },
  },
  plugins: [],
};
