/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#1a1a1a',
        secondary: '#2a2a2a',
        accent: '#ffd700',
      },
      animation: {
        'slow-pulse': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      typography: {
        DEFAULT: {
          css: {
            color: 'white',
            a: {
              color: '#ffd700',
              '&:hover': {
                color: '#ffed4a',
              },
            },
            strong: {
              color: '#ffd700',
            },
            code: {
              color: 'white',
            },
          },
        },
      },
      keyframes: {
        bounce: {
          '0%, 80%, 100%': { transform: 'translateY(0)' },
          '40%': { transform: 'translateY(-4px)' },
        },
      },
      animation: {
        bounce: 'bounce 1.4s infinite',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
} 