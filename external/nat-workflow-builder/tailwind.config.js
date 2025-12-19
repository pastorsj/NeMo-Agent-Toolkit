/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // NAT Component Colors - Each type has a distinct color
        'nat-agent': '#A855F7', // Purple - for agents
        'nat-embedder': '#10B981', // Emerald
        'nat-function': '#3B82F6', // Blue
        'nat-function-group': '#6366F1', // Indigo
        'nat-llm': '#F59E0B', // Amber
        'nat-memory': '#EC4899', // Pink
        'nat-object-store': '#8B5CF6', // Violet
        'nat-retriever': '#14B8A6', // Teal
        'nat-auth': '#EF4444', // Red
        'nat-middleware': '#06B6D4', // Cyan
        // Front-end and observability
        'nat-frontend': '#10B981', // Emerald (same as embedder)
        'nat-logger': '#64748B', // Slate
        'nat-telemetry': '#F59E0B', // Amber
        // Evaluation
        'nat-evaluator': '#0EA5E9', // Sky
        // Finetuning components
        'nat-trainer': '#84CC16', // Lime
        'nat-trajectory': '#A855F7', // Purple
        'nat-adapter': '#78716C', // Stone
        // Workflow-level configuration containers
        'nat-workflow': '#76B900', // NVIDIA Green
        'nat-config': '#6B7280', // Gray
        'nat-optimizer': '#E25A1C', // Spark orange
        'nat-finetuner': '#EE4C2C', // PyTorch red

        // UI Colors
        canvas: '#0F172A',
        'canvas-light': '#1E293B',
        sidebar: '#1E293B',
        'sidebar-hover': '#334155',
        accent: '#76B900', // NVIDIA Green
        'accent-hover': '#8BD100',
      },
      fontFamily: {
        display: ['JetBrains Mono', 'Fira Code', 'monospace'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        component: '0 4px 20px -2px rgba(0, 0, 0, 0.3)',
        'component-hover': '0 8px 30px -4px rgba(0, 0, 0, 0.4)',
        glow: '0 0 20px rgba(118, 185, 0, 0.3)',
      },
      animation: {
        'pulse-soft': 'pulse-soft 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in': 'slide-in 0.3s ease-out',
        'fade-in': 'fade-in 0.2s ease-out',
        drop: 'drop 0.3s ease-out',
      },
      keyframes: {
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        'slide-in': {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        drop: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
