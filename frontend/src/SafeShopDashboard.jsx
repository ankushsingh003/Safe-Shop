
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>SafeShop — Real-Time Analytics</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/tabler-icons.min.css" />
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f9f9f8;
        color: #1a1a1a;
        -webkit-font-smoothing: antialiased;
      }
      :root {
        --color-background-primary:   #ffffff;
        --color-background-secondary: #f4f3ef;
        --color-background-tertiary:  #f9f9f8;
        --color-text-primary:         #1a1a1a;
        --color-text-secondary:       #6b6b68;
        --color-border-tertiary:      rgba(0,0,0,0.12);
        --color-border-secondary:     rgba(0,0,0,0.22);
        --font-mono:                  'SF Mono', 'Fira Code', monospace;
        --border-radius-md:           8px;
        --border-radius-lg:           12px;
        --border-radius-xl:           16px;
      }
      @media (prefers-color-scheme: dark) {
        body { background: #1a1a1a; color: #f0efeb; }
        :root {
          --color-background-primary:   #242423;
          --color-background-secondary: #2c2c2a;
          --color-background-tertiary:  #1a1a1a;
          --color-text-primary:         #f0efeb;
          --color-text-secondary:       #9e9d99;
          --color-border-tertiary:      rgba(255,255,255,0.1);
          --color-border-secondary:     rgba(255,255,255,0.18);
        }
      }
      input, button, select, textarea {
        font-family: inherit;
        font-size: 14px;
        padding: 8px 12px;
        border: 0.5px solid var(--color-border-secondary);
        border-radius: var(--border-radius-md);
        background: var(--color-background-primary);
        color: var(--color-text-primary);
        outline: none;
        cursor: pointer;
      }
      input { cursor: text; }
      input:focus { box-shadow: 0 0 0 2px rgba(55,138,221,0.25); border-color: #378add; }
      button:hover { background: var(--color-background-secondary); }
      button:active { transform: scale(0.98); }
      button:disabled { opacity: 0.45; cursor: not-allowed; }
    </style>
  </head>
  <body>
    <div id="root" style="max-width: 960px; margin: 0 auto; padding: 1rem 1.25rem;"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>