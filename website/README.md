# Zarqaa Al-Yamama & Sahm — Marketing Website

Arabic RTL single-page marketing site for the drone interception system.

## Opening locally

Just open `index.html` in any modern browser — no build step, no server required.

```
website/index.html  →  double-click or drag into browser
```

All assets are loaded from CDN (Tailwind CSS, Font Awesome, Google Fonts).
An internet connection is required to render fonts and icons correctly.

## Deploy to GitHub Pages

1. Push the `website/` folder to GitHub.
2. Go to **Settings → Pages** in your repository.
3. Set **Source** to `Deploy from a branch`, choose your branch, set the folder to `/website`.
4. GitHub Pages will serve `index.html` at `https://<username>.github.io/<repo>/`.

Alternatively, copy `index.html` to the repo root and set folder to `/ (root)`.

## Customization

| What to change | Where in index.html |
|---|---|
| Primary accent color (cyan `#06b6d4`) | CSS `:root` block + Tailwind config |
| Success/lock color (emerald `#10b981`) | CSS `:root` block |
| Contact email | `<a href="mailto:...">` near the bottom |
| Hero headline | `<h1>` inside `section` with radar sweep |
| Statistics numbers | Big `stat-num` divs in Section 2 |
| Partner badges | `.partner-badge` spans in Section 6 |
| Radar sweep speed | `animation: radar-sweep 4s` — lower number = faster |
| Drone float speed | `animation: float 5s` |
