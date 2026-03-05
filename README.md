## Big Web Simulation

This folder simulates a larger web system that contains the **Minimum Web** project as a sub-site.

### Minimum Web (portrait galleries)

Inside `big-web-simulation/` you will find **Minimum Web**, a very simple static project that hosts **three small portrait gallery sites**, each with **100 dummy portraits** (men and women).

#### Structure

- **`big-web-simulation/index.html`** – Landing page with links to all three sites.
- **`big-web-simulation/site1/index.html`** – First portrait gallery.
- **`big-web-simulation/site2/index.html`** – Second portrait gallery.
- **`big-web-simulation/site3/index.html`** – Third portrait gallery.
- **`big-web-simulation/styles.css`** – Shared styling for all pages.
- **`big-web-simulation/portraits.js`** – Shared JavaScript that generates 100 portrait cards per site.

#### Portrait Images

All portraits use online dummy images from a public placeholder API (`randomuser.me`).  
Each gallery page shows 100 portraits in a responsive grid.

#### How to view

1. Open `big-web-simulation/index.html` in a browser (for example by double-clicking it), or  
2. Run a small local web server in the project root and open `http://localhost:PORT/big-web-simulation/`.


