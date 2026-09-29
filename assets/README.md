# assets/ — images for GitHub (README banner, logo, diagrams)

Drop files here and reference them from `README.md` with a relative path:

```markdown
![inference engine overview](assets/banner.png)
```

Conventions:

| File | Purpose | Suggested size/format |
|---|---|---|
| `banner.png` | wide header at top of README | 1280×640 PNG (or SVG), < 500 KB |
| `logo.png` / `logo.svg` | square icon (docs, social posts) | 512×512, transparent background |
| `arch.png` / `arch.svg` | architecture diagram (mirrors `SPEC/01`) | SVG preferred (scales, diffable) |
| `demo-*.png` | terminal screenshots (chat mode, bench table) | actual terminal width, trim chrome |

Rules:
- Compress everything (SVG for diagrams, optimized PNG for screenshots).
  Big binaries slow every clone for marginal visual gain.
- Use lowercase hyphenated names (`perf-table-mac.png`, not `Perf Table v2.PNG`).
- Always include alt text in the markdown reference (accessibility + broken-link fallback).
- Nothing generated or gitignored lives here — unlike `build/` and
  `models/*/weights/`, everything in `assets/` is committed.
- The repo's link-preview thumbnail is *not* a file: set it in GitHub →
  Settings → General → Social preview (1280×640 upload).
