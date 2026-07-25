# asr-data documentation website

This directory contains the static Next.js documentation site for `asr-data`.

## Local development

```bash
npm ci
npm run dev
```

The production build is generated with `npm run build`. It exports the static
site to `out/` and indexes the exported pages with Pagefind.

## GitHub Pages

The repository workflow builds the site for the project URL
`https://di-osc.github.io/asr-data/`. It is triggered by a published GitHub
Release and deploys the release tag that was published.
