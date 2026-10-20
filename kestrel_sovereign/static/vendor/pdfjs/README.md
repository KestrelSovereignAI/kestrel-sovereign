# PDF.js runtime

Unmodified files from Mozilla's `pdfjs-dist` **6.4.299** npm distribution,
licensed under Apache-2.0 (see `LICENSE` and the component licenses in each
asset directory). Source: https://github.com/mozilla/pdf.js

The npm archive `pdfjs-dist-6.4.299.tgz` was verified against this SHA-512 SRI:

```
sha512-AVl138zALtfaAPvADulE0PZThbYzCBS79nL4pOSL/6Sm/4AH5A21BD9VHt97OlCuzJuCpmeZtAtkinisF4Vb1g==
```

`SHA256SUMS` records every included upstream runtime/license file. To update,
download an exact version from npm, verify its archive integrity, copy
`build/pdf.min.mjs`, `build/pdf.worker.min.mjs`, `LICENSE`, `cmaps/`,
`standard_fonts/`, and `wasm/` except `quickjs-*`, then regenerate the checksum
list. Keep upstream component licenses. Run the artifact PDF frontend tests
and `node --test tests/frontend/artifact_pdf.browser.mjs` after updating.

The application renders verified source bytes onto one bounded canvas at a
time. It does not instantiate PDF.js's viewer, annotation, form, action or
scripting layers; the QuickJS scripting runtime is not included. All runtime
assets are served locally. There are no CDN requests. Unsupported, encrypted,
malformed or timed-out documents retain a download action and show an explicit
preview error.
