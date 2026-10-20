# PDF.js runtime

Runtime files from Mozilla's `pdfjs-dist` **6.4.299** npm distribution,
licensed under Apache-2.0 (see `LICENSE` and the component licenses in each
asset directory). Source: https://github.com/mozilla/pdf.js

The npm archive `pdfjs-dist-6.4.299.tgz` was verified against this SHA-512 SRI:

```
sha512-AVl138zALtfaAPvADulE0PZThbYzCBS79nL4pOSL/6Sm/4AH5A21BD9VHt97OlCuzJuCpmeZtAtkinisF4Vb1g==
```

`SHA256SUMS` records every included runtime/license file, including the patched main module. To update,
download an exact version from npm, verify its archive integrity, copy
`build/pdf.min.mjs`, `build/pdf.worker.min.mjs`, `LICENSE`, `cmaps/`,
`standard_fonts/`, and `wasm/` except `quickjs-*`, then regenerate the checksum
list after applying the patch below. Keep upstream component licenses. Run the artifact PDF frontend tests
and `node --test tests/frontend/artifact_pdf.browser.mjs` after updating.

The application renders verified source bytes onto one bounded canvas at a
time. It does not instantiate PDF.js's viewer, annotation, form, action or
scripting layers; the QuickJS scripting runtime is not included. All runtime
assets are served locally. There are no CDN requests. Unsupported, encrypted,
malformed or timed-out documents retain a download action and show an explicit
preview error.

## Local correctness patch

`pdf.min.mjs` has one marked change to `PDFPageProxy._pumpOperatorList`'s
stream-error handler. In upstream 6.4.299 that handler marks the operator list
complete and calls render callbacks before rejecting the promise. A size-limit
error can therefore resolve rendering successfully with missing image content,
even with `stopAtErrors: true`. A real compressed scan reproduced this.

The patch rejects the operator-list promise first and cancels active rendering
with the original error before notifying completion-only consumers. It does not
change PDF parsing or relax error handling. The worker and all other upstream
files remain unmodified. Source images are bounded separately at 80 million
pixels (including 600-dpi A3 scans); output remains at most 4 million pixels.
Oversized images cause an explicit preview error, retaining source download.

Exact upstream main-module SHA-256: `57456c8e0c81e46be31174b499ef77f2b9f5ee46d04412ba627320a36755d4c2`.
Patched main-module SHA-256: `020eb41b2e413b91862c03d8a58d1727c6dedf12c7bf21c5078d242868cecf2b`.

To reproduce against that exact upstream file, prepend the marked comment at
the start of the shipped file and replace this unique minified sequence:

```js
if(o.operatorList){o.operatorList.lastChunk=!0;for(const t of o.renderTasks)t.operatorListChanged();this.#Zr()}if(o.displayReadyCapability)
```

with:

```js
o.opListReadCapability?.reject(t);if(o.operatorList){o.operatorList.lastChunk=!0;for(const r of o.renderTasks)r.cancel?r.cancel(t):r.operatorListChanged();this.#Zr()}if(o.displayReadyCapability)
```

Re-check this change against upstream on every upgrade. The real Chromium test
covers a 300-dpi scan and an over-limit compressed image as well as text pages.
