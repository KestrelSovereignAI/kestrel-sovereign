// Execute the compiled browser search engine offline. Only its fetch transport
// is replaced; Pagefind loads and ranks its real metadata, WASM and index chunks.
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [dist, base, ...queries] = process.argv.slice(2);
const indexPath = `${base}/pagefind/`;
globalThis.fetch = async (input) => {
  const url = new URL(input, 'https://handbook.invalid');
  assert.equal(url.origin, 'https://handbook.invalid');
  assert.ok(url.pathname.startsWith(indexPath), `unexpected index request: ${url.pathname}`);
  const relative = url.pathname.slice(indexPath.length);
  assert.ok(!relative.split('/').includes('..'));
  return new Response(await readFile(path.join(dist, 'pagefind', relative)));
};
// This is the same language the compiled handbook declares on <html>.
globalThis.document = {
  currentScript: null,
  querySelector: () => ({ getAttribute: () => 'en' }),
};
const pagefind = await import(pathToFileURL(path.join(dist, 'pagefind/pagefind.js')));
// Use the exact ranking options bundled into Starlight's search dialog. An
// older projection has none, so this also measures the pre-fix ranking.
const rankingFile = path.join(dist, '../src/search-ranking.js');
const ranking = existsSync(rankingFile) ? (await import(pathToFileURL(rankingFile))).searchRanking : {};
await pagefind.options({ basePath: indexPath, ranking });
try {
  const results = {};
  for (const query of queries) {
    const search = await pagefind.search(query);
    results[query] = await Promise.all(search.results.slice(0, 5).map(async (hit) => {
      const data = await hit.data();
      return { url: data.url, title: data.meta.title };
    }));
  }
  console.log(JSON.stringify(results));
} finally {
  await pagefind.destroy();
}
