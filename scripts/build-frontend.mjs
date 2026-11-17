import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(root, 'app');
const output = path.join(root, 'dist');
const configured = (process.env.SYNAPSENSE_API_BASE_URL ?? '').trim().replace(/\/$/, '');

if (configured) {
  let endpoint;
  try { endpoint = new URL(configured); } catch { throw new Error('SYNAPSENSE_API_BASE_URL must be an absolute URL'); }
  if (!['http:', 'https:'].includes(endpoint.protocol) || endpoint.username || endpoint.password || endpoint.search || endpoint.hash || endpoint.pathname !== '/') {
    throw new Error('SYNAPSENSE_API_BASE_URL must be an origin URL without credentials, path, query, or fragment');
  }
  if (endpoint.protocol !== 'https:' && !['localhost', '127.0.0.1', '[::1]'].includes(endpoint.hostname)) {
    throw new Error('SYNAPSENSE_API_BASE_URL must use HTTPS outside localhost');
  }
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(source, output, { recursive: true });
await writeFile(path.join(output, 'runtime-config.js'), `window.SYNAPSENSE_CONFIG = Object.freeze(${JSON.stringify({ apiBaseUrl: configured })});\n`);
console.log(`Built static frontend at dist/ (API base ${configured || 'local fallback only'})`);
