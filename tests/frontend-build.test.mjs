import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const node = process.execPath;
const builder = path.join(root, 'scripts', 'build-frontend.mjs');

test('frontend build injects the public API origin for a non-local deployment', async () => {
  const result = spawnSync(node, [builder], { cwd: root, env: { ...process.env, SYNAPSENSE_API_BASE_URL: 'https://inference.example' }, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const config = await readFile(path.join(root, 'dist', 'runtime-config.js'), 'utf8');
  assert.match(config, /"apiBaseUrl":"https:\/\/inference\.example"/);
  const html = await readFile(path.join(root, 'dist', 'index.html'), 'utf8');
  assert.match(html, /runtime-config\.js/);
});

test('frontend build rejects insecure non-local API origins', () => {
  const result = spawnSync(node, [builder], { cwd: root, env: { ...process.env, SYNAPSENSE_API_BASE_URL: 'http://inference.example' }, encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /must use HTTPS/);
});
