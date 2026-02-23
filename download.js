/**
 * PureBee — Model Downloader
 *
 * Downloads model weights for PureBee. Uses only Node.js built-in https.
 * Zero dependencies.
 *
 * Usage:
 *   node download.js llama3      — Llama 3.2 1B Instruct (~770MB)
 *   node download.js smollm      — SmolLM2 135M Instruct (~145MB)
 *   node download.js 15M         — TinyStories 15M (~58MB)
 */

'use strict';

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const MODELS_DIR = path.join(__dirname, 'models');

const MODELS = {
  '15M': {
    name: 'stories15M.bin',
    url: 'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin',
    desc: 'TinyStories 15M (6 layers, dim=288)',
    size: '~58MB',
  },
  '42M': {
    name: 'stories42M.bin',
    url: 'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin',
    desc: 'TinyStories 42M (8 layers, dim=512)',
    size: '~167MB',
  },
  '110M': {
    name: 'stories110M.bin',
    url: 'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin',
    desc: 'TinyStories 110M (12 layers, dim=768)',
    size: '~438MB',
  },
  'smollm': {
    name: 'SmolLM2-135M-Instruct-Q8_0.gguf',
    url: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf',
    desc: 'SmolLM2 135M Instruct Q8 (30 layers, dim=576, GGUF)',
    size: '~145MB',
  },
  'llama3': {
    name: 'Llama-3.2-1B-Instruct-Q4_0.gguf',
    url: 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf',
    desc: 'Llama 3.2 1B Instruct Q4 (16 layers, dim=2048, GGUF)',
    size: '~770MB',
  },
  'tokenizer': {
    name: 'tokenizer.bin',
    url: 'https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin',
    desc: 'SentencePiece tokenizer (32K vocabulary)',
    size: '~305KB',
  },
};

/**
 * Download a file with redirect following and progress reporting.
 */
function downloadFile(url, destPath, desc) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    let totalBytes = 0;
    let downloadedBytes = 0;

    function doRequest(requestUrl, redirectCount) {
      if (redirectCount > 5) {
        reject(new Error('Too many redirects'));
        return;
      }

      const protocol = requestUrl.startsWith('https') ? https : http;
      const req = protocol.get(requestUrl, (res) => {
        // Handle redirects
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume(); // consume response to free memory
          doRequest(res.headers.location, redirectCount + 1);
          return;
        }

        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${requestUrl}`));
          return;
        }

        totalBytes = parseInt(res.headers['content-length'], 10) || 0;
        let lastPercent = -1;

        res.on('data', (chunk) => {
          downloadedBytes += chunk.length;
          if (totalBytes > 0) {
            const percent = Math.floor(downloadedBytes / totalBytes * 100);
            if (percent !== lastPercent && percent % 10 === 0) {
              lastPercent = percent;
              const mb = (downloadedBytes / 1024 / 1024).toFixed(1);
              const totalMb = (totalBytes / 1024 / 1024).toFixed(1);
              process.stdout.write(`\r    ${desc}: ${mb}MB / ${totalMb}MB (${percent}%)`);
            }
          }
        });

        res.pipe(file);

        file.on('finish', () => {
          file.close();
          const mb = (downloadedBytes / 1024 / 1024).toFixed(1);
          console.log(`\r    ${desc}: ${mb}MB — done                    `);
          resolve();
        });
      });

      req.on('error', (err) => {
        fs.unlink(destPath, () => {}); // cleanup partial file
        reject(err);
      });

      req.setTimeout(300000, () => {
        req.destroy();
        reject(new Error('Download timeout'));
      });
    }

    doRequest(url, 0);
  });
}

async function main() {
  // Parse args: node download.js [15M|42M|110M|all]
  const arg = process.argv[2] || '15M';
  const allModels = ['15M', '42M', '110M', 'smollm', 'llama3'];
  const ggufModels = ['smollm', 'llama3'];
  const requestedModels = arg === 'all' ? allModels :
    ggufModels.includes(arg.toLowerCase()) ? [arg.toLowerCase()] : [arg];

  console.log('');
  console.log('  ╔═════════════════════════════════════════╗');
  console.log('  ║   PureBee — Model Download              ║');
  console.log('  ╚═════════════════════════════════════════╝');
  console.log('');
  console.log(`  Models: ${requestedModels.join(', ')}`);
  console.log('');

  if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
    console.log(`  Created ${MODELS_DIR}`);
  }

  // Download tokenizer (unless only downloading GGUF models which embed their own)
  const needsTokenizer = requestedModels.some(m => !ggufModels.includes(m));
  const files = [
    ...(needsTokenizer ? [MODELS['tokenizer']] : []),
    ...requestedModels.map(m => MODELS[m]).filter(Boolean),
  ];

  for (const file of files) {
    const destPath = path.join(MODELS_DIR, file.name);

    if (fs.existsSync(destPath)) {
      const stat = fs.statSync(destPath);
      if (stat.size > 100) { // skip empty/corrupt files
        console.log(`  ✓ ${file.name} already exists (${(stat.size / 1024 / 1024).toFixed(1)}MB)`);
        continue;
      }
      fs.unlinkSync(destPath); // remove empty file
    }

    console.log(`  Downloading ${file.desc} (${file.size})...`);

    try {
      await downloadFile(file.url, destPath, file.name);
    } catch (err) {
      console.error(`\n  ✗ Failed to download ${file.name}: ${err.message}`);
      console.error(`    You can manually download from:`);
      console.error(`    ${file.url}`);
      console.error(`    Place it in: ${MODELS_DIR}`);
      process.exit(1);
    }
  }

  console.log('');
  console.log('  All files ready.');
  console.log('');
  if (requestedModels.includes('llama3')) {
    console.log('  Run:');
    console.log('    node --max-old-space-size=4096 chat-llama3.js');
  } else if (requestedModels.includes('smollm')) {
    console.log('  Run:');
    console.log('    node chat.js smollm');
  } else {
    console.log('  Run:');
    console.log('    node run.js [15M|42M|110M]');
  }
  console.log('');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
