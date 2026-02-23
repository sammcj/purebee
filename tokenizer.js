/**
 * PureBee Tokenizer — minimal character-level tokenizer
 * 
 * For Phase 1 proof of concept we use a simple character-level
 * tokenizer. Real deployment would use tiktoken (GPT-2 BPE).
 * The architecture is identical — this just proves the pipeline.
 */

'use strict';

class CharTokenizer {
  constructor() {
    // Build a vocabulary from printable ASCII + common chars
    this.vocab = [];
    this.tokenToId = {};
    this.idToToken = {};

    // Special tokens
    this._addToken('<|endoftext|>');
    this._addToken('<|pad|>');
    this._addToken('<|unk|>');

    // Printable ASCII
    for (let i = 32; i < 127; i++) {
      this._addToken(String.fromCharCode(i));
    }

    // Common subwords for slightly better generation
    const common = [
      'the', 'The', 'and', 'And', 'is', 'was', 'are', 'were',
      'he', 'she', 'it', 'they', 'we', 'I', 'you',
      'in', 'on', 'at', 'to', 'of', 'for', 'with',
      'a', 'an', 'this', 'that', 'his', 'her',
      ' the', ' and', ' of', ' to', ' a', ' in',
      '\n', '\t', '  ', '   '
    ];
    for (const w of common) this._addToken(w);
  }

  _addToken(token) {
    if (this.tokenToId[token] === undefined) {
      const id = this.vocab.length;
      this.vocab.push(token);
      this.tokenToId[token] = id;
      this.idToToken[id] = token;
    }
  }

  get vocabSize() { return this.vocab.length; }
  get eosId() { return this.tokenToId['<|endoftext|>']; }
  get unkId() { return this.tokenToId['<|unk|>']; }

  encode(text) {
    const ids = [];
    let i = 0;
    while (i < text.length) {
      // Try multi-char tokens first (longest match)
      let matched = false;
      for (let len = 6; len > 1; len--) {
        const chunk = text.slice(i, i + len);
        if (this.tokenToId[chunk] !== undefined) {
          ids.push(this.tokenToId[chunk]);
          i += len;
          matched = true;
          break;
        }
      }
      if (!matched) {
        const ch = text[i];
        ids.push(this.tokenToId[ch] !== undefined ? this.tokenToId[ch] : this.unkId);
        i++;
      }
    }
    return ids;
  }

  decode(ids) {
    return ids.map(id => this.idToToken[id] || '?').join('');
  }
}

module.exports = { CharTokenizer };
