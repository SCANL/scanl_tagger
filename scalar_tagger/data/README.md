# Bundled data

`english_words.txt.gz` backs the `dictionary` flag in tagging responses. It holds one lowercased word per line, deduplicated and sorted, taken from the NLTK `words` corpus:

- `en`: the Unix `words` list (Webster's Second International Dictionary, 1934, which is in the public domain). See https://en.wikipedia.org/wiki/Words_(Unix).
- `en-basic`: 850 words from C. K. Ogden's *The ABC of Basic English* (1932).

It is bundled so the tagger starts without downloading NLTK data. To regenerate it:

```python
import gzip, nltk
words = sorted(set(w.lower() for w in nltk.corpus.words.words()))
with gzip.GzipFile("english_words.txt.gz", "wb", mtime=0) as f:
    f.write(("\n".join(words) + "\n").encode())
```
