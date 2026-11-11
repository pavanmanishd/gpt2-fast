# N11: cache blocking for prompts

**Result:** `results/log.jsonl`, milestones `N11`, `N11-int8`, `N11-gpt2-medium-f32`. Raw output in `results/runs/N11*/`. Code at tag `n11`.

## What changed

N8 found that on bigger models gpt2-fast processed prompts slower than PyTorch (123 against 406 tok/s on gpt2-medium), and guessed why: the prompt kernel took four prompt rows at a time and streamed the whole weight matrix past them, so for a 256-token prompt every weight was read from main memory 64 times. gpt2-medium's MLP matrices are 16.8 MB, larger than the M4 Pro's 16 MB L2 cache.

N11 adds one outer loop. The weights are walked in blocks of 16 output rows (64 KB to 256 KB), and every prompt row goes over a block while it is in cache. The inner 4 × 4 kernel is unchanged and each output keeps its summation order, so the results are bit-identical to N10. Decoding one token never uses this path and is unchanged.

Block sizes tried (host, gpt2-medium, 4 threads, 255-token prompt): 8 and 16 rows were tied, 32 and 64 were slower. At 32 rows of 1,024 floats a block is 128 KB, the size of the core's L1 data cache.

## Results

Same gated protocol as N10: 5 sessions, each kept only if a control run of N6 right before and after it ran at full speed.

| prompt processing, tok/s | N10 | N11 | change |
|---|---|---|---|
| gpt2 (124M), median over prompts | 1,265 | **1,397** | +10% |
| gpt2-medium, median over prompts | 368 | **416** | +13% |
| gpt2-medium, 255-token prompt | 398 | 455 | +14% |
| gpt2-medium, 16-token prompt | 269 | 266 | |

The gain grows with prompt length, as it should: a 16-token prompt reuses each weight only 16 times and fits the old pattern anyway. On gpt2-medium gpt2-fast now processes prompts slightly faster than PyTorch (416 against 406 tok/s); PyTorch had been 3.3 times faster at N8.

Decode does not use this code. N11's decode measured 323.6 tok/s against N10's 306.5 in the same evening, which is session-to-session variation: N10's five session medians ranged 275-324, N11's 317-337.

int8 prompt processing is unchanged (976 → 995 tok/s): its matrices are four times smaller and already fit in cache, so there was nothing to save.

The host test on a quiet moment showed more (+36% on gpt2-medium at 4 threads for the 255-token prompt, timed with `--max-new-tokens 1`). The bench's headline number is the median across five prompt lengths of 16 to 255 tokens, and short prompts barely benefit.

## What didn't work

- Blocks of 32 and 64 rows: slower on gpt2-medium, consistent with a block no longer fitting in L1.
- Nothing else was tried; this step is one loop.

## Post draft

> Step 11: one more loop. For long prompts the kernel streamed each 16.8 MB weight matrix from memory 64 times, once per group of 4 prompt rows. Walk the weights in blocks of 16 rows and pass every prompt row over a block while it is in cache: same bits out, prompt processing +13% on GPT-2 medium, which now edges past PyTorch (416 vs 406 tok/s; PyTorch was 3.3x faster before).
