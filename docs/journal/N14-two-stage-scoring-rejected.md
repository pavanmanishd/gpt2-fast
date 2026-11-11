# N14: two-stage scoring (tried, rejected)

**Result:** none recorded. This step made decoding slower and was not merged. The numbers below are host measurements from the attempt, not container results. The code is kept as a patch outside the repository.

## The idea

After N13, decoding one token in the 8-bit mode reads about 137 MB. 41 MB of that (30%) is the scoring layer: the final hidden state against all 50,257 rows of the embedding matrix, only to pick the largest. Decode is limited by how fast memory arrives, so reading fewer bytes should be the way to go faster.

The plan: score every row with a coarse 4-bit copy of the matrix (19 MB), work out how wrong each coarse score can be, and rescore with the real int8 weights only the rows that could still win. Done with a guaranteed error bound, the chosen token is exactly the same.

## What happened

It was exact (0 mismatches over 2,816 generated tokens and 1,644 teacher-forced positions) and about twice as slow:

| model | threads | full scoring | two-stage |
|---|---|---|---|
| gpt2 | 4 | 805 tok/s | 396 |
| gpt2 | 1 | 309 | 172 |
| gpt2-medium | 4 | 164 | 120 |

Two measured reasons:

1. **The guarantee is too weak to shortlist anything.** A 4-bit weight can be off by half a level; added up over 768 dimensions the worst-case error per row is 8 to 39 logits. But the top 1,000 candidates all sit within about 10 logits of the winner, and the winner beats the runner-up by a median of 0.9. With a bound wider than the whole field, the "shortlist" had a median of 45,751 rows out of 50,257 (91%), and the whole vocabulary on one token in a hundred. Stage 2 rescored nearly everything.
2. **4-bit is slower to multiply than 8-bit.** Two weights per byte have to be pulled apart before they can be used, about five instructions per 16 weights. The coarse pass cost 0.84 ms per token; full int8 scoring costs about 0.35 ms. Fewer bytes, more work.

Dropping the guarantee does not rescue it. Taking a fixed top 1,024 by coarse score picked the wrong token 8% of the time on the test passage and was still slower (775 → 496 tok/s), because of reason 2.

## What it tells us

- The chosen-token margin in GPT-2 is small (median gap 0.9 logits), so any cheap pre-filter needs to be accurate to well under a logit to be exact.
- Sub-byte weights are not a free win on this CPU: unpacking them costs more than the memory they save. That also applies to the idea of 4-bit weights for the layers.
- Scoring is 30% of the bytes, so even a free scoring pass would cap the gain at 1.43×.

A phase timing from the same session (4 threads, 8-bit mode, full scoring, about 1.24 ms per token): layer multiplies 0.69 ms, scoring 0.35 ms, attention 0.13-0.20 ms, layer norm about 0.03 ms, with about 6% of the time spent handing work to threads. The multiplies and scoring run at about 120 GB/s of weights, close to what four cores can pull from memory.

## Not tried

- 2-bit weights and "first 256 dimensions only": both have looser bounds than 4-bit by construction, so their shortlists could only be larger.
