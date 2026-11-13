#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u32,
}

pub fn spread(seed: u32, step: u32) -> u32 {
    let mut x = seed ^ 0x9e37_79b9u32.wrapping_mul(step.wrapping_add(1));
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

pub fn uniform(seed: u32, step: u32) -> f32 {
    (spread(seed, step) >> 8) as f32 * (1.0 / 16_777_216.0)
}

pub struct Sampler {
    sampling: Sampling,
    counts: Vec<u32>,
    order: Vec<(f32, u32)>,
    probs: Vec<f32>,
    step: u32,
}

impl Sampler {
    pub fn new(vocab: usize, sampling: Sampling) -> Sampler {
        Sampler {
            sampling,
            counts: vec![0; vocab],
            order: Vec::with_capacity(vocab),
            probs: Vec::with_capacity(vocab.min(1024)),
            step: 0,
        }
    }

    pub fn observe(&mut self, ids: &[u32]) {
        for &id in ids {
            if let Some(count) = self.counts.get_mut(id as usize) {
                *count += 1;
            }
        }
    }

    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        let penalty = self.sampling.repetition_penalty;
        self.order.clear();
        for (index, &logit) in logits.iter().enumerate() {
            let seen = self.counts.get(index).copied().unwrap_or(0) > 0;
            let value = if seen {
                if logit > 0.0 {
                    logit / penalty
                } else {
                    logit * penalty
                }
            } else {
                logit
            };
            self.order.push((value, index as u32));
        }
        if self.order.is_empty() {
            return 0;
        }

        let limit = match self.sampling.top_k {
            0 => self.order.len(),
            k => k.min(self.order.len()),
        };
        let descending = |left: &(f32, u32), right: &(f32, u32)| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        };
        if limit < self.order.len() {
            self.order.select_nth_unstable_by(limit - 1, descending);
            self.order.truncate(limit);
        }
        self.order.sort_unstable_by(descending);

        let token = if self.sampling.temperature <= 0.0 {
            self.order[0].1
        } else {
            self.draw()
        };
        self.step = self.step.wrapping_add(1);
        if let Some(count) = self.counts.get_mut(token as usize) {
            *count += 1;
        }
        token
    }

    fn draw(&mut self) -> u32 {
        let scale = 1.0 / self.sampling.temperature;
        let highest = self.order[0].0;
        self.probs.clear();
        let mut total = 0.0;
        for &(value, _) in &self.order {
            let probability = ((value - highest) * scale).exp();
            self.probs.push(probability);
            total += probability;
        }

        let mut keep = self.order.len();
        let mut cutoff = 1.0;
        if self.sampling.top_p < 1.0 {
            let mut cumulative = 0.0;
            for (index, &probability) in self.probs.iter().enumerate() {
                cumulative += probability / total;
                keep = index + 1;
                if cumulative >= self.sampling.top_p {
                    break;
                }
            }
            cutoff = cumulative;
        }

        let target = uniform(self.sampling.seed, self.step) * cutoff;
        let mut running = 0.0;
        let mut token = self.order[keep - 1].1;
        for (index, &probability) in self.probs[..keep].iter().enumerate() {
            running += probability / total;
            if target < running {
                token = self.order[index].1;
                break;
            }
        }
        token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling(temperature: f32, top_k: usize, top_p: f32, penalty: f32, seed: u32) -> Sampling {
        Sampling {
            temperature,
            top_k,
            top_p,
            repetition_penalty: penalty,
            seed,
        }
    }

    fn picks(logits: &[f32], history: &[u32], settings: Sampling, steps: usize) -> Vec<u32> {
        let mut sampler = Sampler::new(logits.len(), settings);
        sampler.observe(history);
        (0..steps).map(|_| sampler.pick(logits)).collect()
    }

    #[test]
    fn uniform_stays_in_range() {
        for step in 0..4096 {
            let value = uniform(0x1234_5678, step);
            assert!((0.0..1.0).contains(&value), "step {step}: {value}");
        }
        assert_ne!(spread(7, 0), spread(7, 1));
    }

    #[test]
    fn temperature_zero_equals_argmax() {
        let logits = [0.3, -1.5, 4.25, 4.25, 2.0];
        let settings = sampling(0.0, 40, 0.95, 1.0, 9);
        assert_eq!(picks(&logits, &[], settings, 1)[0], 2);
        assert_eq!(
            picks(&logits, &[], settings, 1)[0],
            gpt2_engine::argmax(&logits)
        );
    }

    #[test]
    fn top_k_one_is_argmax() {
        let logits = [-2.0, 0.5, 0.5, -0.1];
        for seed in 0..32 {
            let settings = sampling(0.7, 1, 0.95, 1.0, seed);
            assert_eq!(picks(&logits, &[], settings, 1)[0], 1);
        }
    }

    #[test]
    fn top_p_zero_keeps_the_most_likely() {
        let logits = [1.0, 2.0, 0.5];
        for seed in 0..32 {
            let settings = sampling(1.0, 40, 0.0, 1.0, seed);
            assert_eq!(picks(&logits, &[], settings, 1)[0], 1);
        }
    }

    #[test]
    fn top_k_limits_the_support() {
        let logits = [5.0, 4.0, 3.0];
        let settings = sampling(1.0, 2, 1.0, 1.0, 3);
        for token in picks(&logits, &[], settings, 2000) {
            assert!(token < 2, "token {token} outside top-k");
        }
    }

    #[test]
    fn top_p_limits_the_support() {
        let logits = [8.0, 7.0, 0.0];
        let settings = sampling(1.0, 3, 0.5, 1.0, 5);
        for token in picks(&logits, &[], settings, 2000) {
            assert_eq!(token, 0, "token {token} outside top-p");
        }
    }

    #[test]
    fn penalty_lowers_seen_positive_logits() {
        let logits = [2.0, 2.0, 0.0];
        let settings = sampling(0.0, 40, 0.95, 2.0, 1);
        assert_eq!(picks(&logits, &[0], settings, 1)[0], 1);
        assert_eq!(picks(&logits, &[], settings, 1)[0], 0);
    }

    #[test]
    fn penalty_lowers_seen_negative_logits() {
        let logits = [-1.0, -1.1];
        let settings = sampling(0.0, 40, 0.95, 2.0, 1);
        assert_eq!(picks(&logits, &[0], settings, 1)[0], 1);
    }

    #[test]
    fn penalty_of_one_changes_nothing() {
        let logits = [0.4, 0.1, -0.2, 0.9];
        let plain = picks(&logits, &[], sampling(0.8, 40, 0.95, 1.0, 11), 50);
        let seen = picks(&logits, &[0, 1, 2, 3], sampling(0.8, 40, 0.95, 1.0, 11), 50);
        assert_eq!(plain, seen);
    }

    #[test]
    fn same_seed_reproduces_the_tokens() {
        let logits = [1.0, 1.2, 0.8, 0.4, -0.3, 2.0, 1.9];
        let settings = sampling(0.9, 5, 0.9, 1.2, 4242);
        let first = picks(&logits, &[0, 2], settings, 64);
        let second = picks(&logits, &[0, 2], settings, 64);
        assert_eq!(first, second);
        let other = picks(&logits, &[0, 2], sampling(0.9, 5, 0.9, 1.2, 4243), 64);
        assert_ne!(first, other);
    }

    #[test]
    fn generated_tokens_feed_the_penalty() {
        let logits = [3.0, 2.0];
        let mut sampler = Sampler::new(2, sampling(0.0, 40, 0.95, 2.0, 0));
        assert_eq!(sampler.pick(&logits), 0);
        assert_eq!(sampler.pick(&logits), 1);
        let mut sampler = Sampler::new(2, sampling(0.0, 40, 0.95, 1.6, 0));
        assert_eq!(sampler.pick(&logits), 0);
        assert_eq!(sampler.pick(&logits), 1);
    }

    #[test]
    fn single_token_vocabulary() {
        let logits = [0.0];
        assert_eq!(
            picks(&logits, &[], sampling(0.8, 40, 0.95, 1.2, 0), 4),
            vec![0; 4]
        );
        assert_eq!(
            picks(&logits, &[], sampling(0.0, 40, 0.95, 1.2, 0), 1)[0],
            0
        );
    }
}
