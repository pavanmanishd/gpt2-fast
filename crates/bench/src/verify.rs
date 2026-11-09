pub struct PromptCheck {
    pub near_tie: bool,
    pub logits_ok: bool,
    pub tokens_match: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verdict {
    pub matches_reference: bool,
    pub near_tie_mismatch: bool,
}

pub fn verdict(checks: &[PromptCheck]) -> Verdict {
    let mut matches_reference = true;
    let mut near_tie_mismatch = false;
    for check in checks {
        if !check.logits_ok {
            matches_reference = false;
        }
        if check.tokens_match {
            continue;
        }
        if check.near_tie {
            near_tie_mismatch = true;
        } else {
            matches_reference = false;
        }
    }
    Verdict {
        matches_reference,
        near_tie_mismatch,
    }
}

#[cfg(test)]
mod tests {
    use super::{verdict, PromptCheck};

    fn check(near_tie: bool, logits_ok: bool, tokens_match: bool) -> PromptCheck {
        PromptCheck {
            near_tie,
            logits_ok,
            tokens_match,
        }
    }

    #[test]
    fn a_mismatch_on_a_near_tie_prompt_is_reported_but_does_not_fail() {
        let outcome = verdict(&[
            check(false, true, true),
            check(true, true, false),
            check(true, true, true),
        ]);
        assert!(outcome.matches_reference);
        assert!(outcome.near_tie_mismatch);
    }

    #[test]
    fn a_mismatch_on_a_strict_prompt_fails_even_next_to_a_near_tie() {
        let outcome = verdict(&[
            check(true, true, false),
            check(false, true, false),
            check(false, true, true),
        ]);
        assert!(!outcome.matches_reference);
        assert!(outcome.near_tie_mismatch);
    }

    #[test]
    fn logits_beyond_tolerance_always_fail() {
        let outcome = verdict(&[check(false, true, true), check(true, false, false)]);
        assert!(!outcome.matches_reference);
        assert!(outcome.near_tie_mismatch);
    }

    #[test]
    fn matching_near_ties_report_nothing() {
        let outcome = verdict(&[check(false, true, true), check(true, true, true)]);
        assert!(outcome.matches_reference);
        assert!(!outcome.near_tie_mismatch);
    }

    #[test]
    fn an_empty_session_matches() {
        let outcome = verdict(&[]);
        assert!(outcome.matches_reference);
        assert!(!outcome.near_tie_mismatch);
    }
}
