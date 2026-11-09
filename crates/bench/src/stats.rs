use std::cmp::Ordering;

pub fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_of_odd_count() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn median_of_even_count_is_mean_of_middle_two() {
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }

    #[test]
    fn median_of_one_value() {
        assert_eq!(median(vec![7.5]), Some(7.5));
    }

    #[test]
    fn median_of_nothing() {
        assert_eq!(median(Vec::new()), None);
    }
}
