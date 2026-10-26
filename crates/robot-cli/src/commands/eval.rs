//! `robot eval summarize <values...>` — a real subset of PLAN.md §4's
//! proposed `robot eval --bundle <id> --suite <path>`.
//!
//! The full bundle/suite-driven evaluation needs a bundle format and
//! suite runner this workspace does not have yet; what's wired here is
//! the statistic itself, `robot_eval::stats::summarize`, over values
//! supplied directly on the command line (e.g. per-seed returns a caller
//! already collected).

use super::{EXIT_FAILURE, EXIT_USAGE};
use robot_eval::stats::summarize;

/// Runs `robot eval summarize <values...>`. Returns the process exit code.
pub fn summarize_cli(values: &[String]) -> i32 {
    let mut samples = Vec::with_capacity(values.len());
    for raw in values {
        match raw.parse::<f64>() {
            Ok(v) => samples.push(v),
            Err(_) => {
                eprintln!("error: {raw:?} is not a number");
                return EXIT_USAGE;
            }
        }
    }

    match summarize(&samples) {
        Ok(stats) => {
            println!("n:                        {}", stats.n);
            println!("mean:                     {:.6}", stats.mean);
            println!("std_dev:                  {:.6}", stats.std_dev);
            println!(
                "95% CI half-width:        {:.6}",
                stats.confidence_95_halfwidth
            );
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            EXIT_FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizing_valid_numbers_succeeds() {
        let values = ["1.0".to_owned(), "2.0".to_owned(), "3.0".to_owned()];
        assert_eq!(summarize_cli(&values), 0);
    }

    #[test]
    fn a_non_numeric_argument_is_a_usage_error() {
        let values = ["1.0".to_owned(), "not-a-number".to_owned()];
        assert_eq!(summarize_cli(&values), EXIT_USAGE);
    }

    #[test]
    fn fewer_than_two_samples_is_a_failure_not_a_usage_error() {
        let values = ["1.0".to_owned()];
        assert_eq!(summarize_cli(&values), EXIT_FAILURE);
    }
}
