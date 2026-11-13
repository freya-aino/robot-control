//! "Test whether style is recognizable without degrading task and stability
//! metrics" (SPECS.md Section 12.5).
//!
//! **Recognizability** is measured, not asserted: each behavior log is
//! summarized as a feature vector (for instance the frequency of each
//! expressive skill, the mean resolved cadence, idle amplitude), and a
//! leave-one-out nearest-centroid classifier tries to name the persona
//! that produced each log. [`recognizability`] reports its accuracy next to
//! chance (one over the persona count); a style counts as recognizable
//! only when accuracy beats chance by the caller's margin.
//!
//! **Non-degradation** compares task and stability metric samples under a
//! persona with the same metrics under the neutral (no-expression)
//! configuration: [`degradation`] lists every metric whose mean got worse
//! by more than the caller's tolerance. Lower is better for every metric
//! passed in; a caller turns a higher-is-better metric into a cost first.

/// One persona's behavior logs as feature vectors.
#[derive(Clone, PartialEq, Debug)]
pub struct PersonaLogs {
    /// Persona name.
    pub persona: String,
    /// One feature vector per log.
    pub logs: Vec<Vec<f64>>,
}

/// The recognizability result.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Recognizability {
    /// Leave-one-out nearest-centroid accuracy.
    pub accuracy: f64,
    /// Chance accuracy.
    pub chance: f64,
    /// Logs classified.
    pub trials: usize,
}

impl Recognizability {
    /// Whether accuracy beats chance by `margin`.
    #[must_use]
    pub fn recognizable(&self, margin: f64) -> bool {
        self.accuracy > self.chance + margin
    }
}

fn centroid<'a>(logs: impl Iterator<Item = &'a Vec<f64>>, width: usize) -> Option<Vec<f64>> {
    let mut sum = vec![0.0; width];
    let mut n = 0usize;
    for l in logs {
        for (s, v) in sum.iter_mut().zip(l) {
            *s += v;
        }
        n += 1;
    }
    (n > 0).then(|| sum.into_iter().map(|s| s / n as f64).collect())
}

/// Leave-one-out nearest-centroid recognizability over `personas`.
#[must_use]
pub fn recognizability(personas: &[PersonaLogs]) -> Recognizability {
    let width = personas
        .iter()
        .flat_map(|p| p.logs.first())
        .map(Vec::len)
        .next()
        .unwrap_or(0);
    let (mut correct, mut trials) = (0usize, 0usize);
    for (pi, p) in personas.iter().enumerate() {
        for (li, log) in p.logs.iter().enumerate() {
            let best = personas
                .iter()
                .enumerate()
                .filter_map(|(qi, q)| {
                    let c = if qi == pi {
                        centroid(
                            q.logs
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| *i != li)
                                .map(|(_, l)| l),
                            width,
                        )
                    } else {
                        centroid(q.logs.iter(), width)
                    }?;
                    let d: f64 = c.iter().zip(log).map(|(a, b)| (a - b).powi(2)).sum();
                    Some((qi, d))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            trials += 1;
            if best.is_some_and(|(qi, _)| qi == pi) {
                correct += 1;
            }
        }
    }
    Recognizability {
        accuracy: if trials == 0 {
            0.0
        } else {
            correct as f64 / trials as f64
        },
        chance: if personas.is_empty() {
            0.0
        } else {
            1.0 / personas.len() as f64
        },
        trials,
    }
}

/// A metric's samples under the neutral configuration and under a persona.
#[derive(Clone, PartialEq, Debug)]
pub struct MetricSamples {
    /// Metric name (lower is better).
    pub name: String,
    /// Samples under the neutral configuration.
    pub neutral: Vec<f64>,
    /// Samples under the persona.
    pub persona: Vec<f64>,
}

/// Metrics whose persona mean exceeds the neutral mean by more than
/// `tolerance` (relative to the neutral mean's magnitude, with an absolute
/// floor of `tolerance` for a zero mean), with the relative change.
#[must_use]
pub fn degradation(metrics: &[MetricSamples], tolerance: f64) -> Vec<(String, f64)> {
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    metrics
        .iter()
        .filter_map(|m| {
            let (a, b) = (mean(&m.neutral), mean(&m.persona));
            let scale = a.abs().max(1.0);
            let change = (b - a) / scale;
            (change > tolerance).then(|| (m.name.clone(), change))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::rng::{Rng, SeedStream};

    fn logs(name: &str, centre: [f64; 2], spread: f64, rng: &mut Rng) -> PersonaLogs {
        PersonaLogs {
            persona: name.into(),
            logs: (0..20)
                .map(|_| {
                    vec![
                        centre[0] + rng.normal(0.0, spread),
                        centre[1] + rng.normal(0.0, spread),
                    ]
                })
                .collect(),
        }
    }

    #[test]
    fn distinct_styles_are_recognizable_and_identical_ones_are_not() {
        let mut rng = Rng::new(SeedStream::new(5, 0));
        let distinct = [
            logs("calm", [0.2, 0.1], 0.05, &mut rng),
            logs("showboat", [0.8, 0.6], 0.05, &mut rng),
            logs("counter", [0.2, 0.9], 0.05, &mut rng),
        ];
        let r = recognizability(&distinct);
        assert!(r.recognizable(0.3), "{r:?}");
        let same = [
            logs("a", [0.5, 0.5], 0.1, &mut rng),
            logs("b", [0.5, 0.5], 0.1, &mut rng),
        ];
        let r = recognizability(&same);
        assert!(!r.recognizable(0.2), "{r:?}");
    }

    #[test]
    fn a_worse_metric_is_listed_and_an_equal_one_is_not() {
        let m = [
            MetricSamples {
                name: "falls_per_minute".into(),
                neutral: vec![0.10, 0.12],
                persona: vec![0.30, 0.28],
            },
            MetricSamples {
                name: "tracking_rms".into(),
                neutral: vec![0.05, 0.05],
                persona: vec![0.05, 0.051],
            },
        ];
        let d = degradation(&m, 0.05);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].0, "falls_per_minute");
    }
}
