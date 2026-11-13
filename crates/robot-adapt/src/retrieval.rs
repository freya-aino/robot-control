//! Retrieval-conditioned adaptation (SPECS.md Section 11.3, row 8):
//! select prior episodes similar to the observed deployment condition and
//! adapt on their union with the few real episodes.
//!
//! The row's limitation is that "similarity in observation space is not
//! similarity in dynamics; the admission filter, not the metric, is what
//! makes it safe". [`retrieve`] therefore cannot be called without an
//! admission filter, applies it *before* ranking (so a near but
//! inadmissible episode can never displace a farther admissible one by
//! being counted and dropped later), and reports how many candidates the
//! filter refused. Section 9.7's corpus rules (licences, embodiment
//! provenance) are what a caller's filter encodes.

/// A prior episode as the retriever sees it.
#[derive(Clone, PartialEq, Debug)]
pub struct PriorEpisode {
    /// Stable identifier (a content ID).
    pub id: String,
    /// Descriptor of the condition it was recorded under.
    pub descriptor: Vec<f64>,
    /// Embodiment it was recorded on.
    pub embodiment: String,
    /// Whether its licence permits training use.
    pub training_permitted: bool,
}

/// What a retrieval returned.
#[derive(Clone, PartialEq, Debug)]
pub struct Retrieval {
    /// Selected `(id, distance)`, nearest first.
    pub selected: Vec<(String, f64)>,
    /// Candidates the admission filter refused.
    pub refused: usize,
}

/// The `k` admissible episodes nearest `query` (Euclidean distance on the
/// descriptor; a descriptor of another width is inadmissible).
pub fn retrieve(
    corpus: &[PriorEpisode],
    query: &[f64],
    k: usize,
    admit: impl Fn(&PriorEpisode) -> bool,
) -> Retrieval {
    let mut refused = 0;
    let mut ranked: Vec<(String, f64)> = Vec::new();
    for e in corpus {
        if e.descriptor.len() != query.len() || !admit(e) {
            refused += 1;
            continue;
        }
        let d = e
            .descriptor
            .iter()
            .zip(query)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        ranked.push((e.id.clone(), d));
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    Retrieval {
        selected: ranked,
        refused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(id: &str, d: f64, embodiment: &str, ok: bool) -> PriorEpisode {
        PriorEpisode {
            id: id.into(),
            descriptor: vec![d],
            embodiment: embodiment.into(),
            training_permitted: ok,
        }
    }

    #[test]
    fn the_filter_runs_before_ranking() {
        let corpus = [
            ep("near-other-robot", 0.1, "h1", true),
            ep("near-unlicensed", 0.2, "g1", false),
            ep("mid", 0.5, "g1", true),
            ep("far", 2.0, "g1", true),
        ];
        let r = retrieve(&corpus, &[0.0], 2, |e| {
            e.embodiment == "g1" && e.training_permitted
        });
        let ids: Vec<_> = r.selected.iter().map(|(i, _)| i.as_str()).collect();
        assert_eq!(ids, ["mid", "far"]);
        assert_eq!(r.refused, 2);
    }

    #[test]
    fn a_descriptor_of_the_wrong_width_is_refused() {
        let mut e = ep("x", 0.0, "g1", true);
        e.descriptor.push(1.0);
        assert_eq!(retrieve(&[e], &[0.0], 1, |_| true).refused, 1);
    }
}
