/// A common Rust trait for streaming‐anomaly models.
pub trait BaseModel {
    /// Incorporate one instance.
    fn fit_partial(&mut self, x: &[f64]);

    /// Score one instance.
    fn score_partial(&mut self, x: &[f64]) -> f64;

    /// Default: fit then score.
    fn fit_score_partial(&mut self, x: &[f64]) -> f64 {
        self.fit_partial(x);
        self.score_partial(x)
    }

    /// Default: fit a batch of instances.
    fn fit(&mut self, xs: &[Vec<f64>]) {
        for x in xs {
            self.fit_partial(x);
        }
    }

    /// Default: score a batch.
    fn score(&mut self, xs: &[Vec<f64>]) -> Vec<f64> {
        xs.iter().map(|x| self.score_partial(x)).collect()
    }

    /// Default: fit+score a batch.
    fn fit_score(&mut self, xs: &[Vec<f64>]) -> Vec<f64> {
        xs.iter().map(|x| self.fit_score_partial(x)).collect()
    }
}
