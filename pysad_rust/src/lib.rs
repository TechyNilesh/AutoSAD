use pyo3::prelude::*;

mod models;
mod utils;

use models::hst::HalfSpaceTrees;
use models::loda::LODA;
use models::oif::OnlineIsolationForest;
use models::rrcf::RobustRandomCutForest;
use models::iforest_asd::IForestASD;
use models::xstream::XStream;
use models::rshash::RSHash;

use utils::evaluation::Evaluator;
use utils::ss::StreamStatistic;

/// A Python module implemented in Rust.
#[pymodule]
fn pysad_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // LODA:
    m.add_class::<LODA>()?;
    // OIF:
    m.add_class::<OnlineIsolationForest>()?;

    // HST:
    m.add_class::<HalfSpaceTrees>()?;

    // RRCF:
    m.add_class::<RobustRandomCutForest>()?; 

    // IForestASD:
    m.add_class::<IForestASD>()?;

    // XStream:
    m.add_class::<XStream>()?;

    // RSHash:
    m.add_class::<RSHash>()?;

    // Evaluator:
    m.add_class::<Evaluator>()?;
    // StreamStatistic:
    m.add_class::<StreamStatistic>()?;
    Ok(())
}
