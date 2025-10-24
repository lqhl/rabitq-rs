use rabitq_rs::mstg::{MstgConfig, MstgIndex, SearchParams};
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;

    // Generate 10,000 random vectors
    let dataset: Vec<Vec<f32>> = (0..10_000)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Build index with default configuration
    let index = MstgIndex::build(&dataset, MstgConfig::default())?;

    // Search: balanced mode with top 10 results
    let params = SearchParams::balanced(10);
    let results = index.search(&dataset[0], &params);

    println!(
        "Top neighbor ID: {}, distance: {:.6}",
        results[0].vector_id, results[0].distance
    );
    Ok(())
}
