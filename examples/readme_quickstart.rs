use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType, SearchParams};
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 128;

    // Generate 10,000 random vectors
    let dataset: Vec<Vec<f32>> = (0..10_000)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Train index with 256 clusters, 7-bit quantization
    let index = IvfRabitqIndex::train(
        &dataset,
        256, // nlist (number of clusters)
        7,   // total_bits (1 sign + 6 magnitude)
        Metric::L2,
        RotatorType::FhtKacRotator, // Fast Hadamard Transform
        42,                         // random seed
        false,                      // use_faster_config
    )?;

    // Search: probe 32 clusters, return top 10 neighbors
    let params = SearchParams::new(10, 32);
    let results = index.search(&dataset[0], params)?;

    println!(
        "Top neighbor ID: {}, distance: {}",
        results[0].id, results[0].score
    );
    Ok(())
}
