use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType};

fn main() {
    println!("Testing huge pages support...");

    // Create some simple test data
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![13.0, 14.0, 15.0, 16.0],
    ];

    // Train a small index - this should trigger huge pages logging
    let _index = IvfRabitqIndex::train(
        &data,
        2, // nlist
        7, // bits
        Metric::L2,
        RotatorType::FhtKacRotator,
        42,   // seed
        true, // use_faster_config
    )
    .expect("Failed to train index");

    println!("Index created successfully!");
}
