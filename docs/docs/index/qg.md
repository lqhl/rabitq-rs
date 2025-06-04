# QG + RaBitQ (SymphonyQG)

[QG](https://medium.com/@masajiro.iwasaki/fusion-of-graph-based-indexing-and-product-quantization-for-ann-search-7d1f0336d0d0) is a graph-based index originated from the [NGT library](https://github.com/yahoojapan/NGT). Different from HNSW, it creates multiple quantization codes for every vector and carefully re-organizes their layout to minimize random memory accesses in querying. RaBitQ + QG in developped from our research project [SymphonyQG](https://dl.acm.org/doi/10.1145/3709730). Unlike IVF + RaBitQ and HNSW + RaBitQ, which consumes less memory than the raw datasets, RaBitQ + QG consumes more memory to pursue the best time-accuracy trade-off.
Here, we offer a toy example for the indexing and querying of QG.
To test QG on real-world datasets, please refer to `sample/symqg_indexing.cpp` and `sample/symqg_querying.cpp` for
detailed information

## Index Construction

We build the QG by iteratively refining the graph structure.
Since the QG is more complicated than other indices, we need a QGBuilder to help us construct the index.

At the beginning, we need to intialize a QG and a QGBuilder by following construtor.
```cpp
QuantizedGraph::QuantizedGraph(
        size_t num,
        size_t dim,
        size_t max_deg,
        RotatorType type = RotatorType::FhtKacRotator
    );

QGBuilder::QGBuilder(
        QuantizedGraph<float>& index,
        uint32_t ef_build,
        const float* data,
        size_t num_threads = std::numeric_limits<size_t>::max()
    )
```
- **num**: Number of vertices (vectors) in the dataset.  
- **dim**: Dimension of the dataset.  
- **max_deg**: Degree bound of QG, must be a multiple of 32.  
- **index**: Previously initialized QG.  
- **ef_build**: Search window size during indexing.  
- **data**: Pointer to the dataset, size of num * dim.  
- **num_threads**: Number of threads to use (default: std::numeric_limits<size_t>::max(), which auto-selects).  
```cpp
size_t rows = 1000000;
size_t cols = 128;
size_t degree = 32;
size_t ef = 200

float* data = new float[rows * cols]; // only for illustration

QuantizedGraph qg(rows, cols, degree); // init qg

QGBuilder builder(qg, ef, data.data()); // init builder
```

Then, we can use the builder to construct the index. Then we can save the index.
```cpp
builder.build();    // build index interatively

const char* index_file = "./qg_example.index"
qg.save(index_file);    // save index
```

### Data Layout

Each indexed element is stored in the following layout.
```
[Raw data vector]
[Batch data for QG]
[Edges]
```

## Querying

For querying, code is pretty simple.
```cpp
void QuantizedGraph::search(
    const T* __restrict__ query, 
    uint32_t k, 
    uint32_t* __restrict__ results);
```
- **query**: Query vector.  
- **k**: Top-k.  
- **results**: Result buffer, size of k.  
Then we can use a pre-constructed index to search.
```cpp
QuantizedGraph<float> qg;
qg.load("./qg_example.index"); // load pre-constructed index


size_t ef = 100;
size_t topk = 10;
std::vector<PID> results(topk); // result buffer
float* query = new float[cols]; // query vector (only for illustration)

qg.set_ef(ef);  // set search window size
qg.search(query, topk, results.data()); // search knn, result will be stored in results
```