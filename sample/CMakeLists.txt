set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(symqg_indexing symqg_indexing.cpp)
target_link_libraries(symqg_indexing stdc++fs)
add_executable(symqg_querying symqg_querying.cpp)
target_link_libraries(symqg_querying stdc++fs)

add_executable(ivf_rabitq_indexing ivf_rabitq_indexing.cpp)
target_link_libraries(ivf_rabitq_indexing stdc++fs)
add_executable(ivf_rabitq_querying ivf_rabitq_querying.cpp)
target_link_libraries(ivf_rabitq_querying stdc++fs)

add_executable(hnsw_rabitq_indexing hnsw_rabitq_indexing.cpp)
target_link_libraries(hnsw_rabitq_indexing stdc++fs)
add_executable(hnsw_rabitq_querying hnsw_rabitq_querying.cpp)
target_link_libraries(hnsw_rabitq_querying stdc++fs)
