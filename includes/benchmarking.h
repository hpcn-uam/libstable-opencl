#ifndef BENCHMARKING_H
#define BENCHMARKING_H

#ifdef BENCHMARK

#define BENCHMARK_BEGIN bc_start = get_ms_time()
#define BENCHMARK_END(iters, name) do { \
	bc_end = get_ms_time(); \
	fprintf(stderr, "%s: %f ms total, %f ms per item (%d items)\n", name, (bc_end - bc_start), (bc_end - bc_start) / iters, iters); \
} while(0)

#else

#define BENCHMARK_BEGIN 
#define BENCHMARK_END(iters, name) 

#endif
double get_ms_time();

#endif
