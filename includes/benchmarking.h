#ifndef BENCHMARKING_H
#define BENCHMARKING_H

#define bench_begin(value, enabler) if(enabler) benchmark_begin(&value);
#define bench_end(value, enabler) if(enabler) benchmark_end(&value);

double get_ms_time();
void benchmark_begin(double *bc);
void benchmark_end(double *bc);

#endif
