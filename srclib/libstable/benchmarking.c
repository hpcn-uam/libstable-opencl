#include "benchmarking.h"

#include <sys/time.h>
#include <stdlib.h>

double get_ms_time()
{
	struct timeval t;
    gettimeofday(&t, NULL);

    return (double) t.tv_sec * 1000 + (double) t.tv_usec / 1000;
}

