/*
 * Copyright (C) 2015 - Naudit High Performance Computing and Networking
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
 */

#include "benchmarking.h"

#include <sys/time.h>
#include <stdlib.h>

double get_ms_time()
{
	struct timeval t;
    gettimeofday(&t, NULL);

    return (double) t.tv_sec * 1000 + (double) t.tv_usec / 1000;
}

void benchmark_begin(double *bc)
{
	*bc = get_ms_time();
}

void benchmark_end(double *bc)
{
	double now = get_ms_time();
	*bc = now - *bc;
}

