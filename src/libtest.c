/*
 * Copyright (C) 2017 - Naudit High Performance Computing and Networking
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

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include "stable_api.h"
#include "benchmarking.h"
#include "opencl_integ.h"
#include "kde.h"

#define MAX_SAMPLES 10000

static const char* _last_header = NULL;
static size_t _tests_passed = 0;
static size_t _tests_failed = 0;

static void test_begin(const char* header)
{
	printf("\x1b[1mTesting %s... \x1b[0m\n", header);
	_last_header = header;
}

static void test_ok()
{
	printf("\x1b[1mTest %s \x1b[0m\x1b[32mOK\x1b[0m\n", _last_header);
	_tests_passed++;
}

static void test_fail(const char* reason)
{
	printf("\x1b[1mTest %s \x1b[0m\x1b[31mFAIL\x1b[0m (%s)\n", _last_header, reason);
	_tests_failed++;
}

static void kolmogorov_conv_test()
{
	StableDist* dist;
	double samples[MAX_SAMPLES];
	double ks_test;

	dist = stable_create(1.5, 0.4, 5, 1, 0);

	test_begin("Kolmogorov-Smirnov distance convergence");

	for (size_t i = 1000; i < MAX_SAMPLES; i += 1000) {
		stable_rnd(dist, samples, i);
		ks_test = stable_kolmogorov_smirnov_gof(dist, samples, i, NULL);
		printf("At size %zu KS test statistic is %lf\n", i, ks_test);

		if (ks_test < 0.05) {
			test_fail("KS statistic is too low");
			return;
		}
	}

	test_ok();
}

int main(int argc, char** argv)
{
	kolmogorov_conv_test();
}
