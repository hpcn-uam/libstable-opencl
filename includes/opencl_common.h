#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

#define GK_USE_127_POINTS

#ifdef GK_USE_127_POINTS
#define GK_POINTS 127
#endif

#ifdef GK_USE_61_POINTS
#define GK_POINTS 61
#endif

#define GK_SUBDIVISIONS 20

#define PDF_ALPHA_EQ1 1
#define PDF_ALPHA_NEQ1 2
#define GPU_TEST_INTEGRAND 100
#define GPU_TEST_INTEGRAND_SIMPLE 101

#if defined(FLOAT_GPU_UNIT) || (defined(__OPENCL_VERSION__) && !defined(cl_khr_fp64) && !defined(cl_amd_fp64))
#define cl_precision float
#define cl_precision2 float2
#define cl_precision_type "float"
#else
#define cl_precision double
#define cl_precision2 double2
#define cl_precision_type "double"
#endif


struct stable_info {
    cl_precision theta;
    cl_precision beta_;
    cl_precision k1;
    cl_precision xxipow;
    cl_precision ibegin;
    cl_precision iend;
    cl_precision subinterval_length;
    cl_precision half_subint_length;
    cl_precision theta0_;
    cl_precision alfa;
    cl_precision alfainvalfa1;
    unsigned int threads_per_interval;
    unsigned int gauss_points;
    unsigned int kronrod_points;
    unsigned int integrand;
};

#endif
