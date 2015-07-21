#ifndef OPENCL_COMMON_H
#define OPENCL_COMMON_H

#define GK_USE_127_POINTS

#ifdef GK_USE_127_POINTS
#define GK_POINTS 127
#endif

#ifdef GK_USE_61_POINTS
#define GK_POINTS 61
#endif

#define POINTS_EVAL 2

#ifdef AMD_GPU
#define MAX_WORKGROUPS 4
#else
#define MAX_WORKGROUPS 8
#endif

#define GK_SUBDIVISIONS (POINTS_EVAL * MAX_WORKGROUPS)
#define KRONROD_EVAL_POINTS (GK_POINTS / 2 + 1)

// Just byte markers: XYZ where
//      Z == 1 if is an evaluation on α = 1 (0 if α ≠ 1),
//      Y == 1 if PDF evaluation
//      X == 1 if CDF evaluation
#define PDF_ALPHA_NEQ1 2
#define PDF_ALPHA_EQ1 3
#define CDF_ALPHA_NEQ1 4
#define CDF_ALPHA_EQ1 5
#define PCDF_ALPHA_NEQ1 6
#define PCDF_ALPHA_EQ1  7
#define GPU_TEST_INTEGRAND 100
#define GPU_TEST_INTEGRAND_SIMPLE 101

#define MODEMARKER_PDF 2
#define MODEMARKER_CDF 4
#define MODEMARKER_PCDF (2 | 4)
#define MODEMARKER_EQ1 1
#define MODEMARKER_NEQ1 0

#define is_integrand_pdf(integrand) (integrand & MODEMARKER_PDF)
#define is_integrand_cdf(integrand) (integrand & MODEMARKER_CDF)
#define is_integrand_eq1(integrand) (integrand & MODEMARKER_EQ1)
#define is_integrand_neq1(integrand) (!is_integrand_eq1(integrand))
#define is_integrand_pcdf(integrand) (is_integrand_cdf(integrand) && is_integrand_pdf(integrand))

#if defined(FLOAT_GPU_UNIT) || (defined(__OPENCL_VERSION__) && !defined(cl_khr_fp64) && !defined(cl_amd_fp64))
#define cl_precision float
#define cl_precision2 float2
#define cl_precision4 float4
#define cl_precision8 float8
#define cl_precision_type "float"
#define CL_PRECISION_IS_FLOAT
#else
#define cl_precision double
#define cl_precision2 double2
#define cl_precision4 double4
#define cl_precision8 double8
#define cl_precision_type "double"
#endif


struct stable_info {
    cl_precision k1;
    cl_precision theta0;
    cl_precision alfa;
    cl_precision alfainvalfa1;
    cl_precision mu_0;
    cl_precision sigma;
    cl_precision xi;
    cl_precision xxi_th;
    cl_precision c2_part;
    cl_precision c1;
    cl_precision THETA_TH;
    cl_precision beta;
    cl_precision xi_coef;
    short is_xxi_negative;
    unsigned int integrand;
    cl_precision final_pdf_factor;
    cl_precision final_cdf_factor;
    cl_precision final_cdf_addition;
    cl_precision quantile_tolerance;
    size_t max_reevaluations;
};

#endif
