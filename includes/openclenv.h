#ifndef OPENCLENV_H
#define OPENCLENV_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

struct openclenv
{
    int              device_index;
    cl_device_type   device_type;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        kernel;
};


typedef enum {
    log_message, log_warning, log_err
} log_level;

#ifndef STABLE_MIN_LOG
#define STABLE_MIN_LOG 0
#endif

int opencl_initenv(struct openclenv* env, const char* bitcode_path, const char* kernname);
int opencl_teardown(struct openclenv* env);
const char* opencl_strerr(cl_int err);
void stablecl_log(log_level level, const char* string, ...);

#endif

