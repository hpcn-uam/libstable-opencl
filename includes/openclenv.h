#ifndef OPENCLENV_H
#define OPENCLENV_H

#include <CL/opencl.h>

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

int opencl_initenv(struct openclenv* env, const char* bitcode_path, const char* kernname);
int opencl_teardown(struct openclenv* env);
const char* opencl_strerr(cl_int err);


#endif

