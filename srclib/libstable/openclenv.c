#include "openclenv.h"
#include <stdarg.h>

#define OPENCL_BUILD_OPTIONS "-cl-no-signed-zeros"

#define MAX_OPENCL_PLATFORMS 5
#define MAX_BUILD_OPTS_LENGTH 1000

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

char *_read_file(const char *filename, size_t *contents_len)
{
    FILE *f = fopen(filename, "r");
    char *contents = NULL;
    size_t read;

    if (!f)
        goto error;

    if (fseek(f, 0, SEEK_END))
        goto error;

    *contents_len = ftell(f);

    if (fseek(f, 0, SEEK_SET))
        goto error;

    contents = calloc(*contents_len + 1, sizeof(char));

    if (!contents)
        goto error;

    read = fread(contents, *contents_len, 1, f);

    if (read < 1)
        goto error;

    fclose(f);
    contents[*contents_len] = 0;
    *contents_len += 1;

    return contents;

error:
    if (contents) free(contents);
    if (f) fclose(f);

    return NULL;
}

static void _opencl_kernel_info(struct openclenv* env, cl_kernel kernel)
{
#if STABLE_MIN_LOG <= 0
    int j;
    cl_int err;
    size_t wg_sizes[3], max_dims;
    cl_ulong local_memsize, priv_memsize;

    err = clGetKernelWorkGroupInfo(kernel, env->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg_sizes), wg_sizes, &max_dims);
    err |= clGetKernelWorkGroupInfo(kernel, env->device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_memsize, NULL);
    err |= clGetKernelWorkGroupInfo(kernel, env->device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &priv_memsize, NULL);

    local_memsize /= 1024;

    max_dims /= sizeof(size_t);
    if(err)
    {
        stablecl_log(log_err, "Unable to get kernel info: %s (%d)", opencl_strerr(err), err);
        return;
    }

    stablecl_log(log_message, "Max dimensions for kernel: %zu.", max_dims);

    for (j = 0; j < max_dims; j++)
        stablecl_log(log_message, "Max kernel workgroup size for dimension %zu: %zu", j, wg_sizes[j]);

    stablecl_log(log_message, "Kernel memory: %zu kB local, %zu B private", local_memsize, priv_memsize);
#endif
}


static void _opencl_device_info(cl_device_id device)
{
#if STABLE_MIN_LOG <= 0
    char dev_name[128];
    int j;
    size_t wg_sizes[5], max_dims;
    cl_ulong global_memsize, local_memsize, constant_memsize;

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128 * sizeof(char), dev_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof wg_sizes, wg_sizes, &max_dims);

    stablecl_log(log_message, "Device name: %s", dev_name);

    max_dims /= sizeof(size_t);
    stablecl_log(log_message, "Max dimensions: %zu.", max_dims);

    for (j = 0; j < max_dims; j++)
        stablecl_log(log_message, "Max workgroup size for dimension %zu: %zu", j, wg_sizes[j]);

    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_memsize, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_memsize, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &constant_memsize, NULL);

    stablecl_log(log_message, "Memory: global %zu kB, local %zu kB, constant buffer %zu kB",
        global_memsize / 1024, local_memsize / 1024, constant_memsize / 1024);
#endif
}

static void _opencl_platform_info(cl_platform_id *platforms, cl_uint platform_num)
{
#if STABLE_MIN_LOG <= 0
    int i;
    char version[500], name[500], vendor[500], extensions[500];
    cl_uint float_vecwidth = 0, double_vecwidth = 0;

    stablecl_log(log_message, "Available platforms: %d", platform_num);
    for (i = 0; i < platform_num; i++)
    {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 500, name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 500, version, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 500, vendor, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 500, extensions, NULL);
        clGetPlatformInfo(platforms[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &double_vecwidth, NULL);
        clGetPlatformInfo(platforms[i], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &float_vecwidth, NULL);

        stablecl_log(log_message, "%d: %s, OpenCL version %s, vendor %s. Available extensions: %s. Vector widths: double %zu, float %zu.",
         i, name, version, vendor, extensions, double_vecwidth, float_vecwidth);

    }
#endif
}

static void _opencl_devices_info(cl_device_id* devices, cl_uint device_num)
{
#if STABLE_MIN_LOG <= 0
    int i;

    stablecl_log(log_message, "Available devices: %d", device_num);
    for (i = 0; i < device_num; i++)
    {
        stablecl_log(log_message, "Information for device %d", i);
        _opencl_device_info(devices[i]);
    }
#endif
}

int opencl_initenv(struct openclenv *env, size_t platform_index)
{
    char *err_msg = NULL;
    int err = 0;
    cl_platform_id platforms[MAX_OPENCL_PLATFORMS];
    cl_device_id devices[MAX_OPENCL_PLATFORMS];
    cl_uint platform_num, device_num;
    size_t device_index = platform_index;

    err = clGetPlatformIDs(MAX_OPENCL_PLATFORMS, platforms, &platform_num);

    if (err)
    {
        err_msg = "clGetPlatformIDs";
        goto error;
    }

    _opencl_platform_info(platforms, platform_num);


    err = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_ALL, MAX_OPENCL_PLATFORMS, devices, &device_num);

    if (err)
    {
        err_msg = "clGetDeviceIDs - group creation";
        goto error;
    }

    env->device = devices[device_index];

    _opencl_devices_info(devices, device_num);

    stablecl_log(log_message, "Chosen device %zu from platform %zu", device_index, platform_index);

    env->context = clCreateContext(0, 1, &env->device, NULL, NULL, &err);
    if (!env->context)
    {
        err_msg = "clCreateContext";
        goto error;
    }

    env->queues = NULL;
    env->queue_count = 0;

    err = opencl_set_queues(env, 1);

    if(err)
    {
    	err_msg = "opencl_set_queues";
    	goto error;
    }

    opencl_set_current_queue(env, 0);
    memset(env->enabled_kernels, 0, sizeof(env->enabled_kernels));
    env->current_kernel = 0;
    env->kernel_count = 0;

error:
    if (err && err_msg)
        stablecl_log(log_err, "Init failed with error %d at %s: %s", err, err_msg, opencl_strerr(err));

    return err;
}

static void _opencl_generate_build_opts(char* build_opts, size_t build_opts_len)
{
#ifdef AMD_GPU
    // AMD's OpenCL compiler doesn't know to search for includes in the current working
    // directory, so we have to set that option explicitly.

    char cwd[300];

    if(getcwd(cwd, sizeof(cwd)) != NULL)
        snprintf(build_opts, build_opts_len, "%s -I%s/", OPENCL_BUILD_OPTIONS, cwd);
    else
    {
        stablecl_log(log_warning, "warning: getcwd failed with error %d: kernel compilation will probably fail", errno);
        snprintf(build_opts, build_opts_len, OPENCL_BUILD_OPTIONS);
    }
#else
    snprintf(build_opts, build_opts_len, OPENCL_BUILD_OPTIONS);
#endif
}

short opencl_load_kernel(struct openclenv* env, const char *bitcode_path, const char *kernname, size_t index)
{
    char *err_msg = NULL;
    int err = 0, log_error;
    size_t pathlen = strlen(bitcode_path);
    char *build_log;
    size_t build_log_size;
    char build_opts[MAX_BUILD_OPTS_LENGTH];


#ifdef __APPLE__
    env->program = clCreateProgramWithSource(env->context, 1, (const char **)&bitcode_path, &pathlen, &err);
#else
    size_t code_length;
    char *code_contents = _read_file(bitcode_path, &code_length);

    if (!code_contents)
    {
        err_msg = strerror(errno);
        err = errno;
        goto error;
    }

    env->program = clCreateProgramWithSource(env->context, 1 , (const char **)&code_contents, &code_length, &err);

    free(code_contents);
#endif


    if (err)
    {
        err_msg = "clCreateProgramWithSource";
        goto error;
    }

    _opencl_generate_build_opts(build_opts, MAX_BUILD_OPTS_LENGTH);

    stablecl_log(log_message, "Building kernel %s from %s...", kernname, bitcode_path);
    stablecl_log(log_message, "Build options: %s", build_opts);

    err = clBuildProgram(env->program, 1, &env->device, build_opts, NULL, NULL);

    log_error = clGetProgramBuildInfo(env->program, env->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);

    if (log_error)
    {
        stablecl_log(log_err, "Error retrieving build log size.");
    }
    else
    {
        build_log = malloc(build_log_size);

        if (!build_log)
        {
            stablecl_log(log_err, "Couldn't allocate enough memory for build log.");
        }
        else
        {
            log_error = clGetProgramBuildInfo(env->program, env->device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, &build_log_size);

            if (log_error)
                stablecl_log(log_err, "Couldn't get build log: %s", opencl_strerr(log_err));
            else if (err)
                stablecl_log(log_err, "Build log (size %zu):\n%s", build_log_size, build_log);

            free(build_log);
        }
    }

    if (err)
    {
        err_msg = "clBuildProgram";
        goto error;
    }

    env->kernel[index] = clCreateKernel(env->program, kernname, &err);

    if (err)
    {
        err_msg = "clCreateKernel";
        goto error;
    }

    _opencl_kernel_info(env, env->kernel[index]);

    env->enabled_kernels[index] = 1;
    env->kernel_count++;

    stablecl_log(log_message, "Kernel %s loaded successfully", kernname);

error:
    if (err && err_msg)
        stablecl_log(log_err, "Kernel load failed with error %d at %s: %s", err, err_msg, opencl_strerr(err));

    return err;
}

short opencl_set_current_queue(struct openclenv* env, size_t queue)
{
	if(queue >= env->queue_count)
		return -1;

	env->current_queue = queue;
	return 0;
}

inline cl_command_queue opencl_get_queue(struct openclenv* env)
{
	return env->queues[env->current_queue];
}

cl_kernel opencl_get_current_kernel(struct openclenv* env)
{
    return env->kernel[env->current_kernel];
}

short opencl_set_queues(struct openclenv* env, size_t new_count)
{
	cl_int err;

    if(env->queue_count == new_count)
    	return 0;

    if(env->queue_count > new_count)
    	return opencl_remove_last_n_queues(env, env->queue_count - new_count);

    env->queues = realloc(env->queues, new_count * sizeof(cl_command_queue));

    if(!env->queues)
    	return -1;

    bzero(env->queues + env->queue_count, (new_count - env->queue_count) * sizeof(cl_command_queue));

    for(size_t i = env->queue_count; i < new_count; i++)
    {
    	env->queues[i] = clCreateCommandQueue(env->context, env->device, CL_QUEUE_PROFILING_ENABLE, &err);

	    if (!env->queues[i])
	    	return err;
	}

	env->queue_count = new_count;

	return 0;
}

short opencl_remove_last_n_queues(struct openclenv* env, size_t n)
{
    int last_deleted_queue = env->queue_count - n;

    if(last_deleted_queue < 0)
        last_deleted_queue = 0;

    for(int i = env->queue_count - 1; i >= last_deleted_queue; i--)
    	if(env->queues[i])
    		clReleaseCommandQueue(env->queues[i]);

    return 0;
}

int opencl_teardown(struct openclenv *env)
{
    size_t i;

    if (env->program)
        clReleaseProgram(env->program);

    for(i = 0; i < MAX_KERNELS; i++)
        if(env->enabled_kernels[i])
            clReleaseKernel(env->kernel[i]);

    if(env->queues)
	    opencl_remove_last_n_queues(env, env->queue_count);

    if (env->context)
        clReleaseContext(env->context);

    return 0;
}

const char *opencl_strerr(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    default: return "Unknown";
    }
}

void stablecl_log(log_level level, const char *string, ...)
{
    va_list ap;

    if (level < STABLE_MIN_LOG)
        return;

    va_start(ap, string);
    fprintf(stderr, "[Stable-OpenCL] ");
    vfprintf(stderr, string, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

int stablecl_finish_all(struct openclenv *env)
{
	cl_int err = 0;

	for(size_t i = 0; i < env->queue_count; i++)
    	err |= clFinish(env->queues[i]);

    return err;
}

void stablecl_profileinfo(struct opencl_profile *prof, cl_event event)
{
    int retval = 0;

    retval |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                                      sizeof(cl_ulong), &prof->queued, NULL);
    retval |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,
                                      sizeof(cl_ulong), &prof->submitted, NULL);
    retval |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &prof->started, NULL);
    retval |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &prof->finished, NULL);

    if (retval != CL_SUCCESS)
        fprintf(stderr, "clGetEventProfilingInfo error %d: %s", retval, opencl_strerr(retval));

    prof->submit_acum = (double)(prof->submitted - prof->queued) / 1000000;
    prof->start_acum = (double)(prof->started - prof->queued) / 1000000;
    prof->finish_acum = (double)(prof->finished - prof->queued) / 1000000;
    prof->exec_time = (double)(prof->finished - prof->started) / 1000000;
}

void opencl_set_current_kernel(struct openclenv* env, size_t kern_index)
{
    env->current_kernel = kern_index;
}


