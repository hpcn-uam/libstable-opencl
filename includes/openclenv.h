#ifndef OPENCLENV_H
#define OPENCLENV_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define MAX_KERNELS 5

struct openclenv
{
	int device_index;
	cl_device_type device_type;
	cl_device_id device;
	cl_context context;
	cl_command_queue* queues;
	size_t queue_count;
	size_t current_queue;
	cl_program program;
	cl_kernel kernel[MAX_KERNELS];
	short enabled_kernels[MAX_KERNELS];
	size_t kernel_count;
	size_t current_kernel;
};

struct opencl_profile
{
	cl_ulong queued;
	cl_ulong submitted;
	cl_ulong started;
	cl_ulong finished;
	double submit_acum;
	double start_acum;
	double finish_acum;
	double exec_time;
	double argset;
	double enqueue;
	double buffer_read;
	double set_results;
	double total;
	double profile_total;
};


typedef enum {
	log_message, log_warning, log_err
} log_level;

#ifndef STABLE_MIN_LOG
#define STABLE_MIN_LOG 0
#endif

int opencl_initenv(struct openclenv* env);
short opencl_load_kernel(struct openclenv* env, const char* bitcode_path, const char* kernname, size_t index);
short opencl_set_current_queue(struct openclenv* env, size_t queue);
short opencl_set_queues(struct openclenv* env, size_t new_count);
short opencl_remove_last_n_queues(struct openclenv* env, size_t n);
cl_command_queue opencl_get_queue(struct openclenv* env);
void opencl_set_current_kernel(struct openclenv* env, size_t kern_index);
cl_kernel opencl_get_current_kernel(struct openclenv* env);
int opencl_teardown(struct openclenv* env);
const char* opencl_strerr(cl_int err);
void stablecl_log(log_level level, const char* string, ...);
void stablecl_profileinfo(struct opencl_profile* prof, cl_event event);
int stablecl_finish_all(struct openclenv* env);
#endif

