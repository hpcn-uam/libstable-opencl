# Libstable-opencl

Libstable-opencl is a fork of the _libstable_ library to offload computations to OpenCL platforms (GPU and more).

Original _libstable_ library by Javier Royuela del Val and Federico Simmross Wattenberg, available in http://www.lpi.tel.uva.es/stable.

Version: 1.0.3

## Library usage

If you want to use the library along with your program, you should link it using the -lstable flag in the compiler. Ensure that the include and library paths are set correctly too, so your compiler can see the headers in the _include_ directory of libstable and the libraries in the _lib/release_ directory (although you may want to link agains debug builds using the _lib/debug_ path).

In your code, you can use the functions `stable_pdf_gpu, stable_cdf_gpu, stable_inv_gpu, stable_rnd_gpu` and `stable_fit_grid` to do calculations related with stable distributions (the last function is present in the _stable_gridfit.h_ header). Remember to activate the GPU before using these functions calling to `stable_activate_gpu`. You can also select the platform where you want the OpenCL code to run changing the `gpu_platform` variable in the `StableDist` struct before calling the GPU activation (you can see the available platforms in your GPU and their corresponding numbers running _bin/debug/gpu_tests_).

## Compilation

The compilation of libstable requires a C compiler (either GCC or Clang are compatible). The code has the following requirements:

* [GNU Scientific Library](http://www.gnu.org/software/gsl/), both the library and header files. In Debian/Ubuntu, the package to install is named _libgsl0-dev_, _gsl-devel_ in Fedora and OS X (MacPorts).
* OpenCL. Both libraries and headers are required. For the software to recognize the accelerator device (either GPU or CPU), the correspondent OpenCL drivers should be installed. You should ensure that the library paths are present in your platform's Makefile.

Once all the requeriments are fulfilled, running `make` will compile the library and the sample programs. `make libs` will only build the library files. The programs will be saved in the _bin_ directory and the libraries in _lib_. Those directories contain different subdirectories corresponding to the different build configurations supported by our Makefile:

* _debug_: Targeted for debugging the program. Includes debugging symbols, shows more log messages to follow the execution flow and disables compiler optimizations.
* _release_: Configuration targeted for production releases. Compiled with full optimization and with _-march=native_ to improve performance.

These two are the main configurations. There are others present just for support that are useful during development:

* _benchmark_: Activates macros in the code that will benchmark the running time of the code.
* _profile_: Compiles a version of the software ready for profiling with [gprof](https://sourceware.org/binutils/docs/gprof) or [Valgrind](http://valgrind.org/): it uses compiler optimizations and also includes debugging symbols.
* _simulator_: A build of the software targeted for the [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) GPU simulator: uses the same flags as the _debug_ configuration and also disables some incompatible OpenCL calls.

## Test programs

Apart from the library, there are some test programs located in the _src_ directory:

* _gpu_tests_: A sample to use the GPU in the PDF and CDF, comparing results with the CPU.
* _gpu_performance_: Tests the GPU performance in the whole parameter space, printing detailed OpenCL profile information.
* _gpu_mpoints_perftest_: Outputs the performance results of the library depending on the number of simultaneous points. The outputs are a series of rows containing the following data: _point-count gpu-duration gpu-time-per-point cpu-duration cpu-time-per-point cpu-multithreaded-duration cpu-multithreaded-time-per-point_.
* _gpu_precision_: Outputs a comparison of the precision of the GPU against the CPU in different intervals.
* _fit_eval_: Tests the different parameter estimators in the library and outputs a summary of the results.
* _fitperf_: Outputs a summary of the performance of the different estimators.
* _quantile_perf_: Shows the performance of the quantile function depending on the parameters.
* _quantile_eval_: Evaluates the accuracy of the quantile function.

## Possible bugs / failures

* In some instances, the GPU seems to "hang" after running repeatedly the OpenCL kernels. The solution found was to reboot the system. It seems to be a bug with NVIDIA drivers.
* Your GPU may not support the default workgroup sizes, printing an error about insufficient resources. You may try modifying in _includes/opencl_common.h_ the max workgroup sizes (`MAX_WORKGROUPS`) or reducing the number of Gauss-Kronrod points to 61 defining the `GK_USE_61_POINTS` (not recommended if you want to maintain precision).
* Your GPU may not support double-precision numbers (seen on Intel Iris HD graphic cards). Our software detects it if it's the case, issuing a warning during the compilation of the OpenCL kernel (you may not see it if you're not running debug builds). However, the precision is severly affected and you may see weird results if your card doesn't support double-precision floating point numbers.
* Some NVIDIA drivers cache the OpenCL kernels but don't rebuild correctly when included headers changes. If you change a header included in the OpenCL code (currently only _opencl_common.h, gk_points.h_ and _stable_inv_precalcs.h) remove the _$HOME/.nv_ folder to force a rebuild.
* Some AMD drivers fail when including header files in OpenCL code. An easy workaround is to copy the included headers to the _/tmp_ folder, where AMD does the temporary copies when compiling code. Running _cp -r includes /tmp_ should suffice to work around this driver bug.

## Changelog

- v1.0: Initial version
- v1.0.1: Add random number generation capabilities.
- v1.0.2: Improve efficienty of the random number generator
- v1.0.3: Support situations when the GPU does not have enough memory for a single kernel call.
