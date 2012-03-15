#include <stdio.h>
#include <OpenCL/OpenCL.h>
#include <assert.h>

//#include "fann/fann.h"
#include "floatfann.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TRUE 1
#define FALSE 0
#define SIZE_ARGS 10
#define PARAM_ARGS 1

#define NDEBUG 1

//Use for stringizing defined vars
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

void clEnqueueNDRangeKernel_err(cl_int err);
void clEnqueueReadBuffer_err(cl_int err, int line);
void clGetEventProfilingInfo_err(cl_int err, cl_event ev);
void clFinish_err(cl_int err);
void clSetCommandQueueProperty_err(cl_int err);
void clSetKernelArg_err(cl_int err, int line);

cl_device_id get_device(cl_context* cont);

void make_working_mem(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                      unsigned int *sizes, fann_type *input,
                      cl_mem *inputs_cl, cl_mem *sums_cl, cl_mem *outputs_cl);

void make_training_mem(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                       unsigned int *sizes, fann_type *output, unsigned int grp_size,
                       cl_mem *train_errors_cl, cl_mem *actual_outputs_cl, cl_mem *MSE_values_cl,
                       cl_mem *num_bit_fail_cl, cl_mem *weights_deltas_cl);
cl_platform_id GetIntelOCLPlatform();
#ifdef __cplusplus
}
#endif
