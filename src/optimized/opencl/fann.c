/*
  Fast Artificial Neural Network Library (fann)
  Copyright (C) 2010 Seth Price (seth@pricepages.org)

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "fann_cl.h"
#include "fann_cl_kernel.h"
#include "fann_cl_ann.h"
#include "fann_cl_run.h"
#include "fann_cl_train.h"

//Since the topography of the network may change during training, don't
//keep anything resident that's dimensions may change.
struct fann_resident_cl
{
    cl_command_queue cmd_queue;
    cl_context context;
    cl_kernel t_kern;
    cl_kernel r_kern;
    cl_device_id dev;
    
    cl_mem inputs_cl;
    
    cl_mem actual_outputs_cl;
    cl_mem MSE_values_cl;
    cl_mem num_bit_fail_cl;
};

struct fann_resident_cl* malloc_res()
{
    struct fann_resident_cl *res_cl;
    res_cl = (struct fann_resident_cl *)malloc(sizeof(struct fann_resident_cl));
    
    res_cl->cmd_queue = NULL;
    res_cl->context = NULL;
    res_cl->t_kern = NULL;
    res_cl->r_kern = NULL;
    res_cl->dev = NULL;
    
    res_cl->inputs_cl = NULL;
    res_cl->actual_outputs_cl = NULL;
    res_cl->MSE_values_cl = NULL;
    res_cl->num_bit_fail_cl = NULL;
    
    return res_cl;
}

void free_res(struct fann_resident_cl *res_cl)
{
    if (res_cl->inputs_cl != NULL)
        clReleaseMemObject(res_cl->inputs_cl);
    if (res_cl->actual_outputs_cl != NULL)
        clReleaseMemObject(res_cl->actual_outputs_cl);
    if (res_cl->MSE_values_cl != NULL)
        clReleaseMemObject(res_cl->MSE_values_cl);
    if (res_cl->num_bit_fail_cl != NULL)
        clReleaseMemObject(res_cl->num_bit_fail_cl);
    
    if (res_cl->t_kern != NULL)
        clReleaseKernel(res_cl->t_kern);
    if (res_cl->r_kern != NULL)
        clReleaseKernel(res_cl->r_kern);
    if (res_cl->cmd_queue != NULL)
        clReleaseCommandQueue(res_cl->cmd_queue);
    if (res_cl->context != NULL)
        clReleaseContext(res_cl->context);
    
    free(res_cl);
}

void opt_run_many(struct fann **anns, fann_type *input,
                  fann_type **output, int num_anns, int num_runs)
{
	cl_command_queue cmd_queue;
	cl_context context;
    cl_kernel kern;
    
	size_t group_size, shared_size;
    unsigned int sizes[SIZE_ARGS];
    float f_params[PARAM_ARGS];
	cl_int err = 0;
	
	cl_mem  sizes_cl, f_params_cl, num_layers_cl, num_neurons_cl, num_inputs_cl,
            num_outputs_cl, steepness_cl, act_cl, weights_cl;
    cl_mem  sums_cl, outputs_cl;
    cl_mem  inputs_cl = NULL;
    
    cl_device_id dev = get_device(&context);

    // Now create a context to perform our calculation with the specified device 
    //context = clCreateContext(0, 1, &dev, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    // And also a command queue for the context
    cmd_queue = clCreateCommandQueue(context, dev, 0, NULL);
	kern = get_kernel("run", context, dev);
    
    //What's the recommended group size?
	clGetKernelWorkGroupInfo(kern, dev, CL_KERNEL_WORK_GROUP_SIZE,
							 sizeof(size_t), &group_size, NULL);
    shared_size = group_size*sizeof(float);
#ifndef NDEBUG
	printf("Recommended Size: %lu\n", group_size);
#endif
    
    //Allocate & copy the ANN structure on the device
    make_anns(kern, anns, num_anns, num_runs, group_size, sizes, f_params, cmd_queue, context,
              &sizes_cl, &f_params_cl, &num_layers_cl, &num_neurons_cl,
              &num_inputs_cl, &num_outputs_cl,
              &steepness_cl, &act_cl, &weights_cl );
    
    //Allocate working device memory
    make_working_mem(cmd_queue, context, kern, sizes, input,
                     &inputs_cl, &sums_cl, &outputs_cl);
    
    //Set shared memeory
	err = clSetKernelArg(kern, 12, sizeof(float)*sizes[2], NULL);
    assert(err == CL_SUCCESS);
	err = clSetKernelArg(kern, 13, sizeof(int)*sizes[2], NULL);
    assert(err == CL_SUCCESS);
	err = clSetKernelArg(kern, 14, shared_size, NULL);
    assert(err == CL_SUCCESS);
    
    //Execute kernel
    execute_kern(cmd_queue, context, kern, sizes[5], group_size);
    
    //Copy output
    copy_run_output(cmd_queue, context, anns, sizes, outputs_cl, output);
    
    //Release resources
    clReleaseMemObject(sizes_cl);
    clReleaseMemObject(num_layers_cl);
    clReleaseMemObject(num_neurons_cl);
    clReleaseMemObject(num_inputs_cl);
    clReleaseMemObject(num_outputs_cl);
    
    clReleaseMemObject(steepness_cl);
    clReleaseMemObject(act_cl);
    clReleaseMemObject(weights_cl);
    
    clReleaseMemObject(inputs_cl);
    clReleaseMemObject(sums_cl);
    clReleaseMemObject(outputs_cl);
    
    clReleaseKernel(kern);
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
}

int get_recommended_size(cl_kernel kern, cl_device_id dev)
{
	size_t sug_size, group_size;
    
    //What's the recommended group size?
	clGetKernelWorkGroupInfo(kern, dev, CL_KERNEL_WORK_GROUP_SIZE,
							 sizeof(size_t), &sug_size, NULL);
    
    //That this rounds to a power of two
    if (sug_size >= 512)
        group_size = 512;
    else if (sug_size >= 256)
        group_size = 256;
    else if (sug_size >= 128)
        group_size = 128;
    else if (sug_size >= 64)
        group_size = 64;
    else
        group_size = 1;
    
#ifndef NDEBUG
	printf("Recommended Size %lu, using %lu\n", sug_size, group_size);
#endif
    return group_size;
}

void opt_train_epoch_batch(struct fann **anns,
                            unsigned int num_anns,
                            struct fann_train_data *data,
                            struct fann_resident_cl *res_cl,
                            float *MSE_values)
{
	size_t group_size, shared_size;
    unsigned int sizes[SIZE_ARGS];
    float f_params[PARAM_ARGS];
	cl_int err = 0;
    unsigned int i;
    
	cl_mem  sizes_cl, num_layers_cl, num_neurons_cl, num_inputs_cl,
            num_outputs_cl, steepness_cl, act_cl, weights_cl,
            f_params_cl, train_errors_cl, weights_deltas_cl,
            sums_cl, outputs_cl;
    
    for(i = 0; i < num_anns; ++i) {
        struct fann *ann = anns[i];
        
        //Check if we can use this network
        if (ann->network_type == FANN_NETTYPE_SOM ||
            ann->network_type == FANN_NETTYPE_GNG) {
            fprintf(stderr, "Network types Incremental, SOM & GNG are not yet supported in OpenCL.\n");
            MSE_values[0] = 0.0;
            return;
        }
        
        //Check if we can use the algorithm
        switch (ann->training_params->training_algorithm)
        {
            case FANN_TRAIN_SARPROP:
            case FANN_TRAIN_RPROP:
            case FANN_TRAIN_QUICKPROP:
            case FANN_TRAIN_BATCH:
                if (fann_check_input_output_sizes(ann, data) == -1) {
                    fprintf(stderr, "Data doesn't match ANN.\n");
                    MSE_values[0] = 0.0;
                    return;
                }
                break;
            default:
                fprintf(stderr, "Training algorithm %s is not yet supported in OpenCL.\n",
                        FANN_TRAIN_NAMES[ann->training_params->training_algorithm]);
                MSE_values[0] = 0.0;
                return;
        }
        
        fann_reset_MSE(ann);
    }
    
    if(res_cl->dev == NULL)
        res_cl->dev = get_device(&res_cl->context);
    
    //if (res_cl->context == NULL) {
    //    //Create a context to perform our calculation with the specified device 
    //    res_cl->context = clCreateContext(0, 1, &(res_cl->dev), NULL, NULL, &err);
    //    assert(err == CL_SUCCESS);
    //}
    
    if (res_cl->cmd_queue == NULL)
        // And also a command queue for the context
        res_cl->cmd_queue = clCreateCommandQueue(res_cl->context, res_cl->dev, 0, NULL);
    
    //Get the training kernel
    if (res_cl->t_kern == NULL)
        res_cl->t_kern = get_kernel("train_batch", res_cl->context, res_cl->dev);
    
    //What's the recommended group size?
    group_size = get_recommended_size(res_cl->t_kern, res_cl->dev);
    shared_size = group_size*sizeof(float);
    
    //Allocate & copy the ANN structure on the device
    make_anns(res_cl->t_kern, anns, 1, data->num_data, group_size, sizes,
              f_params, res_cl->cmd_queue, res_cl->context,
              &sizes_cl, &f_params_cl, &num_layers_cl,
              &num_neurons_cl, &num_inputs_cl, &num_outputs_cl,
              &steepness_cl, &act_cl, &weights_cl );
    
    //Allocate working device memory
    make_working_mem(res_cl->cmd_queue, res_cl->context, res_cl->t_kern, sizes, data->input[0],
                     &(res_cl->inputs_cl), &sums_cl, &outputs_cl);
    
    //Allocate training device memory
    make_training_mem(res_cl->cmd_queue, res_cl->context, res_cl->t_kern, sizes,
                      data->output[0], group_size,
                      &train_errors_cl, &(res_cl->actual_outputs_cl),
                      &(res_cl->MSE_values_cl),
                      &(res_cl->num_bit_fail_cl), &weights_deltas_cl);
    
    //Setup local memory
	err = clSetKernelArg(res_cl->t_kern, 17, sizeof(float)*sizes[2], NULL);
    assert(err == CL_SUCCESS);
	err = clSetKernelArg(res_cl->t_kern, 18, sizeof(int)*sizes[2], NULL);
    assert(err == CL_SUCCESS);
 	err = clSetKernelArg(res_cl->t_kern, 19, shared_size, NULL);
    assert(err == CL_SUCCESS);
	err = clSetKernelArg(res_cl->t_kern, 20, shared_size, NULL);
    assert(err == CL_SUCCESS);
    
    //Run training kernel in parallel via OpenCL
    execute_kern(res_cl->cmd_queue, res_cl->context, res_cl->t_kern, sizes[5], group_size);
    
    //Release unneeded ANN resources
    clReleaseMemObject(steepness_cl);
    clReleaseMemObject(act_cl);
    clReleaseMemObject(weights_cl);
    
    clReleaseMemObject(sums_cl);
    clReleaseMemObject(outputs_cl);
    
    //Get the reduce kernel
    if (res_cl->r_kern == NULL)
        res_cl->r_kern = get_kernel("consolidate_train", res_cl->context, res_cl->dev);
    
    //What's the recommended group size?
    group_size = get_recommended_size(res_cl->r_kern, res_cl->dev);
    shared_size = group_size*sizeof(float);
    
    //Set up args for reducing
    err = clSetKernelArg(res_cl->r_kern,  0, sizeof(cl_mem), &sizes_cl);
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  1, sizeof(cl_mem), &num_layers_cl);
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  2, sizeof(cl_mem), &num_neurons_cl);
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  3, sizeof(cl_mem), &num_inputs_cl);
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  4, sizeof(cl_mem), &num_outputs_cl);
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  5, sizeof(cl_mem), &(res_cl->MSE_values_cl));
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  6, sizeof(cl_mem), &(res_cl->num_bit_fail_cl));
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  7, sizeof(cl_mem), &(train_errors_cl));
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    err = clSetKernelArg(res_cl->r_kern,  8, sizeof(cl_mem), &(weights_deltas_cl));
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
	err = clSetKernelArg(res_cl->r_kern,  9, shared_size, NULL);
    if (err != CL_SUCCESS)
        clSetKernelArg_err(err, __LINE__);
    
    //Run reduction kernel to produce fewer outputs
    execute_kern(res_cl->cmd_queue, res_cl->context, res_cl->r_kern, group_size, group_size);

    //Release unneeded ANN resources
    clReleaseMemObject(sizes_cl);
    clReleaseMemObject(f_params_cl);
    clReleaseMemObject(num_layers_cl);
    clReleaseMemObject(num_neurons_cl);
    clReleaseMemObject(num_inputs_cl);
    clReleaseMemObject(num_outputs_cl);
    
    //Copy output back to the ANN
    copy_train_ann(anns, sizes, res_cl->cmd_queue,
                   res_cl->MSE_values_cl, res_cl->num_bit_fail_cl,
                   train_errors_cl, weights_deltas_cl);
    
    //Release remaining resources which aren't resident
    clReleaseMemObject(train_errors_cl);
    clReleaseMemObject(weights_deltas_cl);
    
    for(i = 0; i < num_anns; ++i) {
        struct fann *ann = anns[i];
        
        //Run ANN updating function as normal
        fann_update_weights(ann);
        MSE_values[i] = fann_get_MSE(ann);
    }
}

//Train for one epoch with the selected training algorithm 
FANN_EXTERNAL float FANN_API fann_train_epoch_cl(struct fann *ann, struct fann_train_data *data)
{
    float err = 0.0;
    struct fann_resident_cl *res_cl = malloc_res();

    opt_train_epoch_batch(&ann, 1, data, res_cl, &err);
    free_res(res_cl);
    
	return err;
}

// Training optimized for doing all work in OpenCL
FANN_EXTERNAL void FANN_API fann_train_on_data_cl(struct fann *ann, struct fann_train_data *data,
                                                  unsigned int max_epochs,
                                                  unsigned int epochs_between_reports,
                                                  float desired_error)
{
	float error;
	unsigned int i;
	int desired_error_reached;
    
    //Maintain between epochs
    struct fann_resident_cl *res_cl = malloc_res();
    
#ifndef NDEBUG
	printf("Training with %s\n", FANN_TRAIN_NAMES[ann->training_params->training_algorithm]);
#endif
    
	if(epochs_between_reports && ann->training_params->callback == NULL)
		printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);
    
	for(i = 1; i <= max_epochs; i++) {
		/*
		 * train 
		 */
        opt_train_epoch_batch(&ann, 1, data, res_cl, &error);
		desired_error_reached = fann_desired_error_reached(ann, desired_error);
        
		/*
		 * print current output 
		 */
		if(epochs_between_reports &&
		   (i % epochs_between_reports == 0 || i == max_epochs || i == 1 ||
			desired_error_reached == 0)) {
			if(ann->training_params->callback == NULL) {
				printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error,
					   ann->training_params->num_bit_fail);
			} else if(((*ann->training_params->callback)(ann, data, max_epochs,
                                                         epochs_between_reports, 
                                                         desired_error, i)) == -1) {
				/*
				 * you can break the training by returning -1 
				 */
				break;
			}
		}
        
		if(desired_error_reached == 0)
			break;
	}
    
    free_res(res_cl);
}

//Run the same set of ANNs on many inputs
FANN_EXTERNAL void FANN_API fann_run_many(struct fann **ann, fann_type * input,
                                          fann_type **output, int num_anns, int num_runs)
{
    //FIXME: ensure that the ANN(s) are compatible with the optimized method here
    
    opt_run_many(ann, input, output, num_anns, num_runs);
}

/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap
 */
