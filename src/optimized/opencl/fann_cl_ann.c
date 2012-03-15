#include "fann_cl.h"

void find_limits(struct fann **anns, unsigned int num_anns, unsigned int num_runs, unsigned int group_size, unsigned int *sizes, float *f_params)
{
    unsigned int ann_num;
    unsigned int max_num_layers = 0;
    unsigned int max_num_neurons = 0;
    unsigned int max_num_inputs = 0;
    unsigned int max_num_outputs = 0;
    unsigned int max_num_fin_out = 0;
    
    //Find the maximum in any dimension
    for(ann_num = 0; ann_num < num_anns; ++ann_num)
    {
        struct fann *ann = anns[ann_num];
        struct fann_layer *layer;
        unsigned int num_layers = 0;
        
        for(layer = ann->first_layer; layer != ann->last_layer; ++layer){
            struct fann_neuron * neuron;
            unsigned int num_outputs = 0;
            
            for(neuron = layer->first_neuron; neuron != layer->last_neuron; ++neuron){
                
                if (max_num_inputs < neuron->num_inputs)
                    max_num_inputs = neuron->num_inputs;
                
                num_outputs += neuron->num_outputs;
            }
            
            if (max_num_neurons < layer->num_neurons)
                max_num_neurons = layer->num_neurons;
            
            if(max_num_outputs < num_outputs)
                max_num_outputs = num_outputs;
            
            ++num_layers;
        }
        
        if (max_num_fin_out < ann->num_output)
            max_num_fin_out = ann->num_output;
        
        if (max_num_layers < num_layers)
            max_num_layers = num_layers;
    }
    
    //Save our numbers!
    sizes[0] = num_anns;
    sizes[1] = max_num_layers;
    sizes[2] = max_num_neurons;
    sizes[3] = max_num_inputs;
    sizes[4] = max_num_outputs;
    sizes[5] = num_runs;
    sizes[6] = anns[0]->num_input;
    sizes[7] = group_size;
    sizes[8] = max_num_fin_out;

    //Training params (FIXME: make this work for multiple ANNs)
    if (anns[0]->training_params != NULL) {
        sizes[9] = anns[0]->training_params->train_error_function;
        f_params[0] = anns[0]->training_params->bit_fail_limit;
    }
}

void fill_arrays(struct fann **anns, unsigned int num_anns, unsigned int *sizes,
                 unsigned int *num_layers, unsigned int *num_neurons,
                 unsigned int *num_outputs_arr, unsigned int *num_inputs_arr,
                 float *steepness, int *act, float *weights)
{
    unsigned int ann_num;
    unsigned int layer_i;
    unsigned int neuron_i;
    
    //Arrange host memory
    for(ann_num = 0; ann_num < num_anns; ++ann_num){
        struct fann *ann = anns[ann_num];
        struct fann_layer *layer;
        layer_i = 0;
        
        for(layer = ann->first_layer; layer != ann->last_layer; ++layer){
            struct fann_neuron * neuron;
            unsigned int n_layer_off = sizes[2]*(ann_num*sizes[1]+layer_i);
            unsigned int o_layer_off = sizes[4]*(ann_num*sizes[1]+layer_i);
            neuron_i = 0;
            
            for(neuron = layer->first_neuron; neuron != layer->last_neuron; ++neuron){
                unsigned int n_index = n_layer_off+neuron_i;
                unsigned int num_inputs = neuron->num_inputs;
                unsigned int num_outputs = neuron->num_outputs;
                unsigned int layer_o = 0;
                unsigned int o;
                
                num_inputs_arr[n_index] = num_inputs;
                num_outputs_arr[n_index] = num_outputs;
                steepness[n_index] = neuron->activation_steepness;
                act[n_index] = neuron->activation_function;
                
                for (o=0; o < num_outputs; ++o){
                    unsigned int o_index = (o_layer_off+layer_o+o)*sizes[3];
                    unsigned int in;
                    for(in = 0; in < num_inputs; ++in){
                        weights[o_index+in] = neuron->weights[num_inputs*o+in];
                    }
                }
                
                layer_o += num_outputs;
                ++neuron_i;
            }
            num_neurons[sizes[1]*ann_num+layer_i] = neuron_i;
            
            ++layer_i;
        }
        num_layers[ann_num] = layer_i;
    }
}

void make_anns(cl_kernel kern, struct fann **anns, int num_anns, int num_runs,
               unsigned int group_size, unsigned int *sizes, float *f_params,
               cl_command_queue cmd_queue, cl_context context,
               cl_mem *sizes_cl, cl_mem *f_params_cl, cl_mem *num_layers_cl,
               cl_mem *num_neurons_cl, cl_mem *num_inputs_cl, cl_mem *num_outputs_cl,
               cl_mem *steepness_cl, cl_mem *act_cl, cl_mem *weights_cl )
{
	cl_int err = 0;
    
    cl_mem num_layers_work_cl = NULL;
    cl_mem num_neurons_work_cl = NULL;
    cl_mem num_inputs_work_cl = NULL;
    cl_mem num_outputs_work_cl = NULL;
    
    cl_mem steepness_work_cl = NULL;
    cl_mem act_work_cl = NULL;
    cl_mem weights_work_cl = NULL;
    
    unsigned int *num_layers_work = NULL;
    unsigned int *num_neurons_work = NULL;
    unsigned int *num_inputs_work = NULL;
    unsigned int *num_outputs_work = NULL;
    
    float *steepness_work = NULL;
    int *act_work = NULL;
    float *weights_work = NULL;
    
    unsigned int sz;
    
    //Fill the sizes array
    find_limits(anns, num_anns, num_runs, group_size, sizes, f_params);

    //Alloc & copy to dev
    sz = sizeof(unsigned int) * SIZE_ARGS;
    (*sizes_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    err = clEnqueueWriteBuffer(cmd_queue, (*sizes_cl), CL_TRUE, 0, sz,
                               (void*)sizes, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(unsigned int) * PARAM_ARGS;
    (*f_params_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    err = clEnqueueWriteBuffer(cmd_queue, (*f_params_cl), CL_TRUE, 0, sz,
                               (void*)f_params, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    //Allocate pinned host memory
    sz = sizeof(unsigned int) * num_anns;
    num_layers_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    num_layers_work = (unsigned int *)clEnqueueMapBuffer(cmd_queue, num_layers_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                         0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(unsigned int) * num_anns*sizes[1];
    num_neurons_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    num_neurons_work = (unsigned int *)clEnqueueMapBuffer(cmd_queue, num_neurons_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                          0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(unsigned int) * num_anns*sizes[1]*sizes[2];
    num_inputs_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    num_inputs_work = (unsigned int *)clEnqueueMapBuffer(cmd_queue, num_inputs_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                         0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    num_outputs_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    num_outputs_work = (unsigned int *)clEnqueueMapBuffer(cmd_queue, num_outputs_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                          0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(float) * num_anns*sizes[1]*sizes[2];
    steepness_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    steepness_work = (float *)clEnqueueMapBuffer(cmd_queue, steepness_work_cl, CL_TRUE, CL_MAP_WRITE,
                                                 0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(enum fann_activationfunc_enum) * num_anns*sizes[1]*sizes[2];
    act_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    act_work = (int *)clEnqueueMapBuffer(cmd_queue, act_work_cl, CL_TRUE, CL_MAP_WRITE,
                                         0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(float) * num_anns*sizes[1]*sizes[2]*sizes[3]*sizes[4];
    weights_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    weights_work = (float *)clEnqueueMapBuffer(cmd_queue, weights_work_cl, CL_TRUE, CL_MAP_WRITE,
                                               0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    //Put values in the arrays
    fill_arrays(anns, num_anns, sizes, num_layers_work, num_neurons_work,
                num_outputs_work, num_inputs_work, steepness_work, act_work, weights_work);
    
    //Allocate & copy CL memory
    sz = sizeof(unsigned int) * num_anns;
    (*num_layers_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*num_layers_cl), CL_FALSE, 0, sz,
                               (void*)num_layers_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(unsigned int) * num_anns*sizes[1];
    (*num_neurons_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*num_neurons_cl), CL_FALSE, 0, sz,
                               (void*)num_neurons_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(unsigned int) * num_anns*sizes[1]*sizes[2];
    (*num_inputs_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*num_inputs_cl), CL_FALSE, 0, sz,
                               (void*)num_inputs_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    (*num_outputs_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*num_outputs_cl), CL_FALSE, 0, sz,
                               (void*)num_outputs_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(float) * num_anns*sizes[1]*sizes[2];
    (*steepness_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*steepness_cl), CL_FALSE, 0, sz,
                               (void*)steepness_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(enum fann_activationfunc_enum) * num_anns*sizes[1]*sizes[2];
    (*act_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*act_cl), CL_FALSE, 0, sz,
                               (void*)act_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(float) * num_anns*sizes[1]*sizes[2]*sizes[3]*sizes[4];
    (*weights_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, (*weights_cl), CL_FALSE, 0, sz,
                               (void*)weights_work, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    // Get all of the stuff copied
    clFinish(cmd_queue);
    
    //Free host memory
    clReleaseMemObject(num_layers_work_cl);
    clReleaseMemObject(num_neurons_work_cl);
    clReleaseMemObject(num_inputs_work_cl);
    clReleaseMemObject(num_outputs_work_cl);
    
    clReleaseMemObject(steepness_work_cl);
    clReleaseMemObject(act_work_cl);
    clReleaseMemObject(weights_work_cl);
    
    //Set up kenel args
    err = clSetKernelArg(kern,  0, sizeof(cl_mem), sizes_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  1, sizeof(cl_mem), f_params_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  2, sizeof(cl_mem), num_layers_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  3, sizeof(cl_mem), num_neurons_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  4, sizeof(cl_mem), num_inputs_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  5, sizeof(cl_mem), num_outputs_cl);
    assert(err == CL_SUCCESS);
    
    err = clSetKernelArg(kern,  6, sizeof(cl_mem), steepness_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  7, sizeof(cl_mem), act_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern,  8, sizeof(cl_mem), weights_cl);
    assert(err == CL_SUCCESS);
}

