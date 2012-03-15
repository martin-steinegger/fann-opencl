#include "fann_cl.h"

void copy_run_output(cl_command_queue cmd_queue, cl_context context, struct fann ** anns,
                     unsigned int *sizes, cl_mem output_cl, fann_type **output)
{
    cl_int err = 0;
    unsigned int ann_num;
    cl_mem out_work_cl;
    float *out_work;
    unsigned int num_runs = sizes[5];
    unsigned int sz = sizeof(float) * num_runs*sizes[4];
    
    //Alloc a pinned array to work from
    assert(sizeof(fann_type) == sizeof(float));
    out_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    out_work = (float *)clEnqueueMapBuffer(cmd_queue, out_work_cl, CL_TRUE, CL_MAP_WRITE,
                                           0, sz, 0, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    
    //De-permute & copy into user's output array(s)
    for(ann_num = 0; ann_num < sizes[0]; ++ann_num) {
        struct fann *ann = anns[ann_num];
        unsigned int num_layers = ann->last_layer - ann->first_layer;
        unsigned int j;
        unsigned int num_out = ann->num_output;
        int out_offset = sizeof(float) * (ann_num*sizes[1]+num_layers-1)*sizes[4]*sizes[5];
        fann_type *out = output[ann_num];
        
        //Copy from dev into working memory
        err = clEnqueueReadBuffer(cmd_queue, output_cl, CL_TRUE, out_offset,
                                  sz, out_work, 0, NULL, NULL);
        if (err != CL_SUCCESS)
            clEnqueueReadBuffer_err(err, __LINE__);
        
        //For each output value
        for(j = 0; j < num_out; ++j) {
            unsigned int f_out_off = j*num_runs;
            int adj_runs = num_runs-4;
            int i;

            //For each run of the ANN
            for(i = 0; i <= adj_runs; i += 4) {
                out[ i   *num_out+j] = out_work[f_out_off+i];
                out[(i+1)*num_out+j] = out_work[f_out_off+i+1];
                out[(i+2)*num_out+j] = out_work[f_out_off+i+2];
                out[(i+3)*num_out+j] = out_work[f_out_off+i+3];
            }
            
            switch (num_runs % 4) {
                case 3: out[(num_runs-3)*num_out+j] = out_work[f_out_off+num_runs-3];
                case 2: out[(num_runs-2)*num_out+j] = out_work[f_out_off+num_runs-2];
                case 1: out[(num_runs-1)*num_out+j] = out_work[f_out_off+num_runs-1];
            }
        }
    }
    
    //Release working memory
    clReleaseMemObject(out_work_cl);
}

void execute_kern(cl_command_queue cmd_queue, cl_context context,
                  cl_kernel kern, int num_runs, size_t loc_size)
{
    cl_int err = 0;
    cl_event ev;
    size_t ceil_runs;
#ifndef NDEBUG
    size_t start_time = 0;
    size_t end_time;
#endif
    
    //Round up num_runs
    if(num_runs % loc_size)
        ceil_runs = num_runs + loc_size - num_runs % loc_size;
    else
        ceil_runs = num_runs;
    
#ifndef NDEBUG
    //err = clSetCommandQueueProperty(cmd_queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
    //if (err != CL_SUCCESS)
    //    clSetCommandQueueProperty_err(err);
#endif
    
    // Run the calculation by enqueuing it and forcing the 
    // command queue to complete the task
    err = clEnqueueNDRangeKernel(cmd_queue, kern, 1, NULL, 
                                 &ceil_runs, &loc_size, 0, NULL, &ev);
    if (err != CL_SUCCESS)
        clEnqueueNDRangeKernel_err(err);
    
    err = clFinish(cmd_queue);
    if (err != CL_SUCCESS)
        clFinish_err(err);
    
#ifndef NDEBUG
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(size_t), &start_time, NULL);
    if (err != CL_SUCCESS)
        clGetEventProfilingInfo_err(err, ev);
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(size_t), &end_time, NULL);
    if (err != CL_SUCCESS)
        clGetEventProfilingInfo_err(err, ev);
    assert(end_time != 0);
    assert(start_time != 0);
    printf("Kernel Time: %15lu\n", (long int)((end_time-start_time)/100000));
#endif
}
