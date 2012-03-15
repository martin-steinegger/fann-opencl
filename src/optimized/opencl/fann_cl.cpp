#include "fann_cl.h"

#ifndef NDEBUG
int device_stats(cl_device_id device_id)
{
	
	int err, i;
	size_t j;
	size_t returned_size;
	
	// Report the device vendor and device name
    // 
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
	cl_char device_profile[1024] = {0};
	cl_char device_extensions[1024] = {0};
    
	cl_device_local_mem_type local_mem_type;
    cl_ulong global_mem_size, global_mem_cache_size;
	cl_ulong max_mem_alloc_size;
	cl_uint clock_frequency, vector_width, max_compute_units;
	size_t max_work_item_dims,max_work_group_size, max_work_item_sizes[3];
	
	cl_uint vector_types[] = {CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE}; 
	char *vector_type_names[] = {"char","short","int","long","float","double"};
	
	err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(device_profile), device_profile, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_extensions), device_extensions, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(global_mem_cache_size), &global_mem_cache_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dims), &max_work_item_dims, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, &returned_size);
	err|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, &returned_size);
	
	printf("Vendor: %s\n", vendor_name);
	printf("Device Name: %s\n", device_name);
	printf("Profile: %s\n", device_profile);
	printf("Supported Extensions: %s\n\n", device_extensions);
	
	printf("Local Mem Type (Local=1, Global=2): %i\n",(int)local_mem_type);
	printf("Global Mem Size (MB): %i\n",(int)global_mem_size/(1024*1024));
	printf("Global Mem Cache Size (Bytes): %i\n",(int)global_mem_cache_size);
	printf("Max Mem Alloc Size (MB): %ld\n",(long int)max_mem_alloc_size/(1024*1024));
	
	printf("Clock Frequency (MHz): %i\n\n",clock_frequency);
	
	for(i=0;i<6;i++){
		err|= clGetDeviceInfo(device_id, vector_types[i], sizeof(clock_frequency), &vector_width, &returned_size);
		printf("Vector type width for: %s = %i\n",vector_type_names[i],vector_width);
	}
	
	printf("\nMax Work Group Size: %lu\n",max_work_group_size);
	printf("Max Work Item Dims: %lu\n",max_work_item_dims);
	for(j=0;j<max_work_item_dims;j++) 
		printf("Max Work Items in Dim %lu: %lu\n",(long unsigned)(j+1),(long unsigned)max_work_item_sizes[j]);
	
	printf("Max Compute Units: %i\n",max_compute_units);
	printf("\n");
	
	return CL_SUCCESS;
}
#endif

void clEnqueueNDRangeKernel_err(cl_int err)
{
#ifndef NDEBUG
    printf("clEnqueueNDRangeKernel return value:\n");
    switch (err)
    {
        case CL_INVALID_PROGRAM_EXECUTABLE:
            printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
            break;
        case CL_INVALID_COMMAND_QUEUE:
            printf("CL_INVALID_COMMAND_QUEUE\n");
            break;
        case CL_INVALID_KERNEL:
            printf("CL_INVALID_KERNEL\n");
            break;
        case CL_INVALID_CONTEXT:
            printf("CL_INVALID_CONTEXT\n");
            break;
        case CL_INVALID_KERNEL_ARGS:
            printf("CL_INVALID_KERNEL_ARGS\n");
            break;
        case CL_INVALID_WORK_DIMENSION:
            printf("CL_INVALID_WORK_DIMENSION\n");
            break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
            printf("CL_INVALID_GLOBAL_WORK_SIZE\n");
            break;
        case CL_INVALID_WORK_GROUP_SIZE:
            printf("CL_INVALID_WORK_GROUP_SIZE\n");
            break;
        case CL_INVALID_WORK_ITEM_SIZE:
            printf("CL_INVALID_WORK_ITEM_SIZE\n");
            break;
        case CL_INVALID_GLOBAL_OFFSET:
            printf("CL_INVALID_GLOBAL_OFFSET\n");
            break;
        case CL_OUT_OF_RESOURCES:
            printf("CL_OUT_OF_RESOURCES\n");
            break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
            break;
        case CL_INVALID_EVENT_WAIT_LIST:
            printf("CL_INVALID_EVENT_WAIT_LIST\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            printf("CL_OUT_OF_HOST_MEMORY\n");
            break;
    }
#endif
    exit(-1);
}

void  clSetKernelArg_err(cl_int err, int line)
{
#ifndef NDEBUG
    printf("clSetKernelArg return value (line %d):\n", line);
    switch (err)
    {
        case CL_INVALID_KERNEL:
            printf("CL_INVALID_KERNEL\n");
            break;
        case CL_INVALID_ARG_INDEX:
            printf("CL_INVALID_ARG_INDEX\n");
            break;
        case CL_INVALID_ARG_VALUE:
            printf("CL_INVALID_ARG_VALUE\n");
            break;
        case CL_INVALID_MEM_OBJECT:
            printf("CL_INVALID_MEM_OBJECT\n");
            break;
        case CL_INVALID_SAMPLER:
            printf("CL_INVALID_SAMPLER\n");
            break;
        case CL_INVALID_ARG_SIZE:
            printf("CL_INVALID_ARG_SIZE\n");
            break;
    }
#endif
    exit(-1);
}

void clEnqueueReadBuffer_err(cl_int err, int line)
{
#ifndef NDEBUG
    printf("clEnqueueReadBuffer return value (line %d):\n", line);
    switch (err)
    {
        case CL_INVALID_MEM_OBJECT:
            printf("CL_INVALID_MEM_OBJECT\n");
            break;
        case CL_INVALID_VALUE:
            printf("CL_INVALID_VALUE\n");
            break;
    }
#endif
    exit(-1);
}

void clGetEventProfilingInfo_err(cl_int err, cl_event ev)
{
#ifndef NDEBUG
    cl_int status;
    printf("clGetEventProfilingInfo return value:\n");
    switch (err)
    {
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            printf("CL_PROFILING_INFO_NOT_AVAILABLE\n");
            break;
        case CL_INVALID_VALUE:
            printf("CL_INVALID_VALUE\n");
            break;
        case CL_INVALID_EVENT:
            printf("CL_INVALID_EVENT\n");
            break;
    }
    clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
    printf("clGetEventInfo returns:\n");
    switch (status)
    {
        case CL_QUEUED:
            printf("CL_QUEUED\n");
            break;
        case CL_SUBMITTED:
            printf("CL_SUBMITTED\n");
            break;
        case CL_RUNNING:
            printf("CL_RUNNING\n");
            break;
        case CL_COMPLETE:
            printf("CL_COMPLETE\n");
            break;
    }
#endif
    exit(-1);
}

void clFinish_err(cl_int err)
{
#ifndef NDEBUG
    printf("clFinish return value:\n");
    switch (err)
    {
        case CL_INVALID_COMMAND_QUEUE:
            printf("CL_INVALID_COMMAND_QUEUE\n");
            break;
        case CL_OUT_OF_HOST_MEMORY:
            printf("CL_OUT_OF_HOST_MEMORY\n");
            break;
    }
#endif
    exit(-1);
}

void clSetCommandQueueProperty_err(cl_int err)
{
#ifndef NDEBUG
    printf("clSetCommandQueueProperty return value:\n");
    switch (err)
    {
        case CL_INVALID_COMMAND_QUEUE:
            printf("CL_INVALID_COMMAND_QUEUE\n");
            break;
        case CL_INVALID_VALUE:
            printf("CL_INVALID_VALUE\n");
            break;
        case CL_INVALID_QUEUE_PROPERTIES:
            printf("CL_INVALID_PROPERTIES\n");
            break;
    }
#endif
    exit(-1);
}

cl_device_id get_device(cl_context *cont)
{
	cl_int err = 0;
	cl_device_id device = NULL;
#ifndef NDEBUG
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
#endif

    // Find the GPU CL device, this is what we really want
    // If there is no GPU device is CL capable, fall back to CPU
//    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    printf("Trying to run on a CPU \n");
    cl_platform_id intel_platform_id = GetIntelOCLPlatform();
    if( intel_platform_id == NULL )
    {
        printf("ERROR: Failed to find Intel OpenCL platform.\n");
        return device;
    }

    cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

    // create the OpenCL context on a CPU 
    cl_context context;
	context=*cont=clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
    if (context == (cl_context)0)
        return device;

    // get the list of CPU devices associated with context
    size_t cb;
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);

    cl_device_id *devices;
    devices = (cl_device_id*)malloc(cb);

    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL);
//    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    if (err != CL_SUCCESS)
    {
        // Find the CPU CL device, as a fallback
        err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        assert(err == CL_SUCCESS);
    }
	else
		device = devices[0];
	return device;
    assert(device);
    
#ifndef NDEBUG
    // Get some information about the returned device
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), 
                          vendor_name, &returned_size);
    err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
                           device_name, &returned_size);
    assert(err == CL_SUCCESS);
    printf("Connecting to %s %s...\n", vendor_name, device_name);
    
//    device_stats(device);
#endif
    
    return device;
}

void make_training_mem(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                       unsigned int *sizes, fann_type *output, unsigned int grp_size,
                       cl_mem *train_errors_cl, cl_mem *actual_outputs_cl, cl_mem *MSE_values_cl,
                       cl_mem *num_bit_fail_cl, cl_mem *weights_deltas_cl)
{
    cl_int err = CL_SUCCESS;
    long sz;
	unsigned int i;
    float *out_work = NULL;
    cl_mem out_work_cl = NULL;
    unsigned int num_runs = sizes[5];
    unsigned int num_output = sizes[8];
    unsigned int num_grps;
    
    //Round up num_runs
    if(num_runs % grp_size)
        num_grps = 1+(num_runs/grp_size);
    else
        num_grps = num_runs/grp_size;
    
    //Allocate actual output memory if it doesn't already exist
    if ((*actual_outputs_cl) == NULL) {
        sz = sizeof(float) * num_runs*num_output;
        assert(sz >= 0);
        (*actual_outputs_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
        assert(err == CL_SUCCESS);
        out_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
        assert(err == CL_SUCCESS);
        out_work = (float *)clEnqueueMapBuffer(cmd_queue, out_work_cl, CL_TRUE, CL_MAP_WRITE,
                                               0, sz, 0, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        
        //Copy to pinned memory & permute for coalesced reads on the GPU
        for(i = 0; i < num_output; ++i) {
            unsigned int i_num_runs = i*num_runs;
            int adj_runs = num_runs-4;
            int j;
            
            //Unroll
            for(j = 0; j <= adj_runs; j += 4) {
                out_work[i_num_runs+j  ] = output[ j   *num_output+i];
                out_work[i_num_runs+j+1] = output[(j+1)*num_output+i];
                out_work[i_num_runs+j+2] = output[(j+2)*num_output+i];
                out_work[i_num_runs+j+3] = output[(j+3)*num_output+i];
            }
            
            //Handle unrolled
            switch (num_runs % 4) {
                case 3: out_work[i_num_runs+num_runs-3] = output[(num_runs-3)*num_output+i];
                case 2: out_work[i_num_runs+num_runs-2] = output[(num_runs-2)*num_output+i];
                case 1: out_work[i_num_runs+num_runs-1] = output[(num_runs-1)*num_output+i];
            }
        }
        
        err = clEnqueueWriteBuffer(cmd_queue, (*actual_outputs_cl), CL_FALSE, 0, sz,
                                   (void*)out_work, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }

    //Allocate neuron output memory
    sz = sizeof(float) * num_runs*sizes[4]*sizes[1]*sizes[0];
    assert(sz > 0);
    (*train_errors_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    
    sz = sizeof(float) * num_runs*sizes[0];
    assert(sz > 0);
    
    if ((*MSE_values_cl) == NULL) {
        (*MSE_values_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
        assert(err == CL_SUCCESS);
    }
    
    if ((*num_bit_fail_cl) == NULL) {
        (*num_bit_fail_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
        assert(err == CL_SUCCESS);
    }
    
    sz = sizeof(float) * num_grps*sizes[3]*sizes[4]*sizes[1]*sizes[0];
	assert(sz > 0);
    (*weights_deltas_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    
    // Get all of the stuff written and allocated 
    clFinish(cmd_queue);
    
    //Clean up mem space
    if (out_work_cl != NULL)
        clReleaseMemObject(out_work_cl);
    
    //Set up these as arguments
    err = clSetKernelArg(kern, 12, sizeof(cl_mem), train_errors_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 13, sizeof(cl_mem), actual_outputs_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 14, sizeof(cl_mem), MSE_values_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 15, sizeof(cl_mem), num_bit_fail_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 16, sizeof(cl_mem), weights_deltas_cl);
    assert(err == CL_SUCCESS);
}

void make_working_mem(cl_command_queue cmd_queue, cl_context context, cl_kernel kern,
                      unsigned int *sizes, fann_type *input,
                      cl_mem *inputs_cl, cl_mem *sums_cl, cl_mem *outputs_cl)
{
    cl_int err = CL_SUCCESS;
    unsigned int sz, i;
    float *in_work = NULL;
    cl_mem in_work_cl = NULL;
    unsigned int num_runs = sizes[5];
    unsigned int num_input = sizes[6];
    
    //Allocate input memory if non-existant
    if ((*inputs_cl) == NULL) {
        sz = sizeof(float) * num_runs*num_input;
        assert(sz >= 0);
        (*inputs_cl) = clCreateBuffer(context, CL_MEM_READ_ONLY, sz, NULL, &err);
        assert(err == CL_SUCCESS);
        in_work_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sz, NULL, &err);
        assert(err == CL_SUCCESS);
        in_work = (float *)clEnqueueMapBuffer(cmd_queue, in_work_cl, CL_TRUE, CL_MAP_WRITE,
                                              0, sz, 0, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        
        //Copy to pinned memory & permute for coalesced reads on the GPU
        for(i = 0; i < num_input; ++i) {
            unsigned int i_num_runs = i*num_runs;
            int adj_runs = num_runs-4;
            int j;
            
            //Unroll
            for(j = 0; j <= adj_runs; j += 4) {
                in_work[i_num_runs+j  ] = input[ j   *num_input+i];
                in_work[i_num_runs+j+1] = input[(j+1)*num_input+i];
                in_work[i_num_runs+j+2] = input[(j+2)*num_input+i];
                in_work[i_num_runs+j+3] = input[(j+3)*num_input+i];
            }
            
            //Handle unrolled
            switch (num_runs % 4) {
                case 3: in_work[i_num_runs+num_runs-3] = input[(num_runs-3)*num_input+i];
                case 2: in_work[i_num_runs+num_runs-2] = input[(num_runs-2)*num_input+i];
                case 1: in_work[i_num_runs+num_runs-1] = input[(num_runs-1)*num_input+i];
            }
        }
        
        err = clEnqueueWriteBuffer(cmd_queue, (*inputs_cl), CL_FALSE, 0, sz,
                                   (void*)in_work, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }
    
    //Allocate neuron output memory
    sz = sizeof(float) * num_runs*sizes[4]*sizes[1]*sizes[0];
    assert(sz > 0);
    (*sums_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    
    (*outputs_cl) = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, &err);
    assert(err == CL_SUCCESS);
    
    // Get all of the stuff written and allocated 
    clFinish(cmd_queue);
    
    //Clean up mem space
    if (in_work_cl != NULL)
        clReleaseMemObject(in_work_cl);
    
    //Set up these as arguments
    err = clSetKernelArg(kern,  9, sizeof(cl_mem), inputs_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 10, sizeof(cl_mem), sums_cl);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kern, 11, sizeof(cl_mem), outputs_cl);
    assert(err == CL_SUCCESS);
}

cl_platform_id GetIntelOCLPlatform()
{
    cl_platform_id pPlatforms[10] = { 0 };
    char pPlatformName[128] = { 0 };

    cl_uint uiPlatformsCount = 0;
    cl_int err = clGetPlatformIDs(10, pPlatforms, &uiPlatformsCount);
    for (cl_uint ui = 0; ui < uiPlatformsCount; ++ui)
    {
        err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
        if ( err != CL_SUCCESS )
        {
            printf("ERROR: Failed to retreive platform vendor name.\n", ui);
            return NULL;
        }

        //if (!strcmp(pPlatformName, "Intel(R) OpenCL"))
            return pPlatforms[ui];
    }

    return NULL;
}
