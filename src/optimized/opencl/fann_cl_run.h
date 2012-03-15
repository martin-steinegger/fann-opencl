void execute_kern(cl_command_queue cmd_queue, cl_context context,
                  cl_kernel kern, int num_runs, size_t loc_size);

void copy_run_output(cl_command_queue cmd_queue, cl_context context, struct fann ** anns,
                     unsigned int *sizes, cl_mem outputs_cl, fann_type **output);
