void make_anns(cl_kernel kern, struct fann **anns, int num_anns, int num_runs, unsigned int group_size,
               unsigned int *sizes, float *f_params,
               cl_command_queue cmd_queue, cl_context context,
               cl_mem *sizes_cl, cl_mem *f_params_cl, cl_mem *num_layers_cl, cl_mem *num_neurons_cl,
               cl_mem *num_inputs_cl, cl_mem *num_outputs_cl,
               cl_mem *steepness_cl, cl_mem *act_cl, cl_mem *weights_cl );
