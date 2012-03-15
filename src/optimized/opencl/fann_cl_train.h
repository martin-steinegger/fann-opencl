void copy_train_ann(struct fann **anns, unsigned int *sizes,
                    cl_command_queue cmd_queue, cl_mem MSE_values_cl,
                    cl_mem num_bit_fail_cl,
                    cl_mem train_errors_cl, cl_mem weights_deltas_cl);
