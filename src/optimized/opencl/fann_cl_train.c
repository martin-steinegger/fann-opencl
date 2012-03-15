#include "fann_cl.h"

#ifndef NDEBUG
void dump_train_vals(struct fann *ann)
{
    struct fann_layer *layer_it, *last_layer;
    
    last_layer = ann->last_layer;
    
    printf("  MSE_value, bit_fail, num_MSE: %10f, %10d, %10d\n",
           ann->training_params->MSE_value,
           ann->training_params->num_bit_fail,
           ann->training_params->num_MSE);
    
    //For each layer
    for (layer_it = ann->first_layer; layer_it != last_layer; ++layer_it) {
        struct fann_neuron *neuron_it, *last_neuron;
        unsigned int neuron_num = 0;
        unsigned int l_output_num = 0;
        last_neuron = layer_it->last_neuron;
        //For each neuron
        for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; ++neuron_it) {
            unsigned int n_output_num;
            unsigned int num_outputs = neuron_it->num_outputs;
            unsigned int num_inputs = neuron_it->num_inputs;
            
            printf("             num_backprop_done: %10d\n", neuron_it->num_backprop_done);
            
            //For each output
            for (n_output_num = 0; n_output_num < num_outputs; ++n_output_num) {
                unsigned int in_num;
                
                printf("                  train_errors: %10f\n",
                       neuron_it->train_errors[n_output_num]);
                
                printf("                weights_deltas:");
                
                //For each input
                for (in_num = 0; in_num < num_inputs; ++in_num) {
                    unsigned int n_weight_index = n_output_num*num_inputs+in_num;
                    
                    printf("  %10f", neuron_it->weights_deltas[n_weight_index]);
                }
                printf("\n");
                
                ++l_output_num;
            }
            ++neuron_num;
        }
    }
}
#endif

void copy_train_ann(struct fann **anns, unsigned int *sizes,
                    cl_command_queue cmd_queue, cl_mem MSE_values_cl,
                    cl_mem num_bit_fail_cl,
                    cl_mem train_errors_cl, cl_mem weights_deltas_cl)
{
    unsigned int ann_id, num_grps;
    unsigned int num_runs = sizes[5];
    cl_int err;
    
    //Round up num_runs
    if(num_runs % sizes[7])
        num_grps = 1+(num_runs/sizes[7]);
    else
        num_grps = num_runs/sizes[7];
    
    //For each ANN
    for (ann_id = 0; ann_id < sizes[0]; ++ann_id) {
        struct fann *ann = anns[ann_id];
        struct fann_layer *layer_it, *last_layer;
        float bit_fail_f;
		unsigned int net_index = ann_id*sizes[5];
        unsigned int layer_num = 0;

        last_layer = ann->last_layer;
        ann->training_params->num_MSE = 0;
        
        //For each layer
        for (layer_it = ann->first_layer; layer_it != last_layer; ++layer_it) {
            struct fann_neuron *neuron_it, *last_neuron;
            unsigned int neuron_num = 0;
            unsigned int l_output_num = 0;
            unsigned int out_off = (ann_id*sizes[1]+layer_num)*sizes[4];
            last_neuron = layer_it->last_neuron;
            
            if (layer_it->train_errors==NULL)
                layer_it->initialize_train_errors(ann, layer_it);
            
            //For each neuron
            for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; ++neuron_it) {
                unsigned int n_output_num;
                unsigned int num_outputs = neuron_it->num_outputs;
                unsigned int num_inputs = neuron_it->num_inputs;
                
                //Replicate the counter
                neuron_it->num_backprop_done += sizes[5];
                
                //For each output
                for (n_output_num = 0; n_output_num < num_outputs; ++n_output_num) {
                    unsigned int out_index = (out_off+l_output_num)*sizes[5];
                    unsigned int in_num;
                    
                    //For each input
                    for (in_num = 0; in_num < num_inputs; ++in_num) {
                        unsigned int in_index = ((out_off+l_output_num)*sizes[3]+in_num)*num_grps;
                        unsigned int n_weight_index = n_output_num*num_inputs+in_num;
                        
                        //Copy the weight deltas
                        err = clEnqueueReadBuffer(cmd_queue, weights_deltas_cl, CL_FALSE, in_index*sizeof(float),
                                                  sizeof(float), &(neuron_it->weights_deltas[n_weight_index]), 0, NULL, NULL);
                        if (err != CL_SUCCESS)
                            clEnqueueReadBuffer_err(err, __LINE__);
                    }
                    
                    //Replicate the counter (for outputs in the last layer)
                    if(layer_it+1 == last_layer)
                        ann->training_params->num_MSE += sizes[5];
                    
                    //Copy the train_errors
                    err = clEnqueueReadBuffer(cmd_queue, train_errors_cl, CL_FALSE, out_index*sizeof(float),
                                              sizeof(float), &(neuron_it->train_errors[n_output_num]), 0, NULL, NULL);
                    if (err != CL_SUCCESS)
                        clEnqueueReadBuffer_err(err, __LINE__);
                    
                    ++l_output_num;
                }
                ++neuron_num;
            }
            ++layer_num;
        }
        
        //Copy the MSE value from the device
        err = clEnqueueReadBuffer(cmd_queue, MSE_values_cl, CL_FALSE, net_index*sizeof(float),
                                  sizeof(float), &(ann->training_params->MSE_value), 0, NULL, NULL);
        if (err != CL_SUCCESS)
            clEnqueueReadBuffer_err(err, __LINE__);
        
        //Copy the num_bit_fail from the device
        err = clEnqueueReadBuffer(cmd_queue, num_bit_fail_cl, CL_TRUE, net_index*sizeof(float),
                                  sizeof(float), &bit_fail_f, 0, NULL, NULL);
        if (err != CL_SUCCESS)
            clEnqueueReadBuffer_err(err, __LINE__);
        
//        ann->training_params->num_bit_fail = roundf(bit_fail_f);
        ann->training_params->num_bit_fail = bit_fail_f;
    }
    
    clFinish(cmd_queue);
    
//    dump_train_vals(anns[0]);
}
