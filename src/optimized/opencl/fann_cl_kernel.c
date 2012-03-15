#include "fann_activation.h"
#include "fann_cl.h"

/* Expected structure of arrays:
 num_layers     [ann_id]
 num_neurons    [ann_id][layer_num]
 num_inputs     [ann_id][layer_num][neuron_num]
 num_outputs    [ann_id][layer_num][neuron_num]
 
 steepness      [ann_id][layer_num][neuron_num]
 activation     [ann_id][layer_num][neuron_num]
 weights        [ann_id][layer_num][out_num][in_num]
 
 inputs         [in_num][global_id]
 actual_outputs [ann_id][out_num][global_id]
 sums           [ann_id][layer_num][out_num][global_id]
 outputs        [ann_id][layer_num][out_num][global_id]
 train_errors   [ann_id][layer_num][out_num][global_id]
 weights_deltas [ann_id][layer_num][out_num][in_num][group_id]
 
 num_bit_fail   [ann_id][global_id]
 MSE_values     [ann_id][global_id]
 
 FIXME: When I wrote this code, the __constant address space qualifier
 was causing strange problems (i.e. CLH_ERROR_NO_BINARY_FOR_GPU). When Apple
 fixes it to work as expected add __constant back in as appropriate. I've
 kept it in as much as possible without throwing errors, but it could be more used.
 
 FIXME: We'll get better use of memory and a bit of a speedup if we use group_id
 indexing like in weights_deltas instead of global_id indexing in more arrays.
 Anywhere it's used will have to be write-only, though.
 */

cl_kernel get_kernel(char *kern_name, cl_context context, cl_device_id device)
{
	cl_program program;
	cl_kernel kernel;
	cl_int err = 0;
    
    char *fin_program_src;
    const char *program_source = 
	"void sum_reduce_and_store(__local float *sdata,\n"
							  "__global float *store_arr,\n"
							  "float value,\n"
							  "int store_off)\n"
	"{\n"
		//Note that this draws from NVIDIA's reduction example:
		//- Doesn't use % operator.
		//- Uses contiguous threads.
		//- Uses sequential addressing -- no divergence or bank conflicts.
		//- Is completely unrolled.
		// local size must be a power of 2 and (>= 64 or == 1)
		"unsigned int lsz = get_local_size(0);\n"
		"unsigned int lid = get_local_id(0);\n"
		"sdata[lid] = value;\n"
		"barrier(CLK_LOCAL_MEM_FENCE);\n"
		
		// do reduction in shared mem
        "if (lsz != 1) {\n"
            "if (lsz >= 512) { if (lid < 256) { sdata[lid] += sdata[lid + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
            "if (lsz >= 256) { if (lid < 128) { sdata[lid] += sdata[lid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
            "if (lsz >= 128) { if (lid <  64) { sdata[lid] += sdata[lid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }\n"
            
            //Avoid extra if statements by only using local size >= 64
            "if (lid <  32) { sdata[lid] += sdata[lid +  32]; } barrier(CLK_LOCAL_MEM_FENCE);\n"
            "if (lid <  16) { sdata[lid] += sdata[lid +  16]; } barrier(CLK_LOCAL_MEM_FENCE);\n"
            "if (lid <  8) { sdata[lid] += sdata[lid +  8]; } barrier(CLK_LOCAL_MEM_FENCE);\n"
            "if (lid <  4) { sdata[lid] += sdata[lid +  4]; } barrier(CLK_LOCAL_MEM_FENCE);\n"
            "if (lid <  2) { sdata[lid] += sdata[lid +  2]; } barrier(CLK_LOCAL_MEM_FENCE);\n"
            "if (lid <  1) { sdata[lid] += sdata[lid +  1]; } barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"

        // write result for this block to global mem 
		"if (lid == 0) store_arr[store_off] = sdata[0];\n"
		"barrier(CLK_LOCAL_MEM_FENCE);\n"
	"}\n"
	
	"void global_sum_and_reduce(__local float *reduce_s,\n"
							   "__global float *reduce_arr,\n"
							   "int beg_off,\n"
							   "int arr_len)\n"
	"{\n"
		"unsigned int lsz = get_local_size(0);\n"
        "unsigned int i;"
        "float value = 0.0f;\n"
        
        "if (get_group_id(0) != 0)\n"
            "return;\n"
    
		//Reduce the entire array using one work group
        "for(i = get_local_id(0); i < arr_len; i += lsz)\n"
            "if (i < arr_len)\n"
                "value += reduce_arr[beg_off+i];\n"
    
        "sum_reduce_and_store(reduce_s, reduce_arr, value, beg_off);\n"
	"}\n"
	
	"__kernel void consolidate_train(__constant unsigned int *sizes,\n"
                                    "__global unsigned int *num_layers,\n"
                                    "__global unsigned int *num_neurons, \n"
                                    "__global unsigned int *num_inputs,\n"
                                    "__global unsigned int *num_outputs,\n"
						   
                                    "__global float *MSE_values,\n"
                                    "__global float *num_bit_fail,\n"
                                    "__global float *train_errors,\n"
                                    "__global float *weights_deltas,\n"
                        
                                    "__local float *reduce_s)\n"
    "{\n"
		"unsigned int input_sz = get_global_size(0);\n"
        "unsigned int gnum;\n"
		"unsigned int l;\n"
        
        //Calculate the number of groups used in the training run
        "if (sizes[5] %% sizes[7])\n"
            "gnum = 1 + (sizes[5] / sizes[7]);\n"
        "else\n"
            "gnum = sizes[5] / sizes[7];\n"
    
		//Calculate for all layers
		"for(l = 0; l < num_layers[get_global_id(1)]; ++l) {\n"
			"unsigned int part_layer_off = get_global_id(1)*sizes[1]+l;\n"
			"unsigned int num_neurons_l = num_neurons[part_layer_off];\n"
			"unsigned int n_layer_off = sizes[2]*part_layer_off;\n"
			"unsigned int o_layer_off = sizes[4]*part_layer_off;\n"
			"unsigned int n;\n"
			
			//Calcalate for all neurons
			"for(n = 0; n < num_neurons[part_layer_off]; ++n) {\n"
				"unsigned int num_outputs_l = num_outputs[n_layer_off+n];\n"
				"unsigned int num_inputs_l  = num_inputs[n_layer_off+n];\n"
				"unsigned int o;\n"
				
				//Calculate for all outputs
				"for(o = 0; o < num_outputs_l; ++o) {\n"
					"unsigned int i;\n"
	
					//Sum delta data
					"for(i = 0; i < num_inputs_l; ++i) {\n"
						"global_sum_and_reduce(reduce_s, weights_deltas,\n"
											  "((o_layer_off+o)*sizes[3]+i)*gnum, gnum);\n"
//    "if(get_global_id(0) == 0)\n"
//    "printf(\"d (l, n, o, i) (delta): (%%5d %%5d %%5d %%5d) (%%5d %%10f)\\n\", l, n, o, i, ((o_layer_off+o)*sizes[3]+i)*gsz, weights_deltas[((o_layer_off+o)*sizes[3]+i)*gsz]);\n"
					"}\n"
    
					"global_sum_and_reduce(reduce_s, train_errors,\n"
										  "(o_layer_off+o)*sizes[5], sizes[5]);\n"
//    "printf(\"e (l, n, o) (errs): (%%5d %%5d %%5d) (%%5d %%10f)\\n\", l, n, o, (o_layer_off+o)*input_sz, train_errors[(o_layer_off+o)*input_sz]);\n"
				"}\n"
                "o_layer_off += num_outputs_l;\n"
			"}\n"
		"}\n"

		"global_sum_and_reduce(reduce_s, MSE_values, get_global_id(1)*sizes[5], sizes[5]);\n"
		"global_sum_and_reduce(reduce_s, num_bit_fail, get_global_id(1)*sizes[5], sizes[5]);\n"

//    "printf(\"m (msev): (%%10f)\\n\", MSE_values[get_global_id(1)*input_sz]);\n"
//    "printf(\"n (fail): (%%10f)\\n\", num_bit_fail[get_global_id(1)*input_sz]);\n"
	"}\n"
	
	"float activation_derived(float steepness, int act_func,\n"
                            "__global float *outputs,\n"
                            "__global float *sums, int o_i)\n"
	"{\n"
		"switch (act_func)\n"
		"{\n"
			"case %d:\n"
			"case %d:\n"
			"case %d:\n"
				"return " QUOTEME(fann_linear_derive(steepness, outputs[o_i])) ";\n"
			"case %d:\n"
			"case %d:\n"
				"return " QUOTEME(fann_sigmoid_derive(steepness, fann_clip(outputs[o_i], 0.01f, 0.99f))) ";\n"
			"case %d:\n"
			"case %d:\n"
				"return " QUOTEME(fann_sigmoid_symmetric_derive(steepness, fann_clip(outputs[o_i], -0.98f, 0.98f))) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_gaussian_derive(steepness, outputs[o_i], sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_gaussian_symmetric_derive(steepness, outputs[o_i], sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_elliot_derive(steepness, fann_clip(outputs[o_i], 0.01f, 0.99f), sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_elliot_symmetric_derive(steepness, fann_clip(outputs[o_i], -0.98f, 0.98f), sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_sin_symmetric_derive(steepness, sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_cos_symmetric_derive(steepness, sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_sin_derive(steepness, sums[o_i])) ";\n"
			"case %d:\n"
				"return " QUOTEME(fann_cos_derive(steepness, sums[o_i])) ";\n"
			"case %d: //This should be an error\n"
			"case %d: //This should be an error\n"
			"case %d: //FIXME\n"
				"return -99.0;\n"
		"}\n"
	"}\n"
	
	"void backpropagate_MSE(__constant unsigned int *sizes,\n"
						   "__global unsigned int *num_layers,\n"
						   "__global unsigned int *num_neurons, \n"
						   "__global unsigned int *num_inputs,\n"
						   "__global unsigned int *num_outputs,\n"
						 
						   "__global float *steepness,\n"
						   "__global int *activation,\n"
                           "__global float *weights,\n"
						 
						   "__global float *inputs,\n"
                           "__global float *sums,\n"
						   "__global float *outputs,\n"
						   "__global float *train_errors,\n"
						   "__global float *weights_deltas,\n"
						 
						   //Shared areas for caching
						   "__local float *steep_s,\n"
						   "__local int *act_s,\n"
						   "__local float *weights_s,\n"
						   "__local float *reduce_s )\n"
	"{\n"
		"unsigned int input_id = get_global_id(0);\n"
		"unsigned int lid = get_local_id(0);\n"
		"unsigned int lsz = get_local_size(0);\n"
		"unsigned int gnum;\n"
		"unsigned int gid = get_group_id(0);\n"
		"int l;\n"
    
        //Calculate the number of groups used in the training run
        "if (sizes[5] %% sizes[7])\n"
            "gnum = 1 + (sizes[5] / sizes[7]);\n"
        "else\n"
            "gnum = sizes[5] / sizes[7];\n"
    
        //Calculate for all layers
		"for(l = num_layers[get_global_id(1)]-1; l >= 0; --l) {\n"
			"unsigned int part_layer_off = get_global_id(1)*sizes[1]+l;\n"
			"unsigned int num_neurons_l = num_neurons[part_layer_off];\n"
			"unsigned int n_layer_off = sizes[2]*part_layer_off;\n"
			"unsigned int o_layer_off = sizes[4]*part_layer_off;\n"
			"unsigned int output_off = o_layer_off-sizes[4];\n"
			"unsigned int n;\n"

            //Copy steepness & activation to shared mem
			"barrier(CLK_LOCAL_MEM_FENCE);\n"
			"for(n = 0; n < num_neurons_l; n += lsz) {\n"
				"unsigned int neuron_num = n+lid;\n"
				"if (neuron_num < num_neurons[part_layer_off]){\n"
					"steep_s[neuron_num] = steepness[n_layer_off+neuron_num];\n"
					"act_s[neuron_num] = activation[n_layer_off+neuron_num];\n"
				"}\n"
			"}\n"
			"barrier(CLK_LOCAL_MEM_FENCE);\n"

            //Clear all the previous layer's train_errors
			"for(n = 0; n < num_neurons_l && l != 0; ++n) {\n"
				"unsigned int num_outputs_l = num_outputs[n_layer_off+n];\n"
				"unsigned int o;\n"
				
				//Zero all outputs
				"for(o = 0; o < num_outputs_l; ++o) {\n"
					//Don't overrun data
					"if (sizes[5] > input_id)\n"
						"train_errors[output_off*sizes[5]+input_id] = 0.0f;\n"
					"++output_off;\n"
				"}\n"
			"}\n"
			
			//Reset
			"output_off = o_layer_off;\n"
			
			//Calcalate for all neurons
			"for(n = 0; n < num_neurons[part_layer_off]; ++n) {\n"
				"unsigned int num_outputs_l = num_outputs[n_layer_off+n];\n"
				"unsigned int num_inputs_l  = num_inputs[n_layer_off+n];\n"
				"unsigned int o;\n"
				
				//Calculate for all outputs
				"for(o = 0; o < num_outputs_l; ++o) {\n"
					"unsigned int i;\n"
					"unsigned int o_i = output_off*sizes[5]+input_id;\n"
					"float error;\n"
					
					//Multiply errors with the activation function derivative.
					"if (sizes[5] > input_id)\n"
						"train_errors[o_i] = error =\n"
							"train_errors[o_i]*activation_derived(steep_s[n], act_s[n], outputs, sums, o_i);\n"
					
					//Weight & sum data from the inputs & bias
					"for(i = 0; i < num_inputs_l; ++i) {\n"
						"unsigned int weights_i = 0;\n"
                        "unsigned int prev_output_i = 0;\n"
						"float delta = 0.0f;\n"
						
						//Weights aren't used for first layer
						"if (l != 0) {\n"
							"weights_i = (sizes[3]*o+i) %% lsz;\n"
							
							//Load shared memory as appropriate
							"if (weights_i == 0) {\n"
								"barrier(CLK_LOCAL_MEM_FENCE);\n"
								"if (sizes[3]*o+i+lid < sizes[3]*num_outputs_l)\n"
									"weights_s[lid] = weights[output_off*sizes[3]+i+lid];\n"
								"barrier(CLK_LOCAL_MEM_FENCE);\n"
							"}\n"
						"}\n"
						
						//Don't overrun data
						"if (sizes[5] > input_id) {\n"
							//Figure out what input was used
							"if(i == num_inputs_l-1){\n"
								"prev_output_i = (o_layer_off-sizes[4]+i)*sizes[5]+input_id;\n"
								"delta = error;\n"
							"} else if(l == 0) {\n"
								"delta = inputs[i*sizes[5]+input_id] * error;\n"
							"} else {\n"
								"prev_output_i = (o_layer_off-sizes[4]+i)*sizes[5]+input_id;\n"
								"delta = outputs[prev_output_i] * error;\n"
							"}\n"
						"}\n"
    
//    "printf(\"(id, l, n, o, i) (delta): (%%5d %%5d %%5d %%5d %%5d) (%%10f)\\n\", input_id, l, n, o, i, error*input);\n"
						
						// Calculate the weight deltas
						// Due to memory requirements we're reducing the deltas here
						"sum_reduce_and_store(reduce_s, weights_deltas, delta,\n"
											 "(output_off*sizes[3]+i)*gnum+gid);\n"
//    "weights_deltas[(output_off*sizes[3]+i)*gnum+gid] = gnum;\n"
//    "printf(\"(id, l, n, o, i) (fin delta): (%%5d %%5d %%5d %%5d %%5d) (%%10f)\\n\", input_id, l, n, o, i, weights_deltas[(output_off*sizes[3]+i)*gnum+gid]);\n"
    
						//Calculate the error for previous layer
						"if(l != 0 && sizes[5] > input_id)\n"
							"train_errors[prev_output_i] += error * weights_s[weights_i];\n"
    
					"}\n"
					"++output_off;\n"
				"}\n"
			"}\n"
		"}\n"
	"}\n"
	
	"void compute_MSE(__constant unsigned int *sizes,\n"
                     "__global float *f_params,\n"
					 "__global unsigned int *num_layers,\n"
					 "__global unsigned int *num_neurons, \n"
					 "__global unsigned int *num_outputs,\n"
					 
					 "__global int *activation,\n"
					 
					 "__global float *outputs,\n"
					 "__global float *train_errors,\n"
					 "__global float *actual_outputs,\n"
					 "__global float *MSE_values,\n"
					 "__global float *num_bit_fail,\n"
	
					 "__local int *act_s)\n"
	"{\n"
		"unsigned int ann_id = get_global_id(1);\n"
		"unsigned int input_id = get_global_id(0);\n"
		"unsigned int out_neuron_index = ann_id*sizes[1]+num_layers[ann_id]-1;\n"
		"unsigned int out_off = out_neuron_index*sizes[4]*sizes[5]+input_id;\n"
		"unsigned int num_neurons_l = num_neurons[out_neuron_index];\n"
		"unsigned int layer_off = sizes[2]*out_neuron_index;\n"
		"unsigned int n;\n"
		
		"unsigned int layer_o = 0;\n"
		"unsigned int bit_fail = 0;\n"
		"float MSE_value = 0.0f;\n"
		
		//Copy steepness & activation to shared mem
		"for(n = 0; n < num_neurons_l; n += get_local_size(0)) {\n"
			"unsigned int neuron_off = n+get_local_id(0);\n"
			"if (neuron_off < num_neurons_l)\n"
				"act_s[neuron_off] = activation[layer_off+neuron_off];\n"
		"}\n"
		"barrier(CLK_LOCAL_MEM_FENCE);\n"
		
		//Don't use the extra threads
		"if (input_id >= sizes[5])\n"
			"return;\n"
		
		//Calcalate for all neurons
		"for(n = 0; n < num_neurons_l; ++n) {\n"
			"unsigned int num_outputs_l = num_outputs[layer_off+n];\n"
			"unsigned int act_out_off = (ann_id*sizes[4]+layer_o)*sizes[5]+input_id;\n"
			"unsigned int o;\n"
			
			//Calculate for all outputs
			"for(o = 0; o < num_outputs_l; ++o) {\n"
				"unsigned int out_index = out_off+(layer_o+o)*sizes[5];\n"
				"float neuron_diff = actual_outputs[act_out_off+o*sizes[5]] - outputs[out_index];\n"
    
                //Update MSE macro follows
                "switch (act_s[n]) {\n"
                    "case %d:\n"
                    "case %d:\n"
                    "case %d:\n"
                    "case %d:\n"
                    "case %d:\n"
                    "case %d:\n"
                    "case %d:\n"
                    "case %d:\n"
                        "neuron_diff *= 0.5f;\n"
                "}\n"

                "MSE_value += neuron_diff * neuron_diff;\n"
                "if(fabs(neuron_diff) >= f_params[0])\n"
                    "++bit_fail;\n"
//    "printf(\"neuron_diff MSE_value: %%10f %%10f\\n\", neuron_diff, MSE_value);\n"
				//Update error
				"if (sizes[9]) {\n"
					"if(neuron_diff < -.9999999f)\n"
						"neuron_diff = -17.0f;\n"
					"else if(neuron_diff > .9999999f)\n"
						"neuron_diff = 17.0f;\n"
					"else\n"
						"neuron_diff = log((1.0f + neuron_diff) / (1.0f - neuron_diff));\n"
				"}\n"

//        "printf(\"train_error out_index: %%10f %%5d\\n\", neuron_diff, out_index);\n"
				"train_errors[out_index] = neuron_diff;\n"
				//Don't update ann->training_params->num_MSE because it can be calculated later
//    "printf(\"(%%5d %%5d %%5d) train_errors actual_output neuron_value: %%10f %%10f %%10f\\n\", input_id, n, o, train_errors[out_index], actual_outputs[act_out_off+o*sizes[5]], outputs[out_index]);\n"
			"}\n"
			
			"layer_o += num_outputs_l;\n"
		"}\n"
		
		"unsigned int net_index = ann_id*sizes[5]+input_id;\n"
		"num_bit_fail[net_index] = bit_fail;\n"
		"MSE_values[net_index] = MSE_value;\n"
	"}\n"
	
    "float calc_act(float sum, float steepness, int act)\n"
    "{\n"
        "float max_sum;\n"
        "sum *= steepness;\n"
        
        "max_sum = 150.0f/steepness;\n"
        "if(sum > max_sum)\n"
            "sum = max_sum;\n"
        "else if(sum < -max_sum)\n"
            "sum = -max_sum;\n"
        
        "switch(act)\n"
        "{\n"
            "case %d:\n"
                "return sum;\n"
            "case %d:\n"
                "return ((sum < 0.0f) ? 0.0f : (sum > 1.0f) ? 1.0f : sum);\n"
            "case %d:\n"
                "return ((sum < -1.0f) ? -1.0f : (sum > 1.0f) ? 1.0f : sum);\n"
            "case %d:\n"
                "return " QUOTEME(fann_sigmoid_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_sigmoid_symmetric_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_stepwise(-2.64665293693542480469e+00f, -1.47221934795379638672e+00f, -5.49306154251098632812e-01f, 5.49306154251098632812e-01f, 1.47221934795379638672e+00f, 2.64665293693542480469e+00f, -9.90000009536743164062e-01f, -8.99999976158142089844e-01f, -5.00000000000000000000e-01f, 5.00000000000000000000e-01f, 8.99999976158142089844e-01f, 9.90000009536743164062e-01f, -1.0f, 1.0f, sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_stepwise(-2.64665246009826660156e+00f, -1.47221946716308593750e+00f, -5.49306154251098632812e-01f, 5.49306154251098632812e-01f, 1.47221934795379638672e+00f, 2.64665293693542480469e+00f, 4.99999988824129104614e-03f, 5.00000007450580596924e-02f, 2.50000000000000000000e-01f, 7.50000000000000000000e-01f, 9.49999988079071044922e-01f, 9.95000004768371582031e-01f, 0.0f, 1.0f, sum)) ";\n"
            "case %d:\n"
                "return ((sum < 0.0f) ? 0.0f : 1.0f);\n"
            "case %d:\n"
                "return ((sum < 0.0f) ? -1.0f : 1.0f);\n"
            "case %d:\n"
                "return " QUOTEME(fann_gaussian_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_gaussian_symmetric_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_elliot_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_elliot_symmetric_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_sin_symmetric_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_cos_symmetric_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_sin_real(sum)) ";\n"
            "case %d:\n"
                "return " QUOTEME(fann_cos_real(sum)) ";\n"
            "case %d:\n"
                "return 0;\n"
        "}\n"
    "}\n"
    
    "__kernel void run(__constant unsigned int *sizes,\n"
                        "__global float *f_params,\n"
                        "__global unsigned int *num_layers,\n"
                        "__global unsigned int *num_neurons,\n"
                        "__global unsigned int *num_inputs,\n"
                        "__global unsigned int *num_outputs,\n"
                        
                        "__global float *steepness,\n"
                        "__global int *activation,\n"
                        "__global float *weights,\n"
                        
                        "__global float *inputs,\n"
                        "__global float *sums,\n"
                        "__global float *outputs,\n"
                        
                        "__local float *steep_s,\n"
                        "__local int *act_s,\n"
                        "__local float *weights_s )\n"
    "{\n"
        "unsigned int input_id = get_global_id(0);\n"
        "unsigned int ann_id = get_global_id(1);\n"
        "unsigned int lid = get_local_id(0);\n"
        "unsigned int lsz = get_local_size(0);\n"
        "unsigned int l;\n"
        
        //Calculate for all layers
        "for(l = 0; l < num_layers[ann_id]; ++l) {\n"
            "unsigned int part_layer_off = ann_id*sizes[1]+l;\n"
            "unsigned int n;\n"
            "unsigned int num_neurons_l = num_neurons[part_layer_off];\n"
            "unsigned int n_layer_off = sizes[2]*part_layer_off;\n"
            "unsigned int o_layer_off = sizes[4]*part_layer_off;\n"
            "unsigned int output_off = o_layer_off;\n"
    
            //Copy steepness & activation to shared mem
            "barrier(CLK_LOCAL_MEM_FENCE);\n"
            "for(n = 0; n < num_neurons[part_layer_off]; n += lsz) {\n"
                "unsigned int neuron_num = n+lid;\n"
                "if (neuron_num < num_neurons[part_layer_off]){\n"
                    "steep_s[neuron_num] = steepness[n_layer_off+neuron_num];\n"
                    "act_s[neuron_num] = activation[n_layer_off+neuron_num];\n"
                "}\n"
            "}\n"
            "barrier(CLK_LOCAL_MEM_FENCE);\n"
    
            //Calcalate for all neurons
            "for(n = 0; n < num_neurons[part_layer_off]; ++n) {\n"
                "unsigned int num_outputs_l = num_outputs[n_layer_off+n];\n"
                "unsigned int num_inputs_l  = num_inputs[n_layer_off+n];\n"
                "unsigned int o;\n"
                
                //Calculate for all outputs
                "for(o = 0; o < num_outputs_l; ++o) {\n"
                    "unsigned int i;\n"
                    "float sum = 0.0f;\n"
                    
                    //Weight & sum data from the inputs & bias
                    "for(i = 0; i < num_inputs_l; ++i) {\n"
                        "float in_val;\n"
                        "unsigned int weights_i = (sizes[3]*o+i) %% lsz;\n"
                    
                        //Load shared memory as appropriate
                        "if (weights_i == 0) {\n"
                            "barrier(CLK_LOCAL_MEM_FENCE);\n"
                            "if (sizes[3]*o+i+lid < sizes[3]*num_outputs_l)\n"
                                "weights_s[lid] = weights[output_off*sizes[3]+i+lid];\n"
                            "barrier(CLK_LOCAL_MEM_FENCE);\n"
                        "}\n"
                        
                        //Don't overrun data
                        "if (sizes[5] <= input_id)\n"
                            "continue;\n"
                        
                        "if (i == num_inputs_l-1) {\n"
                            //Bias
                            "in_val = 1.0f;\n"
//    "printf(\"%%5d %%2d %%2d %%2d %%2d: %%15fCL %%15fCL\\n\", input_id, l, n, o ,i, weights_s[weights_i], in_val);\n"
                        "} else if (l == 0) {\n"
                            //Handle input from user
                            "in_val = inputs[i*sizes[5]+input_id];\n"
//    "printf(\"%%5d %%2d %%2d %%2d %%2d: %%15fCL %%15fCL\\n\", input_id, l, n, o ,i, weights_s[weights_i], in_val);\n"
                        "} else {\n"
                            //Handle input from neurons
                            "in_val = outputs[(o_layer_off-sizes[4]+i)*sizes[5]+input_id];\n"
//    "printf(\"%%5d %%2d %%2d %%2d %%2d: %%15fCL %%15fCL N:%%5d\\n\", input_id, l, n, o ,i, weights_s[weights_i], in_val, (o_layer_off-sizes[4]+i)*sizes[5]+input_id);\n"
                        "}\n"
                        
                        //Weight & sum
                        "sum += weights_s[weights_i]*in_val;\n"
                    "}\n"
                    
                    //Don't overrun data
                    "if (sizes[5] > input_id){\n"
                        //Save into output data array
                        "sums[output_off*sizes[5]+input_id] = sum;\n"
                        "outputs[output_off*sizes[5]+input_id] = calc_act(sum, steep_s[n], act_s[n]);\n"
//    "printf(\"%%5d %%2d %%2d %%2d %%2d: %%15f %%15f N:%%5d\\n\", input_id, l, n, o ,i, sums[output_off*sizes[5]+input_id], outputs[output_off*sizes[5]+input_id], output_off*sizes[5]+input_id);\n"
                    "}\n"
    
                    "++output_off;\n"
                "}\n"
            "}\n"
        "}\n"
    "}"
	
	/*
+	 remember to set  ann->training_params->num_MSE on return
+	 remember to init train errors like in fann_compute_MSE()
+	 remember that the group size must be a power of two >= 64 or == 1 for reduce to work
	 remember to set neuron->num_backprop_done on return
	 */
	
	"__kernel void train_batch(\n"//ANN structure
								"__constant unsigned int *sizes,\n"
                                "__global float *f_params,\n"
								"__global unsigned int *num_layers,\n"
								"__global unsigned int *num_neurons,\n"
								"__global unsigned int *num_inputs,\n"
								"__global unsigned int *num_outputs,\n"
								
								//Network values
								"__global float *steepness,\n"
								"__global int *activation,\n"
								"__global float *weights,\n"
								
								//Per-run data
								"__global float *inputs,\n"
								"__global float *sums,\n"
								"__global float *outputs,\n"
							  
								//Per-run training memory
								"__global float *train_errors,\n"
								"__global float *actual_outputs,\n"
								"__global float *MSE_values,\n"
								"__global float *num_bit_fail,\n"
                                "__global float *weights_deltas,\n"
								
								//Shared areas
								"__local float *steep_s,\n"
								"__local int *act_s,\n"
								"__local float *weights_s,\n"
								"__local float *reduce_s)\n"
	"{\n"
		//Do the normal procedure of an epoch
		"run(sizes, f_params, num_layers, num_neurons, num_inputs, num_outputs,\n"
			"steepness, activation, weights, inputs, sums, outputs,\n"
			"steep_s, act_s, weights_s);\n"
		
		"compute_MSE(sizes, f_params, num_layers, num_neurons, num_outputs,\n"
					"activation, outputs, train_errors, actual_outputs,\n"
					"MSE_values, num_bit_fail, act_s);\n"
		
		"backpropagate_MSE(sizes, num_layers, num_neurons, num_inputs, num_outputs,\n"
						  "steepness, activation, weights, inputs, sums,\n"
                          "outputs, train_errors, weights_deltas,\n"
						  "steep_s, act_s, weights_s, reduce_s);\n"
	"}\n";
    
    //Insert enum values here because I can't seem to do it at compile time
    fin_program_src = calloc(128000, sizeof(char));
    sprintf(fin_program_src, program_source,
            
            FANN_LINEAR, FANN_LINEAR_PIECE,
    		FANN_LINEAR_PIECE_SYMMETRIC, FANN_SIGMOID, FANN_SIGMOID_STEPWISE,
    		FANN_SIGMOID_SYMMETRIC, FANN_SIGMOID_SYMMETRIC_STEPWISE, FANN_GAUSSIAN,
    		FANN_GAUSSIAN_SYMMETRIC, FANN_ELLIOT, FANN_ELLIOT_SYMMETRIC,
    		FANN_SIN_SYMMETRIC, FANN_COS_SYMMETRIC, FANN_SIN, FANN_COS,
    		FANN_THRESHOLD_SYMMETRIC, FANN_THRESHOLD, FANN_GAUSSIAN_STEPWISE,
            
            FANN_LINEAR_PIECE_SYMMETRIC, FANN_THRESHOLD_SYMMETRIC,
            FANN_SIGMOID_SYMMETRIC, FANN_SIGMOID_SYMMETRIC_STEPWISE,
            FANN_ELLIOT_SYMMETRIC, FANN_GAUSSIAN_SYMMETRIC, FANN_SIN_SYMMETRIC,
            FANN_COS_SYMMETRIC,
            
    		FANN_LINEAR, FANN_LINEAR_PIECE,
            FANN_LINEAR_PIECE_SYMMETRIC, FANN_SIGMOID, FANN_SIGMOID_SYMMETRIC,
            FANN_SIGMOID_SYMMETRIC_STEPWISE, FANN_SIGMOID_STEPWISE,
            FANN_THRESHOLD, FANN_THRESHOLD_SYMMETRIC, FANN_GAUSSIAN,
            FANN_GAUSSIAN_SYMMETRIC, FANN_ELLIOT, FANN_ELLIOT_SYMMETRIC,
            FANN_SIN_SYMMETRIC, FANN_COS_SYMMETRIC, FANN_SIN, FANN_COS,
            FANN_GAUSSIAN_STEPWISE);
    
    program = clCreateProgramWithSource(context, 1, (const char**)&fin_program_src,
                                        NULL, &err);
    assert(err == CL_SUCCESS);
    
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	//"-cl-fast-relaxed-math -Werror"
    //Detailed debugging info
    if (err != CL_SUCCESS)
    {
        size_t len;
        char *buffer = (char *)calloc(128000, sizeof(char));
        
        printf("Error: Failed to build program executable!\n");
        printf("clBuildProgram return:\n");
        if(err == CL_INVALID_PROGRAM)
            printf("CL_INVALID_PROGRAM\n");
        else if(err == CL_INVALID_VALUE)
            printf("CL_INVALID_VALUE\n");
        else if(err == CL_INVALID_BINARY)
            printf("CL_INVALID_BINARY\n");
        else if(err == CL_INVALID_BUILD_OPTIONS)
            printf("CL_INVALID_BUILD_OPTIONS\n");
        else if(err == CL_INVALID_OPERATION)
            printf("CL_INVALID_OPERATION\n");
        else if(err == CL_COMPILER_NOT_AVAILABLE)
            printf("CL_COMPILER_NOT_AVAILABLE\n");
        else if(err == CL_BUILD_PROGRAM_FAILURE)
            printf("CL_BUILD_PROGRAM_FAILURE\n");
        else if(err == CL_OUT_OF_HOST_MEMORY)
            printf("CL_OUT_OF_HOST_MEMORY\n");
        
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 128000*sizeof(char), buffer, &len);
        assert(err == CL_SUCCESS);
        
        printf("Build Status:\n");
        if(buffer[0] == CL_BUILD_NONE)
            printf("CL_BUILD_NONE\n");
        else if(buffer[0] == CL_BUILD_ERROR)
            printf("CL_BUILD_ERROR\n");
        else if(buffer[0] == CL_BUILD_SUCCESS)
            printf("CL_BUILD_SUCCESS\n");
        else if(buffer[0] == CL_BUILD_IN_PROGRESS)
            printf("CL_BUILD_IN_PROGRESS\n");
        
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 128000*sizeof(char), buffer, &len);
        printf("Get Build Info:\n");
        switch (err) {
            case CL_INVALID_DEVICE:
                printf("CL_INVALID_DEVICE\n"); break;
            case CL_INVALID_VALUE:
                printf("CL_INVALID_VALUE\n"); break;
            case CL_INVALID_PROGRAM:
                printf("CL_INVALID_PROGRAM\n"); break;
        }
        
        printf("Build Info:\n%s\nProgram Source:\n%s\n", buffer, fin_program_src);
        
        free(buffer);
        exit(1);
    }
    
    kernel = clCreateKernel(program, kern_name, &err);
    clReleaseProgram(program);
    
    free(fin_program_src);
    return kernel;
}
