/*
 * Fast Artificial Neural Network Library (fann) Copyright (C) 2003
 * Steffen Nissen (lukesky@diku.dk)
 * 
 * This library is free software; you can redistribute it and/or modify it 
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA 
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "config.h"
#include "fann.h"

/*
 * Reads training data from a file. 
 */
FANN_EXTERNAL struct fann_train_data *FANN_API fann_read_train_from_file(const char *configuration_file)
{
	struct fann_train_data *data;
	FILE *file;
	file = fopen(configuration_file, "r");

	if(!file)
	{
		fann_error(NULL, FANN_E_CANT_OPEN_CONFIG_R, configuration_file);
		return NULL;
	}

	data = fann_read_train_from_fd(file, configuration_file);
	fclose(file);
	return data;
}

/*
 * Save training data to a file
 */
FANN_EXTERNAL int FANN_API fann_save_train(struct fann_train_data *data, const char *filename)
{
	return fann_save_train_internal(data, filename, 0, 0);
}

/*
 * Save training data to a file in fixed point algebra. (Good for testing
 * a network in fixed point)
 */
FANN_EXTERNAL int FANN_API fann_save_train_to_fixed(struct fann_train_data *data, const char *filename,
													 unsigned int decimal_point)
{
	return fann_save_train_internal(data, filename, 1, decimal_point);
}

/*
 * deallocate the train data structure.
 */
FANN_EXTERNAL void FANN_API fann_destroy_train(struct fann_train_data *data)
{
	if(data == NULL)
		return;
	if(data->input != NULL)
		fann_safe_free(data->input[0]);
	if(data->output != NULL)
		fann_safe_free(data->output[0]);
	fann_safe_free(data->input);
	fann_safe_free(data->output);
	fann_safe_free(data);
}

/*
 * Test a set of training data and calculate the MSE
 */
FANN_EXTERNAL float FANN_API fann_test_data(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;
	if(fann_check_input_output_sizes(ann, data) == -1)
		return 0;

	fann_reset_MSE(ann);

	for(i = 0; i != data->num_data; i++)
	{
		fann_test(ann, data->input[i], data->output[i]);
	}

	return fann_get_MSE(ann);
}

#ifndef FIXEDFANN

#if 0
/*
 * Internal train function
 */
float fann_train_epoch_quickprop(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;

	if(ann->rprop_params->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	fann_reset_MSE(ann);

	for(i = 0; i < data->num_data; i++)
	{
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}
	fann_update_weights_quickprop(ann, data->num_data, 0, ann->total_connections);

	return fann_get_MSE(ann);
}

/*
 * Internal train function 
 */
float fann_train_epoch_irpropm(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;

	if(ann->rprop_params->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	fann_reset_MSE(ann);

	for(i = 0; i < data->num_data; i++)
	{
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}

	fann_update_weights_irpropm(ann, 0, ann->total_connections);

	return fann_get_MSE(ann);
}

/*
 * Internal train function 
 */
float fann_train_epoch_sarprop(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;

	if(ann->rprop_params->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	fann_reset_MSE(ann);

	for(i = 0; i < data->num_data; i++)
	{
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_backpropagate_MSE(ann);
		fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
	}

	fann_update_weights_sarprop(ann, ann->rprop_params->sarprop_epoch, 0, ann->total_connections);

	++(ann->rprop_params->sarprop_epoch);

	return fann_get_MSE(ann);
}
#endif
/*
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
                
                printf("   sums, outputs, train_errors: %10f, %10f, %10f\n",
                       neuron_it->sums[n_output_num],
                       neuron_it->outputs[n_output_num],
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
} */

/*
 * Internal train function 
 */
float fann_train_epoch_batch(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;
	
	fann_reset_MSE(ann);

	for(i = 0; i < data->num_data; i++)
	{
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_backpropagate_MSE(ann);
		/*fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);*/
	}
    
//    dump_train_vals(ann);

	fann_update_weights(ann);

    return fann_get_MSE(ann);
}

/*
 * Internal train function 
 */
float fann_train_epoch_incremental(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;

	fann_reset_MSE(ann);

	for(i = 0; i != data->num_data; i++)
	{
		fann_train(ann, data->input[i], data->output[i]);
	}

	return fann_get_MSE(ann);
}

/*
 *  Train the SOM by going through the data only one time
 */
float fann_train_epoch_som(struct fann *ann, struct fann_train_data *data)
{
	fann_train_on_data_som(ann, data, data->num_data, data->num_data, 0.0f);
	return fann_get_MSE_som(ann, data);
}

/*
 *  Train the GNG by going through the data only one time
 */
float fann_train_epoch_gng(struct fann *ann, struct fann_train_data *data)
{
	fann_train_on_data_gng(ann, data, data->num_data, data->num_data, 0.0f);
	return fann_get_MSE_gng(ann, data);
}

/*
 * Train for one epoch with the selected training algorithm 
 */
FANN_EXTERNAL float FANN_API fann_train_epoch(struct fann *ann, struct fann_train_data *data)
{
	switch (ann->training_params->training_algorithm)
	{
	case FANN_TRAIN_SOM:
		return fann_train_epoch_som(ann, data);
	case FANN_TRAIN_GNG:
		return fann_train_epoch_gng(ann, data);
	case FANN_TRAIN_INCREMENTAL:
		if(fann_check_input_output_sizes(ann, data) == -1)
			return 0;
		return fann_train_epoch_incremental(ann, data);
	case FANN_TRAIN_SARPROP:
	case FANN_TRAIN_RPROP:
	case FANN_TRAIN_QUICKPROP:
	case FANN_TRAIN_BATCH:
		if(fann_check_input_output_sizes(ann, data) == -1)
			return 0;
		return fann_train_epoch_batch(ann, data);
	}
	return 0;
}

FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error)
{
	float error;
	unsigned int i;
	int desired_error_reached;

#ifdef DEBUG
	printf("Training with %s\n", FANN_TRAIN_NAMES[ann->training_params->training_algorithm]);
#endif

	if (ann->network_type == FANN_NETTYPE_SOM)
	{
		fann_train_on_data_som(ann, data, max_epochs, epochs_between_reports, desired_error);
		return;
	}
	else if (ann->network_type == FANN_NETTYPE_GNG)
	{
		fann_train_on_data_gng(ann, data, max_epochs, epochs_between_reports, desired_error);
		return;
	}

	if(epochs_between_reports && ann->training_params->callback == NULL)
		printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);

	for(i = 1; i <= max_epochs; i++)
	{
		/*
		 * train 
		 */
		error = fann_train_epoch(ann, data);
		desired_error_reached = fann_desired_error_reached(ann, desired_error);
        
		/*
		 * print current output 
		 */
		if(epochs_between_reports &&
		   (i % epochs_between_reports == 0 || i == max_epochs || i == 1 ||
			desired_error_reached == 0))
		{
			if(ann->training_params->callback == NULL)
			{
				printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error,
					   ann->training_params->num_bit_fail);
			}
			else if(((*ann->training_params->callback)(ann, data, max_epochs,
                                                       epochs_between_reports, 
                                                       desired_error, i)) == -1)
			{
				/*
				 * you can break the training by returning -1 
				 */
				break;
			}
		}

		if(desired_error_reached == 0)
			break;
	}
}

FANN_EXTERNAL void FANN_API fann_train_on_file(struct fann *ann, const char *filename,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error)
{
	struct fann_train_data *data = fann_read_train_from_file(filename);

	if(data == NULL)
	{
		return;
	}
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
	fann_destroy_train(data);
}

#endif

/*
 * shuffles training data, randomizing the order 
 */
FANN_EXTERNAL void FANN_API fann_shuffle_train_data(struct fann_train_data *train_data)
{
	unsigned int dat = 0, elem, swap;
	fann_type temp;

	for(; dat < train_data->num_data; dat++)
	{
		swap = (unsigned int) (rand() % train_data->num_data);
		if(swap != dat)
		{
			for(elem = 0; elem < train_data->num_input; elem++)
			{
				temp = train_data->input[dat][elem];
				train_data->input[dat][elem] = train_data->input[swap][elem];
				train_data->input[swap][elem] = temp;
			}
			for(elem = 0; elem < train_data->num_output; elem++)
			{
				temp = train_data->output[dat][elem];
				train_data->output[dat][elem] = train_data->output[swap][elem];
				train_data->output[swap][elem] = temp;
			}
		}
	}
}

/*
 * INTERNAL FUNCTION Scales data to a specific range 
 */
void fann_scale_data(fann_type ** data, unsigned int num_data, unsigned int num_elem,
					 fann_type new_min, fann_type new_max)
{
	unsigned int dat, elem;
	fann_type old_min, old_max, temp, old_span, new_span, factor;

	old_min = old_max = data[0][0];

	/*
	 * first calculate min and max 
	 */
	for(dat = 0; dat < num_data; dat++)
	{
		for(elem = 0; elem < num_elem; elem++)
		{
			temp = data[dat][elem];
			if(temp < old_min)
				old_min = temp;
			else if(temp > old_max)
				old_max = temp;
		}
	}

	old_span = old_max - old_min;
	new_span = new_max - new_min;
	factor = new_span / old_span;
	/*printf("max %f, min %f, factor %f\n", old_max, old_min, factor);*/

	for(dat = 0; dat < num_data; dat++)
	{
		for(elem = 0; elem < num_elem; elem++)
		{
			temp = (data[dat][elem] - old_min) * factor + new_min;
			if(temp < new_min)
			{
				data[dat][elem] = new_min;
				/*
				 * printf("error %f < %f\n", temp, new_min); 
				 */
			}
			else if(temp > new_max)
			{
				data[dat][elem] = new_max;
				/*
				 * printf("error %f > %f\n", temp, new_max); 
				 */
			}
			else
			{
				data[dat][elem] = temp;
			}
		}
	}
}

/*
 * Scales the inputs in the training data to the specified range 
 */
FANN_EXTERNAL void FANN_API fann_scale_input_train_data(struct fann_train_data *train_data,
														fann_type new_min, fann_type new_max)
{
	fann_scale_data(train_data->input, train_data->num_data, train_data->num_input, new_min,
					new_max);
}

/*
 * Scales the inputs in the training data to the specified range 
 */
FANN_EXTERNAL void FANN_API fann_scale_output_train_data(struct fann_train_data *train_data,
														 fann_type new_min, fann_type new_max)
{
	fann_scale_data(train_data->output, train_data->num_data, train_data->num_output, new_min,
					new_max);
}

/*
 * Scales the inputs in the training data to the specified range 
 */
FANN_EXTERNAL void FANN_API fann_scale_train_data(struct fann_train_data *train_data,
												  fann_type new_min, fann_type new_max)
{
	fann_scale_data(train_data->input, train_data->num_data, train_data->num_input, new_min,
					new_max);
	fann_scale_data(train_data->output, train_data->num_data, train_data->num_output, new_min,
					new_max);
}

/*
 * merges training data into a single struct. 
 */
FANN_EXTERNAL struct fann_train_data *FANN_API fann_merge_train_data(struct fann_train_data *data1,
																	 struct fann_train_data *data2)
{
	unsigned int i;
	fann_type *data_input, *data_output;
	struct fann_train_data *dest =
		(struct fann_train_data *) malloc(sizeof(struct fann_train_data));

	if(dest == NULL)
	{
		fann_error((struct fann_error*)data1, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	if((data1->num_input != data2->num_input) || (data1->num_output != data2->num_output))
	{
		fann_error((struct fann_error*)data1, FANN_E_TRAIN_DATA_MISMATCH);
		return NULL;
	}

	fann_init_error_data((struct fann_error *) dest);
	dest->error_log = data1->error_log;

	dest->num_data = data1->num_data+data2->num_data;
	dest->num_input = data1->num_input;
	dest->num_output = data1->num_output;
	dest->input = (fann_type **) calloc(dest->num_data, sizeof(fann_type *));
	if(dest->input == NULL)
	{
		fann_error((struct fann_error*)data1, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	dest->output = (fann_type **) calloc(dest->num_data, sizeof(fann_type *));
	if(dest->output == NULL)
	{
		fann_error((struct fann_error*)data1, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	data_input = (fann_type *) calloc(dest->num_input * dest->num_data, sizeof(fann_type));
	if(data_input == NULL)
	{
		fann_error((struct fann_error*)data1, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}
	memcpy(data_input, data1->input[0], dest->num_input * data1->num_data * sizeof(fann_type));
	memcpy(data_input + (dest->num_input*data1->num_data), 
		data2->input[0], dest->num_input * data2->num_data * sizeof(fann_type));

	data_output = (fann_type *) calloc(dest->num_output * dest->num_data, sizeof(fann_type));
	if(data_output == NULL)
	{
		fann_error((struct fann_error*)data1, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}
	memcpy(data_output, data1->output[0], dest->num_output * data1->num_data * sizeof(fann_type));
	memcpy(data_output + (dest->num_output*data1->num_data), 
		data2->output[0], dest->num_output * data2->num_data * sizeof(fann_type));

	for(i = 0; i != dest->num_data; i++)
	{
		dest->input[i] = data_input;
		data_input += dest->num_input;
		dest->output[i] = data_output;
		data_output += dest->num_output;
	}
	return dest;
}

/*
 * return a copy of a fann_train_data struct 
 */
FANN_EXTERNAL struct fann_train_data *FANN_API fann_duplicate_train_data(struct fann_train_data
																		 *data)
{
	unsigned int i;
	fann_type *data_input, *data_output;
	struct fann_train_data *dest =
		(struct fann_train_data *) malloc(sizeof(struct fann_train_data));

	if(dest == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	fann_init_error_data((struct fann_error *) dest);
	dest->error_log = data->error_log;

	dest->num_data = data->num_data;
	dest->num_input = data->num_input;
	dest->num_output = data->num_output;
	dest->input = (fann_type **) calloc(dest->num_data, sizeof(fann_type *));
	if(dest->input == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	dest->output = (fann_type **) calloc(dest->num_data, sizeof(fann_type *));
	if(dest->output == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	data_input = (fann_type *) calloc(dest->num_input * dest->num_data, sizeof(fann_type));
	if(data_input == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}
	memcpy(data_input, data->input[0], dest->num_input * dest->num_data * sizeof(fann_type));

	data_output = (fann_type *) calloc(dest->num_output * dest->num_data, sizeof(fann_type));
	if(data_output == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}
	memcpy(data_output, data->output[0], dest->num_output * dest->num_data * sizeof(fann_type));

	for(i = 0; i != dest->num_data; i++)
	{
		dest->input[i] = data_input;
		data_input += dest->num_input;
		dest->output[i] = data_output;
		data_output += dest->num_output;
	}
	return dest;
}

FANN_EXTERNAL struct fann_train_data *FANN_API fann_subset_train_data(struct fann_train_data
																		 *data, unsigned int pos,
																		 unsigned int length)
{
	unsigned int i;
	fann_type *data_input, *data_output;
	struct fann_train_data *dest =
		(struct fann_train_data *) malloc(sizeof(struct fann_train_data));

	if(dest == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	if(pos > data->num_data || pos+length > data->num_data)
	{
		fann_error((struct fann_error*)data, FANN_E_TRAIN_DATA_SUBSET, pos, length, data->num_data);
		return NULL;
	}

	fann_init_error_data((struct fann_error *) dest);
	dest->error_log = data->error_log;

	dest->num_data = length;
	dest->num_input = data->num_input;
	dest->num_output = data->num_output;
	dest->input = (fann_type **) calloc(dest->num_data, sizeof(fann_type *));
	if(dest->input == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	dest->output = (fann_type **) calloc(dest->num_data, sizeof(fann_type *));
	if(dest->output == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	data_input = (fann_type *) calloc(dest->num_input * dest->num_data, sizeof(fann_type));
	if(data_input == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}
	memcpy(data_input, data->input[pos], dest->num_input * dest->num_data * sizeof(fann_type));

	data_output = (fann_type *) calloc(dest->num_output * dest->num_data, sizeof(fann_type));
	if(data_output == NULL)
	{
		fann_error((struct fann_error*)data, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}
	memcpy(data_output, data->output[pos], dest->num_output * dest->num_data * sizeof(fann_type));

	for(i = 0; i != dest->num_data; i++)
	{
		dest->input[i] = data_input;
		data_input += dest->num_input;
		dest->output[i] = data_output;
		data_output += dest->num_output;
	}
	return dest;
}

FANN_EXTERNAL unsigned int FANN_API fann_length_train_data(struct fann_train_data *data)
{
	return data->num_data;
}

FANN_EXTERNAL unsigned int FANN_API fann_num_input_train_data(struct fann_train_data *data)
{
	return data->num_input;
}

FANN_EXTERNAL unsigned int FANN_API fann_num_output_train_data(struct fann_train_data *data)
{
	return data->num_output;
}

/* INTERNAL FUNCTION
   Save the train data structure.
 */
int fann_save_train_internal(struct fann_train_data *data, const char *filename,
							  unsigned int save_as_fixed, unsigned int decimal_point)
{
	int retval = 0;
	FILE *file = fopen(filename, "w");

	if(!file)
	{
		fann_error((struct fann_error *) data, FANN_E_CANT_OPEN_TD_W, filename);
		return -1;
	}
	retval = fann_save_train_internal_fd(data, file, filename, save_as_fixed, decimal_point);
	fclose(file);
	
	return retval;
}

/* INTERNAL FUNCTION
   Save the train data structure.
 */
int fann_save_train_internal_fd(struct fann_train_data *data, FILE * file, const char *filename,
								 unsigned int save_as_fixed, unsigned int decimal_point)
{
	unsigned int num_data = data->num_data;
	unsigned int num_input = data->num_input;
	unsigned int num_output = data->num_output;
	unsigned int i, j;
	int retval = 0;

#ifndef FIXEDFANN
	unsigned int multiplier = 1 << decimal_point;
#endif

	fprintf(file, "%u %u %u\n", data->num_data, data->num_input, data->num_output);

	for(i = 0; i < num_data; i++)
	{
		for(j = 0; j < num_input; j++)
		{
#ifndef FIXEDFANN
			if(save_as_fixed)
			{
				fprintf(file, "%d ", (int) (data->input[i][j] * multiplier));
			}
			else
			{
				if(((int) floor(data->input[i][j] + 0.5) * 1000000) ==
				   ((int) floor(data->input[i][j] * 1000000.0 + 0.5)))
				{
					fprintf(file, "%d ", (int) data->input[i][j]);
				}
				else
				{
					fprintf(file, "%f ", data->input[i][j]);
				}
			}
#else
			fprintf(file, FANNPRINTF " ", data->input[i][j]);
#endif
		}
		fprintf(file, "\n");

		for(j = 0; j < num_output; j++)
		{
#ifndef FIXEDFANN
			if(save_as_fixed)
			{
				fprintf(file, "%d ", (int) (data->output[i][j] * multiplier));
			}
			else
			{
				if(((int) floor(data->output[i][j] + 0.5) * 1000000) ==
				   ((int) floor(data->output[i][j] * 1000000.0 + 0.5)))
				{
					fprintf(file, "%d ", (int) data->output[i][j]);
				}
				else
				{
					fprintf(file, "%f ", data->output[i][j]);
				}
			}
#else
			fprintf(file, FANNPRINTF " ", data->output[i][j]);
#endif
		}
		fprintf(file, "\n");
	}
	
	return retval;
}

/*
 * Creates an empty and initialized training data structure
 * Must fill in with fann_set_train_data() before using
 */
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train(unsigned int num_data, unsigned int num_input, unsigned int num_output)
{
	fann_type *data_input, *data_output;
	unsigned int i;
	struct fann_train_data *data =
		(struct fann_train_data *) malloc(sizeof(struct fann_train_data));

	if(data == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	fann_init_error_data((struct fann_error *) data);

	data->num_data = num_data;
	data->num_input = num_input;
	data->num_output = num_output;
	data->input = (fann_type **) calloc(num_data, sizeof(fann_type *));
	if(data->input == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data->output = (fann_type **) calloc(num_data, sizeof(fann_type *));
	if(data->output == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data_input = (fann_type *) calloc(num_input * num_data, sizeof(fann_type));
	if(data_input == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	data_output = (fann_type *) calloc(num_output * num_data, sizeof(fann_type));
	if(data_output == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}

	for(i = 0; i != num_data; i++)
	{
		data->input[i] = data_input;
		data_input += num_input;
		data->output[i] = data_output;
		data_output += num_output;
	}
	return data;
}

/*
 * Creates training data from a callback function.
 */
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train_from_callback(unsigned int num_data,
                                          unsigned int num_input,
                                          unsigned int num_output,
                                          void (FANN_API *user_function)( unsigned int,
                                                                 unsigned int,
                                                                 unsigned int,
                                                                 fann_type * ,
                                                                 fann_type * ))
{
    unsigned int i;
	struct fann_train_data *data = fann_create_train(num_data, num_input, num_output);
	if(data == NULL)
	{
		return NULL;
	}

    for( i = 0; i != num_data; i++)
    {
        (*user_function)(i, num_input, num_output, data->input[i], data->output[i]);
    }

    return data;
} 


/*
 * Sets the input and desired output values into the specified position in the training data structure
 */
FANN_EXTERNAL int FANN_API fann_set_train(struct fann_train_data* data, unsigned int position, fann_type* input, fann_type* output)
{
    unsigned int i;

    if(position>=data->num_data)
    {
        fann_error(NULL, FANN_E_INDEX_OUT_OF_BOUND);
        return -1;
    }

    for(i=0;i<data->num_input;++i) data->input[position][i] = input[i];
    for(i=0;i<data->num_output;++i) data->output[position][i] = output[i];

    return 0;
}


/*
 * Gets the input and desired output values from the specified position in the training data structure
 */
FANN_EXTERNAL int FANN_API fann_get_train(struct fann_train_data* data, unsigned int position, fann_type* input, fann_type* output)
{
    unsigned int i;

    if(position>=data->num_data)
    {
        fann_error(NULL, FANN_E_INDEX_OUT_OF_BOUND);
        return -1;
    }

    for(i=0;i<data->num_input;++i) input[i] = data->input[position][i];
    for(i=0;i<data->num_output;++i) output[i] = data->output[position][i];

    return 0;
}


/*
 * INTERNAL FUNCTION Reads training data from a file descriptor. 
 */
struct fann_train_data *fann_read_train_from_fd(FILE * file, const char *filename)
{
	unsigned int num_input, num_output, num_data, i, j;
	unsigned int line = 1;
	struct fann_train_data *data;

	if(fscanf(file, "%u %u %u\n", &num_data, &num_input, &num_output) != 3)
	{
		fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
		return NULL;
	}
	line++;

	data = fann_create_train(num_data, num_input, num_output);
	if(data == NULL)
	{
		return NULL;
	}

	for(i = 0; i != num_data; i++)
	{
		for(j = 0; j != num_input; j++)
		{
			if(fscanf(file, FANNSCANF " ", &data->input[i][j]) != 1)
			{
				fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
				fann_destroy_train(data);
				return NULL;
			}
		}
		line++;

		for(j = 0; j != num_output; j++)
		{
			if(fscanf(file, FANNSCANF " ", &data->output[i][j]) != 1)
			{
				fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
				fann_destroy_train(data);
				return NULL;
			}
		}
		line++;
	}
	return data;
}

/*
 * INTERNAL FUNCTION returns 0 if the desired error is reached and -1 if it is not reached
 */
int fann_desired_error_reached(struct fann *ann, float desired_error)
{
	switch (ann->training_params->train_stop_function)
	{
	case FANN_STOPFUNC_MSE:
		if(fann_get_MSE(ann) <= desired_error)
			return 0;
		break;
	case FANN_STOPFUNC_BIT:
		if(ann->training_params->num_bit_fail <= (unsigned int)desired_error)
			return 0;
		break;
	}
	return -1;
}

#ifndef FIXEDFANN
/*
 * Scale data in input vector before feed it to ann based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_scale_input( struct fann *ann, fann_type *input_vector )
{
	unsigned cur_neuron;
	if(ann->scale_params->scale_mean_in == NULL)
	{
		fann_error( (struct fann_error *) ann, FANN_E_SCALE_NOT_PRESENT );
		return;
	}
	
	for( cur_neuron = 0; cur_neuron < ann->num_input; cur_neuron++ )
		input_vector[ cur_neuron ] =
			(
				( input_vector[ cur_neuron ] - ann->scale_params->scale_mean_in[ cur_neuron ] )
				/ ann->scale_params->scale_deviation_in[ cur_neuron ]
				- ( -1.0 ) /* This is old_min */
			)
			* ann->scale_params->scale_factor_in[ cur_neuron ]
			+ ann->scale_params->scale_new_min_in[ cur_neuron ];
}

/*
 * Scale data in output vector before feed it to ann based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_scale_output( struct fann *ann, fann_type *output_vector )
{
	unsigned cur_neuron;
	if(ann->scale_params->scale_mean_in == NULL)
	{
		fann_error( (struct fann_error *) ann, FANN_E_SCALE_NOT_PRESENT );
		return;
	}

	for( cur_neuron = 0; cur_neuron < ann->num_output; cur_neuron++ )
		output_vector[ cur_neuron ] =
			(
				( output_vector[ cur_neuron ] - ann->scale_params->scale_mean_out[ cur_neuron ] )
				/ ann->scale_params->scale_deviation_out[ cur_neuron ]
				- ( -1.0 ) /* This is old_min */
			)
			* ann->scale_params->scale_factor_out[ cur_neuron ]
			+ ann->scale_params->scale_new_min_out[ cur_neuron ];
}

/*
 * Descale data in input vector after based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_descale_input( struct fann *ann, fann_type *input_vector )
{
	unsigned cur_neuron;
	if(ann->scale_params->scale_mean_in == NULL)
	{
		fann_error( (struct fann_error *) ann, FANN_E_SCALE_NOT_PRESENT );
		return;
	}

	for( cur_neuron = 0; cur_neuron < ann->num_input; cur_neuron++ )
		input_vector[ cur_neuron ] =
			(
				(
					input_vector[ cur_neuron ]
					- ann->scale_params->scale_new_min_in[ cur_neuron ]
				)
				/ ann->scale_params->scale_factor_in[ cur_neuron ]
				+ ( -1.0 ) /* This is old_min */
			)
			* ann->scale_params->scale_deviation_in[ cur_neuron ]
			+ ann->scale_params->scale_mean_in[ cur_neuron ];
}

/*
 * Descale data in output vector after get it from ann based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_descale_output( struct fann *ann, fann_type *output_vector )
{
	unsigned cur_neuron;
	if(ann->scale_params->scale_mean_in == NULL)
	{
		fann_error( (struct fann_error *) ann, FANN_E_SCALE_NOT_PRESENT );
		return;
	}

	for( cur_neuron = 0; cur_neuron < ann->num_output; cur_neuron++ )
		output_vector[ cur_neuron ] =
			(
				(
					output_vector[ cur_neuron ]
					- ann->scale_params->scale_new_min_out[ cur_neuron ]
				)
				/ ann->scale_params->scale_factor_out[ cur_neuron ]
				+ ( -1.0 ) /* This is old_min */
			)
			* ann->scale_params->scale_deviation_out[ cur_neuron ]
			+ ann->scale_params->scale_mean_out[ cur_neuron ];
}

/*
 * Scale input and output data based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_scale_train( struct fann *ann, struct fann_train_data *data )
{
	unsigned cur_sample;
	if(ann->scale_params->scale_mean_in == NULL)
	{
		fann_error( (struct fann_error *) ann, FANN_E_SCALE_NOT_PRESENT );
		return;
	}
	/* Check that we have good training data. */
	if(fann_check_input_output_sizes(ann, data) == -1)
		return;

	for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )
	{
		fann_scale_input( ann, data->input[ cur_sample ] );
		fann_scale_output( ann, data->output[ cur_sample ] );
	}
}

/*
 * Scale input and output data based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_descale_train( struct fann *ann, struct fann_train_data *data )
{
	unsigned cur_sample;
	if(ann->scale_params->scale_mean_in == NULL)
	{
		fann_error( (struct fann_error *) ann, FANN_E_SCALE_NOT_PRESENT );
		return;
	}
	/* Check that we have good training data. */
	if(fann_check_input_output_sizes(ann, data) == -1)
		return;

	for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )
	{
		fann_descale_input( ann, data->input[ cur_sample ] );
		fann_descale_output( ann, data->output[ cur_sample ] );
	}
}

#define SCALE_RESET( what, where, default_value )							\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )	\
		ann->scale_params->what##_##where[ cur_neuron ] = ( default_value );

#define SCALE_SET_PARAM( where )																		\
	/* Calculate mean: sum(x)/length */																	\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		ann->scale_params->scale_mean_##where[ cur_neuron ] = 0.0;										\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )								\
			ann->scale_params->scale_mean_##where[ cur_neuron ] += data->where##put[ cur_sample ][ cur_neuron ];		\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		ann->scale_params->scale_mean_##where[ cur_neuron ] /= (float)data->num_data;					\
	/* Calculate deviation: sqrt(sum((x-mean)^2)/length) */												\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		ann->scale_params->scale_deviation_##where[ cur_neuron ] = 0.0; 								\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )								\
			ann->scale_params->scale_deviation_##where[ cur_neuron ] += 								\
				/* Another local variable in macro? Oh no! */											\
				( 																						\
					data->where##put[ cur_sample ][ cur_neuron ] 										\
					- ann->scale_params->scale_mean_##where[ cur_neuron ] 								\
				) 																						\
				*																						\
				( 																						\
					data->where##put[ cur_sample ][ cur_neuron ] 										\
					- ann->scale_params->scale_mean_##where[ cur_neuron ] 								\
				); 																						\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		ann->scale_params->scale_deviation_##where[ cur_neuron ] =										\
			sqrt( ann->scale_params->scale_deviation_##where[ cur_neuron ] / (float)data->num_data ); 	\
	/* Calculate factor: (new_max-new_min)/(old_max(1)-old_min(-1)) */									\
	/* Looks like we dont need whole array of factors? */												\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		ann->scale_params->scale_factor_##where[ cur_neuron ] =											\
			( new_##where##put_max - new_##where##put_min )												\
			/																							\
			( 1.0 - ( -1.0 ) );																			\
	/* Copy new minimum. */																				\
	/* Looks like we dont need whole array of new minimums? */											\
	for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )								\
		ann->scale_params->scale_new_min_##where[ cur_neuron ] = new_##where##put_min;

FANN_EXTERNAL int FANN_API fann_set_input_scaling_params(
	struct fann *ann,
	const struct fann_train_data *data,
	float new_input_min,
	float new_input_max)
{
	unsigned cur_neuron, cur_sample;

	/* Check that we have good training data. */
	/* No need for if( !params || !ann ) */
	if(data->num_input != ann->num_input
	   || data->num_output != ann->num_output)
	{
		fann_error( (struct fann_error *) ann, FANN_E_TRAIN_DATA_MISMATCH );
		return -1;
	}

	if(ann->scale_params->scale_mean_in == NULL)
		fann_allocate_scale(ann);
	
	if(ann->scale_params->scale_mean_in == NULL)
		return -1;
		
	if( !data->num_data )
	{
		SCALE_RESET( scale_mean,		in,	0.0 )
		SCALE_RESET( scale_deviation,	in,	1.0 )
		SCALE_RESET( scale_new_min,		in,	-1.0 )
		SCALE_RESET( scale_factor,		in,	1.0 )
	}
	else
	{
		SCALE_SET_PARAM( in );
	}

	return 0;
}

FANN_EXTERNAL int FANN_API fann_set_output_scaling_params(
	struct fann *ann,
	const struct fann_train_data *data,
	float new_output_min,
	float new_output_max)
{
	unsigned cur_neuron, cur_sample;

	/* Check that we have good training data. */
	/* No need for if( !params || !ann ) */
	if(data->num_input != ann->num_input
	   || data->num_output != ann->num_output)
	{
		fann_error( (struct fann_error *) ann, FANN_E_TRAIN_DATA_MISMATCH );
		return -1;
	}

	if(ann->scale_params->scale_mean_out == NULL)
		fann_allocate_scale(ann);
	
	if(ann->scale_params->scale_mean_out == NULL)
		return -1;
		
	if( !data->num_data )
	{
		SCALE_RESET( scale_mean,		out,	0.0 )
		SCALE_RESET( scale_deviation,	out,	1.0 )
		SCALE_RESET( scale_new_min,		out,	-1.0 )
		SCALE_RESET( scale_factor,		out,	1.0 )
	}
	else
	{
		SCALE_SET_PARAM( out );
	}

	return 0;
}

/*
 * Calculate scaling parameters for future use based on training data.
 */
FANN_EXTERNAL int FANN_API fann_set_scaling_params(
	struct fann *ann,
	const struct fann_train_data *data,
	float new_input_min,
	float new_input_max,
	float new_output_min,
	float new_output_max)
{
	if(fann_set_input_scaling_params(ann, data, new_input_min, new_input_max) == 0)
		return fann_set_output_scaling_params(ann, data, new_output_min, new_output_max);
	else
		return -1;
}

/*
 * Clears scaling parameters.
 */
FANN_EXTERNAL int FANN_API fann_clear_scaling_params(struct fann *ann)
{
	unsigned cur_neuron;

	if(ann->scale_params->scale_mean_out == NULL)
		fann_allocate_scale(ann);
	
	if(ann->scale_params->scale_mean_out == NULL)
		return -1;
	
	SCALE_RESET( scale_mean,		in,	0.0 )
	SCALE_RESET( scale_deviation,	in,	1.0 )
	SCALE_RESET( scale_new_min,		in,	-1.0 )
	SCALE_RESET( scale_factor,		in,	1.0 )

	SCALE_RESET( scale_mean,		out,	0.0 )
	SCALE_RESET( scale_deviation,	out,	1.0 )
	SCALE_RESET( scale_new_min,		out,	-1.0 )
	SCALE_RESET( scale_factor,		out,	1.0 )
	
	return 0;
}

#endif

int fann_check_input_output_sizes(struct fann *ann, struct fann_train_data *data)
{
	if(ann->num_input != data->num_input)
    {
    	fann_error((struct fann_error *) ann, FANN_E_INPUT_NO_MATCH,
        	ann->num_input, data->num_input);
        return -1;
    }
        
	if(ann->num_output != data->num_output)
	{
		fann_error((struct fann_error *) ann, FANN_E_OUTPUT_NO_MATCH,
					ann->num_output, data->num_output);
		return -1;
	}
	
	return 0;
}



/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap
 */

