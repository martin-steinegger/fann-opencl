/*
  Fast Artificial Neural Network Library (fann)
  Copyright (C) 2003 Steffen Nissen (lukesky@diku.dk)
  
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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "fann.h"

/*#define DEBUGTRAIN*/

#ifndef FIXEDFANN
/* INTERNAL FUNCTION
  Calculates the activation of a value, given an activation function
   and a steepness
*/
fann_type fann_activation(struct fann * ann, unsigned int activation_function, fann_type steepness,
						  fann_type value)
{
	value = fann_mult(steepness, value);
	fann_activation_switch(activation_function, value, value);
	return value;
}

/* 
 * Trains the network with the backpropagation algorithm.
 */
FANN_EXTERNAL void FANN_API fann_train(struct fann *ann, fann_type * input, fann_type * desired_output)
{
	fann_run(ann, input);
	fann_compute_MSE(ann, desired_output);
	fann_backpropagate_MSE(ann);
	fann_update_weights(ann);
}
#endif


/* Tests the network.
 */
FANN_EXTERNAL fann_type *FANN_API fann_test(struct fann *ann, fann_type * input,
											fann_type * desired_output)
{
	fann_type *output_begin = fann_run(ann, input);
	struct fann_layer *last_layer;
	struct fann_neuron *neuron_it, *last_neuron;

	last_layer = ann->last_layer - 1;
	if (last_layer->train_errors==NULL)
		last_layer->initialize_train_errors(ann, last_layer);
	last_neuron = last_layer->last_neuron;

	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
		neuron_it->compute_error(ann, neuron_it, desired_output);
	
	return output_begin;
}

/* get the mean square error.
 */
FANN_EXTERNAL float FANN_API fann_get_MSE(struct fann *ann)
{
	if(ann->training_params->num_MSE)
		return ann->training_params->MSE_value / (float) ann->training_params->num_MSE;
	else
		return 0;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_bit_fail(struct fann *ann)
{
	return ann->training_params->num_bit_fail;	
}

/* reset the mean square error.
 */
FANN_EXTERNAL void FANN_API fann_reset_MSE(struct fann *ann)
{
/*printf("resetMSE %d %f\n", ann->num_MSE, ann->MSE_value);*/
	ann->training_params->num_MSE = 0;
	ann->training_params->MSE_value = 0;
	ann->training_params->num_bit_fail = 0;
}

#ifndef FIXEDFANN

/* INTERNAL FUNCTION
	compute the error at the network output
	(usually, after forward propagation of a certain input vector, fann_run)
	the error is a sum of squares for all the output units
	also increments a counter because MSE is an average of such errors

	After this train_errors in the output layer will be set to:
	neuron_value_derived * (desired_output - neuron_value)
 */
void fann_compute_MSE(struct fann *ann, fann_type *desired_output)
{
	struct fann_layer *last_layer;
	struct fann_neuron *neuron_it, *last_neuron;

	/* if no room allocated for the error variables, allocate it now (lazy allocation) */
	last_layer = ann->last_layer - 1;
	if (last_layer->train_errors==NULL)
		last_layer->initialize_train_errors(ann, last_layer);

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
		neuron_it->compute_error(ann, neuron_it, desired_output);
}

/* INTERNAL FUNCTION
   Propagate the error backwards from the output layer.

   After this the train_errors in the hidden layers will be:
   neuron_value_derived * sum(outgoing_weights * connected_neuron)
*/
void fann_backpropagate_MSE(struct fann *ann)
{
	struct fann_layer *layer_it, *first_layer, *last_layer;
	struct fann_neuron *neuron_it, *last_neuron;
	fann_type *prev_layer_errors;

	first_layer = ann->first_layer;
	last_layer = ann->last_layer;
	
	/* Note: the last layer has already been initialized using
   * Skip the first layer. It can't backpropagate, just needs to update the deltas. */
	for (layer_it = last_layer-1; layer_it != first_layer; layer_it--)
	{
    /* initailize the prevoius layer */
    if ((layer_it-1)->train_errors==NULL)
      (layer_it-1)->initialize_train_errors(ann, layer_it-1);
    else
    {
      /* clear just the error variables */
      memset((layer_it-1)->train_errors, 0, ((layer_it-1)->num_outputs) * sizeof(fann_type));
    }
		
    last_neuron = layer_it->last_neuron;
		prev_layer_errors = (layer_it - 1)->train_errors;
		for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			neuron_it->backpropagate(ann, neuron_it, prev_layer_errors);
	}
	/* Iterate only over the first layer (layer_it == first_layer) */
	last_neuron = first_layer->last_neuron;
	for (neuron_it = first_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
		neuron_it->backpropagate(ann, neuron_it, NULL);
}

/* INTERNAL FUNCTION
   Update weights for incremental training
*/
void fann_update_weights(struct fann *ann)
{
	struct fann_layer *layer_it, *first_layer, *last_layer;
	struct fann_neuron *neuron_it, *last_neuron;

	first_layer = ann->first_layer;
	last_layer = ann->last_layer;
	
	for (layer_it = first_layer; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
            neuron_it->update_weights(ann, neuron_it);
	}
}

#else

/* FIXME: Empty body for fixedfann */
void fann_compute_MSE(struct fann *ann, fann_type *desired_output)
{
}

#endif

FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann,
																enum fann_activationfunc_enum activation_function)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it;
	struct fann_layer *last_layer = ann->last_layer - 1;	/* -1 to not update the output layer */

	for(layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->activation_function = activation_function;
		}
	}
}

FANN_EXTERNAL struct fann_layer* FANN_API fann_get_layer(struct fann *ann, int layer)
{
	if(layer <= 0 || layer >= (ann->last_layer - ann->first_layer))
	{
		fann_error((struct fann_error *) ann, FANN_E_INDEX_OUT_OF_BOUND, layer);
		return NULL;
	}
	
	return ann->first_layer + layer;	
}

FANN_EXTERNAL struct fann_neuron* FANN_API fann_get_neuron_layer(struct fann *ann, struct fann_layer* layer, int neuron)
{
	if(neuron >= (layer->last_neuron - layer->first_neuron))
	{
		fann_error((struct fann_error *) ann, FANN_E_INDEX_OUT_OF_BOUND, neuron);
		return NULL;	
	}
	
	return layer->first_neuron + neuron;
}

FANN_EXTERNAL struct fann_neuron* FANN_API fann_get_neuron(struct fann *ann, unsigned int layer, int neuron)
{
	struct fann_layer *layer_it = fann_get_layer(ann, layer);
	if(layer_it == NULL)
		return NULL;
	return fann_get_neuron_layer(ann, layer_it, neuron);
}

FANN_EXTERNAL enum fann_activationfunc_enum FANN_API
    fann_get_activation_function(struct fann *ann, int layer, int neuron)
{
	struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
	if (neuron_it == NULL)
    {
		return (enum fann_activationfunc_enum)-1; /* layer or neuron out of bounds */
    }
    else
    {
	    return neuron_it->activation_function;
    }
}

FANN_EXTERNAL void FANN_API fann_set_activation_function(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function,
																int layer,
																int neuron)
{
	struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
	if(neuron_it == NULL)
		return;

	neuron_it->activation_function = activation_function;
}

FANN_EXTERNAL void FANN_API fann_set_activation_function_layer(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function,
																int layer)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it = fann_get_layer(ann, layer);
	
	if(layer_it == NULL)
		return;

	last_neuron = layer_it->last_neuron;
	for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_function = activation_function;
	}
}


FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann,
																enum fann_activationfunc_enum activation_function)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *last_layer = ann->last_layer - 1;

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_function = activation_function;
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
																 fann_type steepness)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it;
	struct fann_layer *last_layer = ann->last_layer - 1;	/* -1 to not update the output layer */

	for(layer_it = ann->first_layer ; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->activation_steepness = steepness;
		}
	}
}

FANN_EXTERNAL fann_type FANN_API
    fann_get_activation_steepness(struct fann *ann, int layer, int neuron)
{
	struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
	if(neuron_it == NULL)
    {
		return -1; /* layer or neuron out of bounds */
    }
    else
    {
        return neuron_it->activation_steepness;
    }
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness(struct fann *ann,
																fann_type steepness,
																int layer,
																int neuron)
{
	struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
	if(neuron_it == NULL)
		return;

	neuron_it->activation_steepness = steepness;
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_layer(struct fann *ann,
																fann_type steepness,
																int layer)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it = fann_get_layer(ann, layer);
	
	if(layer_it == NULL)
		return;

	last_neuron = layer_it->last_neuron;
	for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_steepness = steepness;
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann,
																 fann_type steepness)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *last_layer = ann->last_layer - 1;

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_steepness = steepness;
	}
}


FANN_GET_SETP(enum fann_errorfunc_enum, training_params, train_error_function)
FANN_GET_SETP(float, rprop_params, quickprop_decay)
FANN_GET_SETP(float, rprop_params, quickprop_mu)
FANN_GET_SETP(float, rprop_params, rprop_increase_factor)
FANN_GET_SETP(float, rprop_params, rprop_decrease_factor)
FANN_GET_SETP(float, rprop_params, rprop_delta_min)
FANN_GET_SETP(float, rprop_params, rprop_delta_max)
FANN_GET_SETP(float, rprop_params, rprop_delta_zero)
FANN_GET_SETP(float, rprop_params, sarprop_weight_decay_shift)
FANN_GET_SETP(float, rprop_params, sarprop_step_error_threshold_factor)
FANN_GET_SETP(float, rprop_params, sarprop_step_error_shift)
FANN_GET_SETP(float, rprop_params, sarprop_temperature)
FANN_GET_SETP(enum fann_stopfunc_enum, training_params, train_stop_function)
FANN_GET_SETP(fann_type, training_params, bit_fail_limit)
FANN_GET_SETP(float, backprop_params, learning_momentum)
FANN_GET_SETP(float, backprop_params, learning_rate)
/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap
 */

