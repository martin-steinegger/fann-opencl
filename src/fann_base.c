#ifndef __fann_base_c
#define __fann_base_c

#include <string.h>
#include "fann.h"

static __inline void fann_base_inline_neuron_destructor(struct fann_neuron* neuron)
{
	fann_safe_free(neuron->weights);
	fann_safe_free(neuron->weights_deltas);
	fann_safe_free(neuron->sums);
	fann_safe_free(neuron->type);
}

static __inline int fann_base_inline_neuron_constructor(struct fann *ann, struct fann_layer *layer, 
		struct fann_neuron *neuron, struct fann_neuron_descr * descr)
{
	unsigned int i;

#ifdef FIXEDFANN
	fann_type multiplier = ann->fixed_params->multiplier;
	neuron->activation_steepness = ann->fixed_params->multiplier / 2;
#else
	neuron->activation_steepness = 0.5;
#endif

	neuron->activation_function = FANN_SIGMOID_STEPWISE;
	neuron->num_outputs=descr->num_outputs;
	neuron->inputs=layer->inputs;
	neuron->num_inputs=layer->num_inputs;

	/* set the error array to null (lazy allocation) */
	neuron->train_errors=NULL;

	/* allocate the weights */
	neuron->num_weights=neuron->num_outputs*neuron->num_inputs;

	if ( (neuron->weights = (fann_type*) malloc(neuron->num_weights*sizeof(fann_type))) == NULL)
		return 1;

	/* randomly initialize the weights */
	for (i=0; i<neuron->num_weights; i++)
		neuron->weights[i] = (fann_type) fann_random_weight();

	/* allocate space for the dot products results */
	if ( (neuron->sums = (fann_type*) malloc(neuron->num_outputs*sizeof(fann_type))) == NULL)
		return 1;
	return 0;
}

static __inline  void  fann_base_inline_layer_destructor(struct fann_layer* layer)
{
	struct fann_neuron *neuron_it;

	for (neuron_it=layer->first_neuron; neuron_it!=layer->last_neuron; neuron_it++)
		neuron_it->destructor(neuron_it);

	fann_safe_free(layer->first_neuron);
	fann_safe_free(layer->train_errors);
	fann_safe_free(layer->outputs);
	fann_safe_free(layer->type);
}

static __inline  int  fann_base_inline_layer_constructor(struct fann *ann, 
		struct fann_layer *layer, struct fann_layer_descr *descr)
{
	/* sanity checks are done before calling this function */
	struct fann_neuron_descr *neurons_descr = descr->neurons_descr;
	unsigned int i=0;
	fann_type *free_output;
	struct fann_neuron *n;

#ifdef FIXEDFANN
	fann_type multiplier = ann->fixed_params->multiplier;
#endif

	layer->num_neurons = descr->num_neurons;

	/* count the number of outputs for the layer */
	layer->num_outputs = 0;

	for (i=0; i< layer->num_neurons; i++)
	{
		layer->num_outputs += neurons_descr[i].num_outputs;
	}

	layer->num_outputs++;	/* +1 for bias */

	/* set the error array to null (lazy allocation)*/
	layer->train_errors=NULL;

	/* allocate the outputs array */
	free_output = layer->outputs = (fann_type*) calloc(layer->num_outputs,sizeof(fann_type));
	if( layer->outputs == NULL)
	{
		return 1;
	}

	/* set bias output to 1*/
#ifdef FIXEDFANN
	layer->outputs[layer->num_outputs-1]=multiplier;
#else
	layer->outputs[layer->num_outputs-1]=1;
#endif

	/* allocate the neurons array */
	if( (layer->first_neuron = (struct fann_neuron *) calloc(layer->num_neurons, sizeof(struct fann_neuron))) == NULL)
	{
		return 1;
	}

	layer->last_neuron=layer->first_neuron+layer->num_neurons;

	for (i=0; i< layer->num_neurons; i++)
	{
		n=layer->first_neuron+i;

		n->outputs=free_output;
		if( neurons_descr[i].constructor(ann, layer, n, neurons_descr+i) != 0)
		{
			return 1;
		}
		free_output+=n->num_outputs;

	}
	return 0;
}

static __inline  void  fann_base_inline_layer_run(struct fann *ann, struct fann_layer* layer)
{
	struct fann_neuron * last_neuron = layer->last_neuron;
	struct fann_neuron * neuron_it;

	for(neuron_it = layer->first_neuron; neuron_it != last_neuron; neuron_it++)
		neuron_it->run(ann, neuron_it);	
}

static __inline  void  fann_base_inline_neuron_train_initialize(struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron)
{
	neuron->num_backprop_done=0;

	/* allocate the weights_deltas */
	if(neuron->weights_deltas == NULL)
	{
		if ( (neuron->weights_deltas = (fann_type*) calloc(neuron->num_weights, sizeof(fann_type))) == NULL)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	else
	{
		/* clear the error variables */
		memset(neuron->weights_deltas, 0,  neuron->num_weights*sizeof(fann_type));
	}
}

static __inline  void  fann_base_inline_layer_train_initialize(struct fann *ann, struct fann_layer *layer)
{
	fann_type *free_train_errors;
	struct fann_neuron *neuron_it, *last_neuron;

	last_neuron = layer->last_neuron;

	/* reset the layer train errors array */
	if(layer->train_errors == NULL)
	{
		if( (layer->train_errors = (fann_type *) calloc(layer->num_outputs, sizeof(fann_type))) == NULL )
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
		/* assign to MIMO neurons a piece of layer train_errors array */
		free_train_errors = layer->train_errors;
		for (neuron_it = layer->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->train_errors = free_train_errors;
			free_train_errors += neuron_it->num_outputs;
		}	
	}
	else
	{
		/* clear the error variables */
		memset(layer->train_errors, 0, (layer->num_outputs) * sizeof(fann_type));
	}

	for (neuron_it = layer->first_neuron; neuron_it != last_neuron; neuron_it++)
		neuron_it->train_initialize(ann, layer, neuron_it);
}

static __inline void fann_recurrent_neuron_run(struct fann *ann, struct fann_neuron *neuron)
{

}

/*
 * Compute the error at the network output
 * (usually, after forward propagation of a certain input vector, neuron->run).
 * The error is a sum of squares for all the output units
 * also increments a counter because MSE is an average of such errors
 *
 * After this train_errors in the output layer will be set to:
 * (desired_output - neuron_value)
 */
#define fann_base_neuron_compute_MSE(ann, neuron, desired_output)\
/*static __inline  void  fann_base_neuron_compute_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type *desired_output)*/\
{\
	unsigned int o;\
	fann_type neuron_value, neuron_diff;\
	fann_type *error_it, *output_it;\
\
	/* assign each neuron its piece of train_errors array */\
	error_it = neuron->train_errors;\
	output_it = neuron->outputs;\
\
	for (o = 0; o < neuron->num_outputs; o++)\
	{\
		neuron_value = output_it[o];\
		neuron_diff = desired_output[o] - neuron_value;\
\
		update_MSE_macro()(ann, neuron, &neuron_diff);\
\
		if(ann->training_params->train_error_function)\
		{	/* TODO make switch when more functions */\
			if(neuron_diff < -.9999999)\
				neuron_diff = -17.0;\
			else if(neuron_diff > .9999999)\
				neuron_diff = 17.0;\
			else\
				neuron_diff = (fann_type) log((1.0 + neuron_diff) / (1.0 - neuron_diff));\
		}\
\
		error_it[o] = neuron_diff;\
		ann->training_params->num_MSE++;\
	}\
}

/*
 * Compute (neuron_value_derived * train_errors), then
 * Backpropagate the error in the previous layer, then
 * Adjust the weights.
 *
 * This function expect to find the errors array not empty 
 * (especially in the first call).
 */
#ifndef FIXEDFANN

#define  fann_base_neuron_backprop(ann, neuron, prev_layer_errors)\
{\
	unsigned int o, j;\
	fann_type *errors, *inputs, *weights, *deltas;\
	fann_type tmp=0;\
	const unsigned int num_outputs = neuron->num_outputs;\
	const unsigned int num_inputs = neuron->num_inputs;\
	\
	/* some assignments to speed up things */\
	errors = neuron->train_errors;\
	inputs = neuron->inputs;\
	weights = neuron->weights;\
	deltas = neuron->weights_deltas;\
	\
	/* detect if we are on the first layer (if so we get no prev_layer_errors) */\
	if (prev_layer_errors != NULL)\
	{\
		for (o = 0; o < num_outputs; o++)\
		{\
			/* multiply errors with the activation function derivative. the errors array must have been already allocated. */\
			activation_derived_macro()(neuron, o, tmp);\
			errors[o] = errors[o] * tmp;\
			\
			for (j = 0; j < num_inputs; j++)\
			{\
				/* calculate the weight deltas */\
				deltas[j] += errors[o] * inputs[j];\
				\
				/* calculate the error for previous layer */\
				prev_layer_errors[j] += errors[o] * weights[j];\
			}\
			weights += num_inputs;\
			deltas += num_inputs;\
		}\
	}\
	else\
	{ /* If on the first layer, don't backpropagate the errros */\
		for (o = 0; o < num_outputs; o++)\
		{\
			/* multiply errors with the activation function derivative. the errors array must have been already allocated. */\
			activation_derived_macro()(neuron, o, tmp);\
			errors[o] = errors[o] * tmp;\
			\
			for (j = 0; j < num_inputs; j++)\
			{\
				/* calculate the weight deltas */\
				deltas[j] += errors[o] * inputs[j];\
			}\
			weights += num_inputs;\
			deltas += num_inputs;\
		}\
	}\
	\
	neuron->num_backprop_done++;\
}

#else /* FIXEDFANN */

/* BUG: Function body needed! */
#define  fann_base_neuron_backprop(ann, neuron, prev_layer_errors)\
{\
}

#endif

static __inline  void  fann_base_inline_neuron_update(struct fann *ann, struct fann_neuron *neuron)
{
}

#define fann_base_neuron_constructor(ann, layer, neuron, descr) fann_base_inline_neuron_constructor(ann, layer, neuron, descr)
#define fann_base_neuron_destructor(neuron) fann_base_inline_neuron_destructor(neuron)
#define fann_base_neuron_train_initialize(ann, layer, neuron) fann_base_inline_neuron_train_initialize(ann, layer, neuron)
#define fann_base_layer_destructor(layer) fann_base_inline_layer_destructor(layer)
#define fann_base_layer_constructor(ann, layer, descr) fann_base_inline_layer_constructor(ann, layer, descr) 
#define fann_base_layer_run(ann, layer) fann_base_inline_layer_run(ann, layer)
#define fann_base_layer_train_initialize(ann, layer) fann_base_inline_layer_train_initialize(ann, layer)
#define fann_base_neuron_run(ann, neuron) MAKE_NAME(base_neuron_run)(ann, neuron)
#define fann_base_neuron_update(ann, neuron) fann_base_inline_neuron_update(ann, neuron)
#endif /*__fann_base_c*/

/* OUTSIDE THE HEADER GUARD */

#ifndef EXCLUDE_BASE_RUN
#ifdef FIXEDFANN 
static __inline  void  MAKE_NAME(base_neuron_run)(struct fann * ann, struct fann_neuron * neuron)
{
	unsigned int i, o, num_connections, num_inputs, num_outputs;
	fann_type *neuron_sums, *outputs, *inputs, *weights;
	unsigned int activation_function;
	fann_type steepness;

	int multiplier = ann->fixed_params->multiplier;
	unsigned int decimal_point = ann->fixed_params->decimal_point;

	/* values used for the stepwise linear sigmoid function */
	fann_type r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
	fann_type v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;

	fann_type last_steepness = 0;
	unsigned int last_activation_function = 0;

	/* Algorith for fully connected networks */
	activation_function = neuron->activation_function;
	steepness = neuron->activation_steepness;

	num_inputs = neuron->num_inputs;
	inputs = neuron->inputs;
	num_outputs = neuron->num_outputs;
	outputs = neuron->outputs;
	num_connections = neuron->num_inputs;
	weights = neuron->weights;
	neuron_sums=neuron->sums;

	for (o=0; o<num_outputs ; o++)
	{
		neuron_sums[o]=0;
		/* unrolled loop start */
		i = num_connections & 3;	/* same as modulo 4 */
		switch (i)
		{
			case 3:
				neuron_sums[o] += fann_mult(weights[2], inputs[2]);
			case 2:
				neuron_sums[o] += fann_mult(weights[1], inputs[1]);
			case 1:
				neuron_sums[o] += fann_mult(weights[0], inputs[0]);
			case 0:
				break;
		}

		for(; i != num_connections; i += 4)
		{
			neuron_sums[o] +=
				fann_mult(weights[i]    , inputs[i]    ) +
				fann_mult(weights[i + 1], inputs[i + 1]) +
				fann_mult(weights[i + 2], inputs[i + 2]) +
				fann_mult(weights[i + 3], inputs[i + 3]);
		}
		weights += num_connections;
		/* unrolled loop end */

		neuron->sums[o] = fann_mult(steepness, neuron_sums[o]);

		if(activation_function != last_activation_function || steepness != last_steepness)
		{
			switch (activation_function)
			{
				case FANN_SIGMOID:
				case FANN_SIGMOID_STEPWISE:
					r1 = ann->fixed_params->sigmoid_results[0];
					r2 = ann->fixed_params->sigmoid_results[1];
					r3 = ann->fixed_params->sigmoid_results[2];
					r4 = ann->fixed_params->sigmoid_results[3];
					r5 = ann->fixed_params->sigmoid_results[4];
					r6 = ann->fixed_params->sigmoid_results[5];
					v1 = ann->fixed_params->sigmoid_values[0] / steepness;
					v2 = ann->fixed_params->sigmoid_values[1] / steepness;
					v3 = ann->fixed_params->sigmoid_values[2] / steepness;
					v4 = ann->fixed_params->sigmoid_values[3] / steepness;
					v5 = ann->fixed_params->sigmoid_values[4] / steepness;
					v6 = ann->fixed_params->sigmoid_values[5] / steepness;
					break;
				case FANN_SIGMOID_SYMMETRIC:
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:
					r1 = ann->fixed_params->sigmoid_symmetric_results[0];
					r2 = ann->fixed_params->sigmoid_symmetric_results[1];
					r3 = ann->fixed_params->sigmoid_symmetric_results[2];
					r4 = ann->fixed_params->sigmoid_symmetric_results[3];
					r5 = ann->fixed_params->sigmoid_symmetric_results[4];
					r6 = ann->fixed_params->sigmoid_symmetric_results[5];
					v1 = ann->fixed_params->sigmoid_symmetric_values[0] / steepness;
					v2 = ann->fixed_params->sigmoid_symmetric_values[1] / steepness;
					v3 = ann->fixed_params->sigmoid_symmetric_values[2] / steepness;
					v4 = ann->fixed_params->sigmoid_symmetric_values[3] / steepness;
					v5 = ann->fixed_params->sigmoid_symmetric_values[4] / steepness;
					v6 = ann->fixed_params->sigmoid_symmetric_values[5] / steepness;
					break;
				case FANN_THRESHOLD:
					break;
			}
		}

		switch (activation_function)
		{
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				neuron->outputs[o] =
					(fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, 0,
							multiplier, neuron_sums[o]);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				neuron->outputs[o] =
					(fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6,
							-multiplier, multiplier, neuron_sums[o]);
				break;
			case FANN_THRESHOLD:
				neuron->outputs[o] = (fann_type) ((neuron_sums[o] < 0) ? 0 : multiplier);
				break;
			case FANN_THRESHOLD_SYMMETRIC:
				neuron->outputs[o] = (fann_type) ((neuron_sums[o] < 0) ? -multiplier : multiplier);
				break;
			case FANN_LINEAR:
				neuron->outputs[o] = neuron_sums[o];
				break;
			case FANN_LINEAR_PIECE:
				neuron->outputs[o] = (fann_type)((neuron_sums[o] < 0) ? 0 : (neuron_sums[o] > multiplier) ? multiplier : neuron_sums[o]);
				break;
			case FANN_LINEAR_PIECE_SYMMETRIC:
				neuron->outputs[o] = (fann_type)((neuron_sums[o] < -multiplier) ? -multiplier : (neuron_sums[o] > multiplier) ? multiplier : neuron_sums[o]);
				break;
			case FANN_ELLIOT:
			case FANN_ELLIOT_SYMMETRIC:
			case FANN_GAUSSIAN:
			case FANN_GAUSSIAN_SYMMETRIC:
			case FANN_GAUSSIAN_STEPWISE:
			case FANN_SIN_SYMMETRIC:
			case FANN_COS_SYMMETRIC:
				fann_error((struct fann_error *) ann, FANN_E_CANT_USE_ACTIVATION);
				break;
		}
		last_steepness = steepness;
		last_activation_function = activation_function;
	}
}
#else
static __inline  void  MAKE_NAME(base_neuron_run)(struct fann * ann, struct fann_neuron * neuron)
{
	unsigned int i, o, num_connections, num_outputs;
	fann_type *neuron_sums, *inputs, *weights;
	fann_type steepness;

	fann_type max_sum = 0;

	/* Algorithm for fully connected networks */
	steepness = neuron->activation_steepness;

	inputs = neuron->inputs;
	num_outputs = neuron->num_outputs;
	num_connections = neuron->num_inputs;
	weights = neuron->weights;
	neuron_sums=neuron->sums;
	
	for (o=0; o<num_outputs ; ++o)
	{
		fann_type sum = 0.0;
        
		/* unrolled loop start */
		i = num_connections & 3;	/* same as modulo 4 */
		switch (i)
		{
			case 3:
				sum += fann_mult(weights[2], inputs[2]);
//                printf("%5d %5d: %15f %15f\n", o, 2, weights[2], inputs[2]);
			case 2:
				sum += fann_mult(weights[1], inputs[1]);
//                printf("%5d %5d: %15f %15f\n", o, 1, weights[1], inputs[1]);
			case 1:
				sum += fann_mult(weights[0], inputs[0]);
//                printf("%5d %5d: %15f %15f\n", o, 0, weights[0], inputs[0]);
			case 0:
				break;
		}
		
		for(; i != num_connections; i += 4)
		{
            
//            printf("%5d %5d: %15f %15f\n", o, i, weights[i]    , inputs[i]    );
//            printf("%5d %5d: %15f %15f\n", o, i+1, weights[i + 1], inputs[i + 1]);
//            printf("%5d %5d: %15f %15f\n", o, i+2, weights[i + 2], inputs[i + 2]);
//            printf("%5d %5d: %15f %15f\n", o, i+3, weights[i + 3], inputs[i + 3]);
            
            sum +=
				fann_mult(weights[i]    , inputs[i]    ) +
				fann_mult(weights[i + 1], inputs[i + 1]) +
				fann_mult(weights[i + 2], inputs[i + 2]) +
				fann_mult(weights[i + 3], inputs[i + 3]);
		}
		weights += num_connections;
		/* unrolled loop end */
		
		sum = fann_mult(steepness, sum);
		
		max_sum = 150/steepness;
		if(sum > max_sum)
			sum = max_sum;
		else if(sum < -max_sum)
			sum = -max_sum;
		
        neuron_sums[o] = sum;
		activation_macro()(neuron, o);
        
//        printf("%5d %5d: %15f\n", o, i, neuron->outputs[o]);
	}
    
//    double wsum = 0.0;
//    for(o=0; o<num_outputs; ++o)
//        wsum += neuron->weights[num_connections+o];
    
//    printf("%20.14f (%20.14f) ", wsum, neuron_sums[1]);
}
#endif /*FIXEDFANN*/
#endif /* EXCLUDE_BASE_RUN */
/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap noet
 */
