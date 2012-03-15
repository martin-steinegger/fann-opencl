#include "fann.h"
#include "fann_sparse.h"
#include <stdlib.h>

struct dice 
{
	int idx;
	float value;
};

int dice_sorter(const void * d1, const void * d2)
{
	return ((const struct dice*)d2)->value < ((const struct dice*)d1)->value;
}

FANN_EXTERNAL void FANN_API fann_sparse_neuron_destructor(struct fann_neuron* neuron)
{
	struct fann_neuron_private_data_connected_any_any *priv = (struct fann_neuron_private_data_connected_any_any *) neuron->private_data;
	fann_safe_free(neuron->weights);
	fann_safe_free(neuron->weights_deltas);
	fann_safe_free(neuron->sums);
	fann_safe_free(neuron->weights_deltas);
	fann_safe_free(priv->prev_weights_deltas);
	fann_safe_free(priv->prev_steps);
	fann_safe_free( ((struct fann_sparse_neuron_private_data*) neuron->private_data)->mask );
	fann_safe_free( ((struct fann_sparse_neuron_private_data*) neuron->private_data)->generic );
	fann_safe_free(neuron->private_data);

}

/* Allocates room inside the neuron for the connections.
 * Creates a fully connected neuron
 */
FANN_EXTERNAL int FANN_API fann_sparse_neuron_constructor(struct fann *ann, struct fann_layer *layer, 
		struct fann_neuron *neuron, struct fann_neuron_descr * descr)
{
	unsigned int i, j;
	unsigned int min_connections, max_connections, num_connections;
	unsigned int connections_per_output;
	float connection_rate = * ((float* )descr->private_data);
	struct fann_sparse_neuron_private_data* private_data;
	struct fann_neuron_private_data_connected_any_any* generic_private_data;
	fann_type *mask, *weights;
	struct dice *dices;

#ifdef FIXEDFANN
	fann_type multiplier = ann->fixed_params->multiplier;
	neuron->activation_steepness = ann->fixed_params->multiplier / 2;
#else
	neuron->activation_steepness = 0.5;
#endif


	connection_rate = connection_rate > 1.0f ? 1.0f : connection_rate;
	
	neuron->activation_function = FANN_SIGMOID_STEPWISE;


	neuron->num_outputs=descr->num_outputs;
	neuron->inputs=layer->inputs;
	neuron->num_inputs=layer->num_inputs;

	/* set the error array to null (lazy allocation) */
	neuron->train_errors=NULL;
	
	/* this is the number of actually allocated weights (some are unused) */
	neuron->num_weights=neuron->num_outputs*neuron->num_inputs;
	
	/* allocate the weights even for unused connections */
	if ( (weights = neuron->weights = (fann_type*) calloc(neuron->num_weights, sizeof(fann_type))) == NULL)
		return 1;
	
	/* allocate space for the dot products results */
	if ( (neuron->sums = (fann_type*) malloc(neuron->num_outputs*sizeof(fann_type))) == NULL)
		return 1;

	/* allocate private data */
	if ( (private_data = neuron->private_data = (struct fann_sparse_neuron_private_data*) malloc(sizeof(struct fann_sparse_neuron_private_data))) == NULL)
		return 1;
	/* private data stores the connection mask, allocate it */
	if ( (mask = private_data->mask = (fann_type*) calloc(neuron->num_weights, sizeof(fann_type))) == NULL)
		return 1;
	if ( (generic_private_data = private_data->generic = (struct fann_neuron_private_data_connected_any_any*) malloc (sizeof(struct fann_neuron_private_data_connected_any_any))) == NULL)
		return 1;
	generic_private_data->prev_steps=NULL;
	generic_private_data->prev_weights_deltas=NULL;

	/* alocate a set of dices to select rows */
	if ( (dices = (struct dice*) malloc(neuron->num_inputs*sizeof(struct dice))) == NULL)
		return 1;
	
	for (i=0; i<neuron->num_inputs; i++)
	{
		dices[i].idx=i;
		dices[i].value=0;
	}
	
	min_connections = fann_max(neuron->num_inputs, neuron->num_outputs);
	max_connections = neuron->num_inputs * neuron->num_outputs;
	num_connections = fann_max(min_connections,
			(unsigned int) (0.5 + (connection_rate * max_connections)));

	connections_per_output = num_connections / neuron->num_outputs;

	/* Dice throw simulation: a float value is assigned to each input.  
	 * The value decimal component is chosen randomly between 0 and 0.4 ("dice throw").
	 * The integer components is equal to the number of output neurons already
	 * connected to this input.
	 * For each output neuron ecah input gets a new "dice throw". Then the inputs and are 
	 * sorted in ascending order according to the value.
	 * The first ones in the array had less output neurons attached to the 
	 * and better luck in "dice thow". This ones are selected and theyr value is incremented.
	 */
	for (i=0; i<neuron->num_outputs; i++)
	{
		/* throw one dice per input */
		for (j=0; j<neuron->num_inputs; j++)
			dices[j].value= ((int)dices[j].value) + fann_rand(0, 0.4);

		/* sort: smaller (dice value + num_connections) wins) */
		qsort((void*) dices, neuron->num_inputs, sizeof(struct dice), dice_sorter);

		/* assign connections to the output to the winner inputs */
		for (j=0; j<connections_per_output; j++)
		{
			dices[j].value+=1;
			mask[dices[j].idx] = (fann_type) 1.0f;
			weights[dices[j].idx] = (fann_type) fann_random_weight();
		}
		weights += neuron->num_inputs;
	}
	free(dices);

	/* set the function pointers */
	neuron->destructor = fann_sparse_neuron_destructor;
	neuron->run = fann_sparse_neuron_run;
	neuron->backpropagate = fann_sparse_neuron_backprop;
	neuron->update_weights = fann_sparse_neuron_update;
	neuron->compute_error = fann_sparse_neuron_compute_MSE;
	
	return 0;
}

void fann_sparse_neuron_standard_backprop_update(struct fann *ann, struct fann_neuron *neuron)
{
	unsigned int o, j;
	fann_type *weights, *deltas, *mask;
	const unsigned int num_outputs = neuron->num_outputs;
	const unsigned int num_inputs = neuron->num_inputs;
	float learning_rate = ann->backprop_params->learning_rate;

	if (neuron->num_backprop_done==0)
	{
		fann_error(NULL, FANN_E_CANT_USE_TRAIN_ALG);
		return;
	}

	learning_rate=learning_rate/neuron->num_backprop_done;

	/* some assignments to speed up things */
	weights = neuron->weights;
	deltas = neuron->weights_deltas;
	mask = ((struct fann_sparse_neuron_private_data*)neuron->private_data)->mask;

	for (o = 0; o < num_outputs; o++)
	{
		for (j = 0; j < num_inputs; j++)
		{
			/* adjust the weight */
			weights[j] += deltas[j] * mask[j] * learning_rate; /* FIXME add the learning momentum here */
			deltas[j]=0;
		}
		weights += num_inputs;
		deltas += num_inputs;
		mask += num_inputs;
	}
	neuron->num_backprop_done=0;
}

/* INTERNAL FUNCTION
	 The iRprop- algorithm
	 */
void fann_sparse_neuron_irpropm_update(struct fann *ann, struct fann_neuron *neuron)
{
	struct fann_neuron_private_data_connected_any_any *priv = (struct fann_neuron_private_data_connected_any_any *) neuron->private_data;

	fann_type *weights = neuron->weights;
	fann_type *weights_deltas = neuron->weights_deltas;
	fann_type *prev_weights_deltas = priv->prev_weights_deltas;
	fann_type *prev_steps = priv->prev_steps;
	fann_type *mask = ((struct fann_sparse_neuron_private_data*) neuron->private_data)->mask;

	const unsigned int num_outputs = neuron->num_outputs;
	const unsigned int num_inputs = neuron->num_inputs;
	float increase_factor = ann->rprop_params->rprop_increase_factor;	/*1.2; */
	float decrease_factor = ann->rprop_params->rprop_decrease_factor;	/*0.5; */
	float delta_min = ann->rprop_params->rprop_delta_min;	/*0.0; */
	float delta_max = ann->rprop_params->rprop_delta_max;	/*50.0; */

	unsigned int o, i;
	fann_type prev_step, delta, prev_delta, next_step, same_sign;
	
	if (neuron->num_backprop_done==0)
	{
		fann_error(NULL, FANN_E_CANT_USE_TRAIN_ALG);
		return;
	}

	for (o = 0; o < num_outputs; o++)
	{
		for (i = 0; i < num_inputs; i++)
		{
			/*don't update masked connections*/
			if (!mask[i])
				continue;
			prev_step = fann_max(prev_steps[i], (fann_type) 0.0001);	/* prev_step may not be zero because then the training will stop */
			/* does 0.0001 make sense????*/
			delta = weights_deltas[i];
			prev_delta = prev_weights_deltas[i];

			same_sign = prev_delta * delta;

			if(same_sign >= 0.0)
				next_step = fann_min(prev_step * increase_factor, delta_max);
			else
			{
				next_step = fann_max(prev_step * decrease_factor, delta_min);
				delta = 0;
			}

			if(delta < 0)
			{
				weights[i] -= next_step;
				if(weights[i] < -1500)
					weights[i] = -1500;
			}
			else
			{
				weights[i] += next_step;
				if(weights[i] > 1500)
					weights[i] = 1500;
			}

			/* update data arrays */
			prev_steps[i] = next_step;
			prev_weights_deltas[i] = delta;
			weights_deltas[i] = 0.0;
		}
		weights += num_inputs;
		weights_deltas += num_inputs;
		prev_weights_deltas += num_inputs;
		prev_steps += num_inputs;
		mask +=num_inputs;
	}
	neuron->num_backprop_done=0;
}

FANN_EXTERNAL void FANN_API fann_sparse_neuron_update(struct fann *ann, struct fann_neuron *neuron)
{
	switch (ann->training_params->training_algorithm)
	{
		case FANN_TRAIN_INCREMENTAL:
		case FANN_TRAIN_SARPROP:
		case FANN_TRAIN_QUICKPROP:
		case FANN_TRAIN_BATCH:
			fann_neuron_standard_backprop_update_connected_any_any(ann, neuron);
			return;
		case FANN_TRAIN_RPROP:
			fann_neuron_irpropm_update_connected_any_any(ann, neuron);
			return;
		default:
			return;
	}
}

/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap noet
 */
