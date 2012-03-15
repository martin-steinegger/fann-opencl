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

#include <assert.h>
#include <string.h>
#include "fann.h"
#include "fann_recurrent.h"

/**************************************************
 HOPFIELD NETWORK

 Fully recurrent N neuron network which is
 used for content-addressable memories. Uses
 one-shot learning for input patterns of length
 N. Then, given an input vector of length N,
 it will recall one of the initial patterns
 (or other fixed points such as their inverses).
 *************************************************/

FANN_EXTERNAL fann_type *FANN_API fann_run_hopfield(struct fann *ann, fann_type *input)
{
	struct fann_neuron *neuron = NULL;
	unsigned int num_neurons = 0;
	unsigned int rand_neuron = 0;
	unsigned int i = 0;
	unsigned int neuron_array_size = 0;
	unsigned int iters = 0;
	int statediff = 0;
	
	fann_type sum = 0;
	fann_type *old_output = NULL;
	fann_type *weights = NULL;

	assert(ann != NULL);
	assert(input != NULL);

	neuron = ann->first_layer->first_neuron;
	num_neurons = ann->first_layer->num_outputs;

	/* Initialization*/
	for (i=0; i<num_neurons; i++)
	{
		ann->output[i] = input[i];
	}

	neuron_array_size = num_neurons * sizeof(fann_type);
	old_output = (fann_type *)malloc(neuron_array_size);

	/* Iterate until states unchanged*/
	/* FIXME: The number of iterations is currently */
	/*   somewhat arbitrary (10*num_neurons once appears*/
	/*   stable). Having a better measure of whether*/
	/*   the network is stable would be nice.*/
	do
	{

		/* Asynchronously update the neurons*/
		rand_neuron = floor(rand()%num_neurons);
		/*printf("Iters: %d (rand = %d)\n", iters, rand_neuron);*/
		weights = neuron[rand_neuron].weights;
		memcpy(old_output, ann->output, neuron_array_size);

		/* Compute the new output vector*/
		sum = 0;
		for (i=0; i<num_neurons; i++)
		{
			sum += weights[i] * ann->output[i];	
		}

		ann->output[rand_neuron] = (sum >= 0) ? 1 : -1;


		/* Compare the old output vector to the new output vector*/
		if ((statediff = memcmp(old_output, ann->output, neuron_array_size)) != 0)
		{
			iters = 0;
		}

		iters += (statediff == 0) ? 1 : 0;

	} while (iters < 10*num_neurons);

	fann_safe_free(old_output);

	/* FIXME: Is it OK to return an internal fp? */
	return ann->output;
}


FANN_EXTERNAL void FANN_API fann_train_hopfield(struct fann *ann, struct fann_train_data *pattern)
{
	unsigned int i = 0, j = 0;
	struct fann_neuron *neuron = NULL;
	unsigned int num_neurons   = 0;
	unsigned int curr_pattern  = 0;
	unsigned int num_patterns  = 0;

	fann_type *curr_weights    = NULL;

	assert(ann     != NULL);
	assert(pattern != NULL);

	/* Only one layer exists*/
	neuron = ann->first_layer->first_neuron;
	num_patterns = pattern->num_data;

	num_neurons = ann->first_layer->num_outputs;
	for (i=0; i<num_neurons; i++)
	{
		curr_weights = neuron[i].weights;

		for (j=0; j<num_neurons; j++)
		{
			*curr_weights = 0;

			/* One shot learning (weighted avg of patterns)*/
			if (i != j)
			{
				for (curr_pattern=0; curr_pattern<num_patterns; curr_pattern++)
				{
					*curr_weights += pattern->input[curr_pattern][i] * pattern->input[curr_pattern][j];
				}

				*curr_weights /= (fann_type)num_patterns;
			}

			curr_weights++;
		}
	}
}



/* A Hopfield network will be implemented as
   a fully recurrent network with 0 inputs and 'num_neurons'
   outputs! */
FANN_EXTERNAL struct fann *FANN_API fann_create_hopfield(
	unsigned int num_neurons)
{
	struct fann *ann = NULL;

	assert(num_neurons > 0);

	ann = (struct fann *)fann_create_fully_recurrent(
		num_neurons, 0, num_neurons);

	return ann;
}


/* Creates a fully recurrent network, where a
     single layer contains all neurons. The
     input layer maps directly to the first
     'num_inputs' of the neurons, and the 
     output layer is the last 'num_outputs' 
     neurons.
   THE WEIGHTS MATRIX:
     + rank <- current neuron
     + file <- neuron which current connects TO
     + The first 'num_inputs' are inputs
     + The last 'num_outputs' are outputs */
FANN_EXTERNAL struct fann *FANN_API fann_create_fully_recurrent(
	unsigned int num_neurons, 
	unsigned int num_inputs, 
	unsigned int num_outputs)
{
    struct fann *ann = NULL;
    
    struct fann_descr descr;
    unsigned int j = 0;
    int exit_error=0;

	assert(num_outputs <= num_neurons);

	/* Create and setup the layers the n-1 hidden layer descriptors*/
	if(fann_setup_descr(&descr, 1, num_inputs) != 0)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

    /* Create single layer with 'num_neurons' MIMO neurons*/
	exit_error = fann_setup_layer_descr(
				descr.layers_descr,
				"fully_recurrent",
				num_neurons,
				NULL
				);

	/* Number of outputs from output layer are the number*/
	/*   of MIMO neurons in it, each MIMO neuron having*/
	/*   a single output*/
	for (j=0; j<descr.layers_descr->num_neurons && ! exit_error; j++)
	{
		exit_error = fann_setup_neuron_descr(
				descr.layers_descr->neurons_descr + j,
				1,
				"fully_recurrent",
				NULL);

		if (exit_error)
		{
			/*FIXME: cleanup neurons*/
			break;
		}
	}

	if (exit_error)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		/*FIXME: cleanup layers*/
		return NULL;
	}

	ann = fann_create_from_descr(&descr);

	ann->network_type = FANN_NETTYPE_FULLY_RECURRENT;
	ann->num_neurons  = num_neurons;
	ann->num_input    = num_inputs;
	ann->num_output   = num_outputs;
	ann->output = ann->first_layer->outputs + ann->first_layer->num_outputs - ann->num_output;
	
    return ann;
}


FANN_EXTERNAL void FANN_API fann_compute_MSE_fully_recurrent(struct fann *ann, fann_type *desired_output)
{
	struct fann_layer *last_layer   = NULL;
	struct fann_neuron *neuron_it   = NULL;
	struct fann_neuron *last_neuron = NULL; 
	
	unsigned int neuron_num         = 0;

	/* if no room allocated for the error variables, allocate it now (lazy allocation) */
	last_layer = ann->last_layer - 1;
	if (last_layer->train_errors == NULL)
	{
		last_layer->initialize_train_errors(ann, last_layer);
	}

	/* Only compute error for last 'ann->num_output' neurons! */
	last_neuron = last_layer->last_neuron;
	neuron_num = 0;
	for(neuron_it = last_layer->last_neuron - ann->num_output; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->train_errors[0] = desired_output[neuron_num] - neuron_it->outputs[0];
		ann->training_params->MSE_value += neuron_it->train_errors[0] * neuron_it->train_errors[0];

		neuron_num++;
		ann->training_params->num_MSE++;
	}
}


/* Creates a feedforward, layered net which is an "unrolled" recurrent network.
 For example, the recurrent net:
  A <-> B <-> C<- (where C autosynapses)
 Becomes (unrolled two time steps):
  A  B  C    input layer
   \/ \/|
  A  B  C    hidden layer I
   \/ \/|
  A  B  C    output layer
*/
FANN_EXTERNAL struct fann *FANN_API fann_create_unrolled_recurrent(
	unsigned int num_neurons, fann_type *weights, unsigned int time_steps)
{
	struct fann *ann        = NULL;
	unsigned int *layers    = NULL;
	unsigned int num_layers = time_steps + 1;
	unsigned int layern     = 0;

	struct fann_layer *curr_layer   = NULL;
	struct fann_neuron *curr_neuron = NULL;
	fann_type *curr_weights         = weights;

	
	/*************************************
	  CREATE THE FEEDFORWARD STRUCTURE 
	 *************************************/

	/* Allocate number of neurons per layer array */
	layers = (unsigned int *)calloc(num_layers, sizeof(unsigned int));
	if (layers == NULL)
	{
		return NULL;
	}

	/* Populate each layer with the number of neurons */
	for (layern=0; layern < num_layers; layern++)
	{
		layers[layern] = num_neurons;
	}

	/* Create the feedforward network */
	ann = fann_create_standard_array(num_layers, layers);
	fann_safe_free(layers);

	/*printf("REQUESTED: LAYERS=%d, NEURONS/LAYER=%d\n", num_layers, num_neurons);
	printf("NUM LAYERS: %d\n", ann->last_layer - ann->first_layer);
	printf("IN: %d, NEURONS: %d, OUTPUT: %d\n",
		ann->num_input, ann->num_neurons, ann->num_output);*/


	/*************************************
	  SET THE FEEDFORWARD WEIGHTS
	 *************************************/

	/* Visit each layer */
    for (curr_layer = ann->first_layer; 
		curr_layer != ann->last_layer; 
		curr_layer++)
	{
		/* The weights are the same for each feedforward layer! */
		curr_weights = weights;

		/* Copy the weight matrix into the neurons, 
		   one row per neuron */
		for (curr_neuron = curr_layer->first_neuron; 
			curr_neuron != curr_layer->last_neuron; 
			curr_neuron++)
		{
            memcpy(curr_neuron->weights, curr_weights, num_neurons * num_neurons * sizeof(fann_type));

			curr_weights += num_neurons;
		}
	}

	return ann;
}


/**************************************************
 REAL-TIME RECURRENT LEARNING

 Williams and Zipser, "A Learning Algorithm for
   Continually Running Fully Recurrent Neural
   Networks," Neural Computation, 1. (1989)

 NOTE: This function is still being debugged.
       MSE does not decrease properly.
 *************************************************/
FANN_EXTERNAL void FANN_API fann_train_rtrl(struct fann *ann, struct fann_train_data *pattern, 
											float max_MSE, unsigned int max_iters, float rate)
{
	struct fann_neuron *neuron = NULL;
	struct fann_layer *layer = NULL;
	fann_type *curr_outputs = NULL;
	fann_type *curr_weight = NULL;

	unsigned int num_neurons = 0;
	unsigned int curr_neuron = 0;
	unsigned int num_iters = 0;
	unsigned int i = 0, j = 0, l = 0;

	float *dodw = NULL;				/* deriv of output wrt weight*/
	float *curr_dodw = NULL;
	float *next_dodw = NULL;		/* dodw for time 'n+1'*/
	float *curr_next_dodw = NULL;
	float *start_dodw = NULL;
	float *temp_swap = NULL;		/* for swapping dodw pointers*/
	float dw = 0.0;					/* change in weight*/

	assert(ann != NULL);
	assert(pattern != NULL);

	/* Only one MIMO neuron and layer in recurrent nets*/
	layer  = ann->first_layer;
	neuron = layer->first_neuron;

	memset(layer->outputs, 0, num_neurons * sizeof(fann_type));

	/* Allocate memory for new outputs*/
	/* TODO: Return an error*/
	num_neurons = layer->num_outputs;
	if ((curr_outputs = calloc(num_neurons, sizeof(fann_type))) == NULL)
	{
		/*fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);*/
		printf("RTRL: Could not allocate 'curr_outputs'\n");
		return;
	}

	/* Allocate memory for derivatives do_k(t)/dw_i,j*/
	/* TODO: Return an error*/
	if ((dodw = calloc(ann->num_output * neuron->num_weights * neuron->num_weights, sizeof(float))) == NULL)
	{
		/*fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);*/
		printf("RTRL: Could not allocate 'dodw'\n");
		return;
	}

	/* Allocate memory for derivatives do_k(t)/dw_i,j*/
	/* TODO: Return an error*/
	if ((next_dodw = calloc(neuron->num_weights * num_neurons, sizeof(float))) == NULL)
	{
		/*fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);*/
		printf("RTRL: Could not allocate 'next_dodw'\n");
		return;
	}

	/* Randomize weights, initialize for training*/
	fann_randomize_weights(ann, -0.5, 0.5);

	if (layer->train_errors==NULL)
	{
		layer->initialize_train_errors(ann, ann->first_layer);
	}

	/* RTRL: Continue learning until MSE low enough or reach*/
	/*       max iterations*/
	num_iters = 0;
	ann->training_params->MSE_value = 100;
	while (ann->training_params->MSE_value > max_MSE && num_iters <= max_iters)
	{
		/* Set the input lines for this time step*/
		/*printf("%d inputs: ", ann->num_input);*/
		for (i=0; i<ann->num_input; i++)
		{
			ann->inputs[i] = pattern->input[num_iters][i];
			printf("%f ", (double) ann->inputs[i]);
		}
		/*printf("(output: %f) (bias: %f) \n", pattern->output[num_iters][0], ann->inputs[ann->num_input]);*/

		/* Copy the outputs of each neuron before they're updated*/
		memcpy(curr_outputs, layer->outputs, num_neurons * sizeof(fann_type));


		/* Update the output of all nodes*/
		layer->run(ann, layer);
		/*printf("NEW OUTPUTS: %f %f %f\n", layer->outputs[0], layer->outputs[1], layer->outputs[2]);*/
		/*printf("ANN OUTPUTS: %f\n", ann->output[0]);*/

		/*curr_weight = neuron->weights;
		for (i=0; i<num_neurons; i++)
		{
			for (j=0; j<layer->num_inputs + num_neurons; j++)
			{
				printf("weight_prev (%d,%d): %f ", i, j, *curr_weight);
				curr_weight++;
			}
		}
		printf("\n");*/

		/* Compute new MSE*/
		fann_reset_MSE(ann);
		fann_compute_MSE(ann, pattern->output[num_iters]);
		printf("%d MSE: %f\n", num_iters, fann_get_MSE(ann));

		/* Modify the weights*/
		start_dodw  = dodw + (num_neurons - ann->num_output) * neuron->num_weights;
		for (i=0; i<num_neurons; i++)
		{
			curr_weight = neuron[i].weights;
			for (j=0; j<layer->num_inputs + num_neurons; j++)
			{
				dw = 0.0;
				curr_dodw = start_dodw;
				/* For each neuron in which is not an input node*/
				for (curr_neuron=num_neurons - ann->num_output; curr_neuron<num_neurons; curr_neuron++)
				{
					dw += (pattern->output[num_iters][curr_neuron - (num_neurons - ann->num_output)] -
						curr_outputs[curr_neuron]) * *curr_dodw;

					curr_dodw += neuron->num_weights;
				}

				*curr_weight += dw * rate;
				/*printf("weight (%d,%d): %f\n", i, j, *curr_weight);*/

				curr_weight++;
				start_dodw++;
			}
		}

		/* Compute next dodw derivatives*/
		curr_next_dodw = next_dodw;
		for (curr_neuron=0; curr_neuron<num_neurons; curr_neuron++)
		{
			start_dodw = dodw;
			curr_weight = neuron->weights;
			for (i=0; i<num_neurons; i++)
			{
				for (j=0; j<layer->num_inputs + num_neurons; j++)
				{
					curr_dodw = start_dodw;

					*curr_next_dodw = 0.0;
					for (l=0; l<num_neurons; l++)
					{
						*curr_next_dodw += *curr_dodw *
							neuron->weights[curr_neuron * (layer->num_inputs + num_neurons) + l + layer->num_inputs];
						curr_dodw += neuron->num_weights;
					}

					/* kronecker_{i,k} * z_j(t)*/
					*curr_next_dodw += (i != curr_neuron) ? 0 :
						((j < layer->num_inputs) ? ann->inputs[j] : curr_outputs[j - layer->num_inputs]);

					*curr_next_dodw *= layer->outputs[curr_neuron]*(1 - layer->outputs[curr_neuron]);
					/*printf("(%d,%d): %f\n", i, j, *curr_next_dodw);*/

					curr_next_dodw++;
					curr_weight++;
					start_dodw++;
				}
			}
		}

		/* Swap the next and the current dodw*/
		/*  (to avoid a costly memory transfer)*/
		temp_swap = dodw;
		dodw = next_dodw;
		next_dodw = temp_swap;

		num_iters++;
	}

	fann_safe_free(dodw);
	fann_safe_free(curr_outputs);
}


/* Displays the connections between the input layer
	 and the internal neurons of a fully recurrent
	 network. Specifically, it prints the number
	 of connections correctly and separately specifies
	 the inputs and outputs. */
FANN_EXTERNAL void FANN_API fann_print_connections_fully_recurrent(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it;
	unsigned int i;
	int value;
	char *neurons;
	unsigned int num_neurons = 0;

	assert(ann != NULL);

	/* Allocate a connection strength per neuron
	 (One extra for bias, one extra for '\0')*/
	neurons = (char *) malloc(num_neurons + 2);
	if(neurons == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	neurons[num_neurons] = 0;

	printf("Neuron ");

	/* Display inputs*/
	for (i=0; i<ann->num_input; i++)
	{
		printf("I");
	}

	/* Display the bias input neuron*/
	printf("b");

	/* Display internal neurons which are not outputs*/
	for(i=0; i<ann->num_neurons - ann->num_output; i++)
	{
		printf("%d", i % 10);
	}

	/* Display outputs*/
	for(i=0; i<ann->num_output; i++)
	{
		printf("O");
	}

	printf("\n");

	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
		{
			memset(neurons, (int) '.', num_neurons);
			for(i = 0; i < neuron_it->num_weights; i++)
			{
				if(neuron_it->weights[i] < 0)
				{
#ifdef FIXEDFANN
					value = (int) ((neuron_it->weights[i] / (double) ann->fixed_params->multiplier) - 0.5);
#else
					value = (int) ((neuron_it->weights[i]) - 0.5);
#endif
					if(value < -25)
						value = -25;
					neurons[i] = (char)('a' - value);
				}
				else
				{
#ifdef FIXEDFANN
					value = (int) ((neuron_it->weights[i] / (double) ann->fixed_params->multiplier) + 0.5);
#else
					value = (int) ((neuron_it->weights[i]) + 0.5);
#endif
					if(value > 25)
						value = 25;
					neurons[i] = (char)('A' + value);
				}
			}

			neurons[i] = '\0';
			printf("N %4d %s\n", (int) (neuron_it - layer_it->first_neuron), neurons);
		}
	}

	free(neurons);
}
