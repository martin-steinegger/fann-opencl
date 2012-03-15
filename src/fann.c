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
#include <time.h>
#include <math.h>

#include <assert.h>

#include "config.h"
#include "fann.h"
#include "fann_som.h"
#include "fann_gng.h"
#include "fann_generic.h"
#include "fann_sparse.h"

#define FANN_NO_SEED

#if ! defined(_MSC_VER) && ! defined(NO_DLOPEN)
#define HAS_DLOPEN
#include <dlfcn.h>
#else
/*************** WORKAROUND ***************/
#ifndef EXCLUDE_SSE
#include "include/optimized/sse/fann.h"
#endif

#ifndef EXCLUDE_BLAS
#include "include/optimized/blas/fann.h"
#endif

#ifndef EXCLUDE_SCALAR
#include "include/optimized/scalar/fann.h"
#endif

#ifndef EXCLUDE_OPENCL
#include "include/optimized/opencl/fann.h"
#endif
#endif


FANN_EXTERNAL struct fann *FANN_API fann_create_standard(unsigned int num_layers, ...)
{
	struct fann *ann;
	va_list layer_sizes;
	int i;
	unsigned int *layers = (unsigned int *) calloc(num_layers, sizeof(unsigned int));

	if(layers == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	va_start(layer_sizes, num_layers);
	for(i = 0; i < (int) num_layers; i++)
	{
		layers[i] = va_arg(layer_sizes, unsigned int);
	}
	va_end(layer_sizes);

	ann = fann_create_standard_array(num_layers, layers);

	free(layers);

	return ann;
}

#ifdef HAS_DLOPEN
void * find_function(const char *name)
{
	void *handle = dlopen(NULL, RTLD_LAZY);
	void *func;
	char *error;

	if (!handle) {
		printf("find_function (%s): No handle\n", name);
		fprintf (stderr, "%s\n", dlerror());
		return NULL;
	}

	dlerror();    /* Clear any existing error*/
	func = dlsym(handle, name);
	if ((error = dlerror()) != NULL)  {
		printf("find_function (%s): No sym\n", name);
		fprintf (stderr, "%s\n", error);
		return NULL;
	}
#if DEBUG
	printf("%s: loaded succesfully\n", name);
#endif
	dlclose(handle);
	return func;
}
#else
void * find_function(const char *name)
{
	void *func=NULL;
#if DEBUG
	printf("USING dlopen WORAROUND: \n");
#endif

  if      (!strcmp(name, "fann_layer_constructor_connected_any_any" ))
			func= (void*) fann_layer_constructor_connected_any_any;
  else if(!strcmp(name, "fann_neuron_constructor_connected_any_any" ))
		   func= (void*) fann_neuron_constructor_connected_any_any;

#ifndef EXCLUDE_SCALAR
  else if(!strcmp(name,  "fann_layer_constructor_scalar_batch_sigmoid" ))
		   func= (void*)  fann_layer_constructor_scalar_batch_sigmoid;
  else if(!strcmp(name,  "fann_layer_constructor_scalar_batch_sigmoid_symmetric" ))
		   func= (void*)  fann_layer_constructor_scalar_batch_sigmoid_symmetric;
  else if(!strcmp(name, "fann_neuron_constructor_scalar_batch_sigmoid" ))
		   func= (void*) fann_neuron_constructor_scalar_batch_sigmoid;
  else if(!strcmp(name, "fann_neuron_constructor_scalar_batch_sigmoid_symmetric" ))
		   func= (void*) fann_neuron_constructor_scalar_batch_sigmoid_symmetric;
  else if(!strcmp(name, "fann_neuron_constructor_scalar_rprop_sigmoid" ))
		   func= (void*) fann_neuron_constructor_scalar_rprop_sigmoid;
  else if(!strcmp(name, "fann_neuron_constructor_scalar_rprop_sigmoid_symmetric" ))
		   func= (void*) fann_neuron_constructor_scalar_rprop_sigmoid_symmetric;
  else if (!strcmp(name, "fann_layer_constructor_scalar_rprop_sigmoid" ))
			func= (void*) fann_layer_constructor_scalar_rprop_sigmoid;
  else if (!strcmp(name, "fann_layer_constructor_scalar_rprop_sigmoid_symmetric" ))
			func= (void*) fann_layer_constructor_scalar_rprop_sigmoid_symmetric;
#endif

#ifndef EXCLUDE_SSE
  else if(!strcmp(name,  "fann_layer_constructor_sse_batch_sigmoid" ))
		   func= (void*)  fann_layer_constructor_sse_batch_sigmoid;
  else if(!strcmp(name,  "fann_layer_constructor_sse_batch_sigmoid_symmetric" ))
		   func= (void*)  fann_layer_constructor_sse_batch_sigmoid_symmetric;
  else if(!strcmp(name, "fann_neuron_constructor_sse_batch_sigmoid" ))
		   func= (void*) fann_neuron_constructor_sse_batch_sigmoid;
  else if(!strcmp(name, "fann_neuron_constructor_sse_batch_sigmoid_symmetric" ))
		   func= (void*) fann_neuron_constructor_sse_batch_sigmoid_symmetric;
  else if(!strcmp(name, "fann_neuron_constructor_sse_rprop_sigmoid" ))
		   func= (void*) fann_neuron_constructor_sse_rprop_sigmoid;
  else if(!strcmp(name, "fann_neuron_constructor_sse_rprop_sigmoid_symmetric" ))
		   func= (void*) fann_neuron_constructor_sse_rprop_sigmoid_symmetric;
  else if (!strcmp(name, "fann_layer_constructor_sse_rprop_sigmoid" ))
			func= (void*) fann_layer_constructor_sse_rprop_sigmoid;
  else if (!strcmp(name, "fann_layer_constructor_sse_rprop_sigmoid_symmetric" ))
			func= (void*) fann_layer_constructor_sse_rprop_sigmoid_symmetric;
#endif

#ifndef EXCLUDE_BLAS
  else if(!strcmp(name,  "fann_layer_constructor_blas_batch_sigmoid" ))
		   func= (void*)  fann_layer_constructor_blas_batch_sigmoid;
  else if(!strcmp(name,  "fann_layer_constructor_blas_batch_sigmoid_symmetric" ))
		   func= (void*)  fann_layer_constructor_blas_batch_sigmoid_symmetric;
  else if(!strcmp(name, "fann_neuron_constructor_blas_batch_sigmoid" ))
		   func= (void*) fann_neuron_constructor_blas_batch_sigmoid;
  else if(!strcmp(name, "fann_neuron_constructor_blas_batch_sigmoid_symmetric" ))
		   func= (void*) fann_neuron_constructor_blas_batch_sigmoid_symmetric;
  else if(!strcmp(name, "fann_neuron_constructor_blas_rprop_sigmoid" ))
		   func= (void*) fann_neuron_constructor_blas_rprop_sigmoid;
  else if(!strcmp(name, "fann_neuron_constructor_blas_rprop_sigmoid_symmetric" ))
		   func= (void*) fann_neuron_constructor_blas_rprop_sigmoid_symmetric;
  else if (!strcmp(name, "fann_layer_constructor_blas_rprop_sigmoid" ))
			func= (void*) fann_layer_constructor_blas_rprop_sigmoid;
  else if (!strcmp(name, "fann_layer_constructor_blas_rprop_sigmoid_symmetric" ))
			func= (void*) fann_layer_constructor_blas_rprop_sigmoid_symmetric;
#endif

#if DEBUG
	if (func)
	printf("%s: loaded succesfully\n", name);
  else
	printf("CRITICAL ERROR:\n\t>>> %s <<<: could not be found\n", name);

#endif
	return func;
}
#endif

FANN_EXTERNAL int FANN_API fann_setup_descr(struct fann_descr* descr, unsigned int num_hidden_layers, unsigned int num_inputs)
{
	descr->num_layers=num_hidden_layers;
	descr->num_inputs=num_inputs;
	descr->layers_descr=(struct fann_layer_descr*) calloc(num_hidden_layers, sizeof(struct fann_layer_descr));
	if(descr->layers_descr == NULL)
		return 1;
	return 0;
}

FANN_EXTERNAL int FANN_API fann_setup_layer_descr(struct fann_layer_descr* layer_descr,
		const char *layer_type,
		unsigned int num_MIMO_neurons,
		void *private_data)
{
	char *layer_constructor;
	unsigned int length = strlen(layer_type)+strlen("fann_layer_constructor_")+1;
	layer_constructor = calloc(length, sizeof(char));

#ifndef _MSC_VER
	snprintf(layer_constructor, length, "fann_layer_constructor_%s", layer_type);
#else
	_snprintf(layer_constructor, length, "fann_layer_constructor_%s", layer_type);
#endif

	layer_descr->constructor = (fann_layer_constructor) find_function(layer_constructor);

	fann_safe_free(layer_constructor);
	if(layer_descr->constructor == NULL )
		return 1;

	layer_descr->num_neurons = num_MIMO_neurons;
	layer_descr->private_data = private_data;

	layer_descr->neurons_descr=(struct fann_neuron_descr*) calloc(layer_descr->num_neurons, sizeof(struct fann_neuron_descr));
	if(layer_descr->neurons_descr == NULL )
		return 1;
	return 0;
}

FANN_EXTERNAL int FANN_API fann_setup_neuron_descr(struct fann_neuron_descr* neuron_descr,
		unsigned int num_outputs,
		const char* neuron_type,
		void *private_data)
{
	char *neuron_constructor;
	unsigned int length = strlen(neuron_type)+strlen("fann_neuron_constructor_")+1;
	neuron_constructor = calloc(length, sizeof(char));

#ifndef _MSC_VER
	snprintf(neuron_constructor, length, "fann_neuron_constructor_%s", neuron_type);
#else
	_snprintf(neuron_constructor, length, "fann_neuron_constructor_%s", neuron_type);
#endif
	neuron_descr->constructor = (fann_neuron_constructor) find_function(neuron_constructor);

	fann_safe_free(neuron_constructor);
	if(neuron_descr->constructor == NULL )
		return 1;
	
	neuron_descr->num_outputs=num_outputs; /*bias not taken into account here*/
	neuron_descr->private_data=private_data;
	return 0;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_standard_array(unsigned int num_layers, 
															   const unsigned int *layers)
{
	return fann_create_standard_array_typed("connected_any_any", "connected_any_any", num_layers, layers);
}

FANN_EXTERNAL struct fann *FANN_API fann_create_standard_array_typed(const char *layer_type, const char *neuron_type, unsigned int num_layers, 
															   const unsigned int *layers)
{
	struct fann *ann;
	unsigned int i,j;
	int exit_error=0;
	struct fann_descr *descr=(struct fann_descr*) calloc(1, sizeof(struct fann_descr));

	if(descr == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	/* Create and setup the layers the n-1 hidden layer descriptors */
	if(fann_setup_descr(descr, num_layers-1, layers[0])!=0)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	for (i=0; i<num_layers-1 && ! exit_error ; i++)
	{
		exit_error = fann_setup_layer_descr(
					descr->layers_descr+i,
					layer_type,
					1,
					NULL
					);

		/* Number of outputs from output layer are the number
		 * of neurons in it
		 */
		for (j=0; j< descr->layers_descr[i].num_neurons && ! exit_error; j++)
		{
			exit_error = fann_setup_neuron_descr(
					descr->layers_descr[i].neurons_descr+j,
					layers[i+1],
					neuron_type,
					NULL);

			if(!exit_error)
			{
				/*FIXME: cleanup neurons*/
			}
		}
		if (exit_error)
		{
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			/*FIXME: cleanup layers*/
			return NULL;
		}
	}

	ann=fann_create_from_descr(descr);

	/* TODO destroy descr	*/

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_sparse(float connection_rate, 
													   unsigned int num_layers, ...)
{
	struct fann *ann;
	va_list layer_sizes;
	int i;
	unsigned int *layers = (unsigned int *) calloc(num_layers, sizeof(unsigned int));

	if(layers == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	va_start(layer_sizes, num_layers);
	for(i = 0; i < (int) num_layers; i++)
	{
		layers[i] = va_arg(layer_sizes, unsigned int);
	}
	va_end(layer_sizes);

	ann = fann_create_sparse_array(connection_rate, num_layers, layers);

	free(layers);

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_sparse_array(float connection_rate, unsigned int num_layers,
															   const unsigned int *layers)
{
	return fann_create_sparse_array_typed("connected_any_any", "connected_any_any", connection_rate, num_layers, layers);
}

FANN_EXTERNAL struct fann *FANN_API fann_create_sparse_array_typed(const char *layer_type, const char *neuron_type, float connection_rate,
															 unsigned int num_layers,
															 const unsigned int *layers)
{
	struct fann *ann;
	unsigned int i,j;
	int exit_error=0;
	struct fann_descr *descr=(struct fann_descr*) calloc(1, sizeof(struct fann_descr));

	if(descr == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	/* Create and setup the layers the n-1 hidden layer descriptors */
	if(fann_setup_descr(descr, num_layers-1, layers[0])!=0)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	for (i=0; i<num_layers-1 && ! exit_error ; i++)
	{
		exit_error = fann_setup_layer_descr(
				descr->layers_descr+i,
				layer_type,
				1,
				NULL
				);

		for (j=0; j< descr->layers_descr[i].num_neurons && ! exit_error; j++)
		{
			exit_error= fann_setup_neuron_descr(
					descr->layers_descr[i].neurons_descr+j,
					layers[i+1],
					neuron_type,
					&(connection_rate));

			if(!exit_error)
			{
				/*FIXME: cleanup neurons*/
			}
		}
		if (exit_error)
		{
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			/*FIXME: cleanup layers*/
			return NULL;
		}
	}

	ann=fann_create_from_descr(descr);
	
	ann->connection_rate=connection_rate;

	/* TODO destroy descr	*/

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_from_descr( struct fann_descr *descr )
{
	struct fann_layer *layer_it;
	struct fann *ann;
	unsigned int i;

	fann_type *parent_outputs;
	unsigned int num_parent_outputs;

#ifdef FIXEDFANN
	unsigned int decimal_point;
	unsigned int multiplier;
#endif
	if( descr->connection_rate > 1 )
	{
		descr->connection_rate = 1;
	}

	/* seed random */
#ifndef FANN_NO_SEED
	fann_seed_rand();
#endif

	/* allocate the general structure */
	ann = fann_allocate_structure(descr->num_layers);
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	/* allocate the input array */
	num_parent_outputs = descr->num_inputs+1;
	ann->num_input = descr->num_inputs;
	parent_outputs = ann->inputs = calloc(num_parent_outputs, sizeof(fann_type));
	if(ann->inputs == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	/* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
	decimal_point = ann->fixed_params->decimal_point;
	multiplier = ann->fixed_params->multiplier;
	fann_update_stepwise(ann);

	ann->inputs[descr->num_inputs] = multiplier;
#else
	ann->inputs[descr->num_inputs] = 1;
#endif
	ann->connection_rate = descr->connection_rate;

	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++, i++)
	{
		layer_it->inputs = parent_outputs;
		layer_it->num_inputs = num_parent_outputs;

		/*FIXME missing error check*/
		descr->layers_descr[i].constructor(ann, layer_it, descr->layers_descr+i );
		
		parent_outputs=layer_it->outputs;
		num_parent_outputs=layer_it->num_outputs;
	}

	ann->output = ann->first_layer[descr->num_layers-1].outputs;
	ann->num_output = ann->first_layer[descr->num_layers-1].num_outputs-1;

	return ann;
}

FANN_EXTERNAL void FANN_API fann_connect_layer(struct fann_layer* layer, unsigned int num_inputs, fann_type * inputs)
{
		layer->inputs = inputs;
		layer->num_inputs = num_inputs;
}

FANN_EXTERNAL void FANN_API fann_connect_layers(struct fann_layer* parent_layer, struct fann_layer* child_layer)
{
		child_layer->inputs = parent_layer->outputs;
		child_layer->num_inputs = parent_layer->num_outputs;
}

#if 0 /* REMOVED */
FANN_EXTERNAL struct fann *FANN_API fann_create_shortcut(unsigned int num_layers, ...)
{
	struct fann *ann;
	int i;
	va_list layer_sizes;
	unsigned int *layers = (unsigned int *) calloc(num_layers, sizeof(unsigned int));

	if(layers == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}


	va_start(layer_sizes, num_layers);
	for(i = 0; i < (int) num_layers; i++)
	{
		layers[i] = va_arg(layer_sizes, unsigned int);
	}
	va_end(layer_sizes);

	ann = fann_create_shortcut_array(num_layers, layers);

	free(layers);

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_shortcut_array(unsigned int num_layers,
															   const unsigned int *layers)
{
	struct fann_layer *layer_it, *layer_it2, *last_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *neuron_it2 = 0;
	unsigned int i;
	unsigned int num_neurons_in, num_neurons_out;

#ifdef FIXEDFANN
	unsigned int decimal_point;
	unsigned int multiplier;
#endif
	/* seed random */
#ifndef FANN_NO_SEED
	fann_seed_rand();
#endif

	/* allocate the general structure */
	ann = fann_allocate_structure(num_layers);
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->connection_rate = 1;
	ann->network_type = FANN_NETTYPE_SHORTCUT;
#ifdef FIXEDFANN
	decimal_point = ann->fixed_params->decimal_point;
	multiplier = ann->fixed_params->multiplier;
	fann_update_stepwise(ann);
#endif

	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		/* we do not allocate room here, but we make sure that
		 * last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layers[i++];
		if(layer_it == ann->first_layer)
		{
			/* there is a bias neuron in the first layer */
			layer_it->last_neuron++;
		}

		ann->total_neurons += layer_it->last_neuron - layer_it->first_neuron;
	}

	ann->num_output = (ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron;
	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1;

	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->error->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

#ifdef DEBUG
	printf("creating fully shortcut connected network.\n");
	printf("input\n");
	printf("  layer       : %d neurons, 1 bias\n",
		   ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);
#endif

	num_neurons_in = ann->num_input;
	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
	{
		num_neurons_out = layer_it->last_neuron - layer_it->first_neuron;

		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++)
		{
			layer_it->first_neuron[i].first_con = ann->total_connections;
			ann->total_connections += num_neurons_in + 1;
			layer_it->first_neuron[i].last_con = ann->total_connections;

			layer_it->first_neuron[i].activation_function = FANN_SIGMOID_STEPWISE;
#ifdef FIXEDFANN
			layer_it->first_neuron[i].activation_steepness = ann->fixed_params->multiplier / 2;
#else
			layer_it->first_neuron[i].activation_steepness = 0.5;
#endif
		}

#ifdef DEBUG
		printf("  layer       : %d neurons, 0 bias\n", num_neurons_out);
#endif
		/* used in the next run of the loop */
		num_neurons_in += num_neurons_out;
	}

	fann_allocate_connections(ann);
	if(ann->error->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	/* Connections are created from all neurons to all neurons in later layers
	 */
	num_neurons_in = ann->num_input + 1;
	for(layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
	{
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
		{

			i = neuron_it->first_con;
			for(layer_it2 = ann->first_layer; layer_it2 != layer_it; layer_it2++)
			{
				for(neuron_it2 = layer_it2->first_neuron; neuron_it2 != layer_it2->last_neuron;
					neuron_it2++)
				{

					ann->weights[i] = (fann_type) fann_random_weight();
					ann->connections[i] = neuron_it2;
					i++;
				}
			}
		}
		num_neurons_in += layer_it->last_neuron - layer_it->first_neuron;
	}

#ifdef DEBUG
	printf("output\n");
#endif

	return ann;
}
#endif /* PENDING */

FANN_EXTERNAL fann_type *FANN_API fann_run(struct fann * ann, fann_type * input)
{
	unsigned int i, num_inputs;
	struct fann_layer *layer_it, *last_layer;
#ifdef FIXEDFANN
	fann_type multiplier = ann->fixed_params->multiplier;
#endif

	/* first set the input */
	num_inputs = ann->num_input;
	for(i = 0; i != num_inputs; i++)
	{
#ifdef FIXEDFANN
		if(fann_abs(input[i]) > multiplier)
		{
			printf
				("Warning input number %d is out of range -%d - %d with value %d, integer overflow may occur.\n",
				 i, multiplier, multiplier, input[i]);
		}
#endif
		ann->inputs[i] = input[i];
	}

	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
		layer_it->run(ann, layer_it);
    
	/* return the output */
	return ann->output;
}

FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann)
{
	struct fann_layer* layer_it;

	assert(ann != NULL);

	if (ann->first_layer != NULL)
	{
		for (layer_it=ann->first_layer; layer_it != ann->last_layer; layer_it++)
		{
			if (layer_it->destructor != NULL)
			{
				layer_it->destructor(layer_it);
			}
		}

		fann_safe_free(ann->first_layer);
	}

	fann_safe_free(ann->inputs);

	if (ann->error != NULL)
	{
		fann_safe_free(ann->error->errstr);
	}

	if (ann->cascade_params != NULL)
	{
		fann_safe_free(ann->cascade_params->cascade_activation_functions);
		fann_safe_free(ann->cascade_params->cascade_activation_steepnesses);
	}
	
#ifdef FIXEDFANN
	fann_safe_free( ann->scale_params->scale_mean_in );
	fann_safe_free( ann->scale_params->scale_deviation_in );
	fann_safe_free( ann->scale_params->scale_new_min_in );
	fann_safe_free( ann->scale_params->scale_factor_in );

	fann_safe_free( ann->scale_params->scale_mean_out );
	fann_safe_free( ann->scale_params->scale_deviation_out );
	fann_safe_free( ann->scale_params->scale_new_min_out );
	fann_safe_free( ann->scale_params->scale_factor_out );
#endif
	
	fann_safe_free(ann->backprop_params);
	fann_safe_free(ann->cascade_params);
#ifdef FIXEDFANN
	fann_safe_free(ann->scale_params);
#endif
	fann_safe_free(ann->rprop_params);
	fann_safe_free(ann->som_params);
	fann_safe_free(ann->gng_params);
	fann_safe_free(ann->training_params);
	fann_safe_free(ann->scale_params);
	fann_safe_free(ann->error);
	fann_safe_free(ann);
}

FANN_EXTERNAL void FANN_API fann_randomize_weights(struct fann *ann, fann_type min_weight,
												   fann_type max_weight)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it;

  if(fann_get_network_type(ann) == FANN_NETTYPE_SOM) {
    fann_randomize_weights_som(ann, min_weight, max_weight);
    return;
  }
  else if(fann_get_network_type(ann) == FANN_NETTYPE_GNG) {
    fann_randomize_weights_gng(ann, min_weight, max_weight);
    return;
  }

	for (layer_it=ann->first_layer; layer_it!= ann->last_layer; layer_it++)
	{
		for (neuron_it=layer_it->first_neuron; neuron_it!=layer_it->last_neuron; neuron_it++)
		{
			fann_type *weights = neuron_it->weights;
			fann_type *last_weight = weights + neuron_it->num_weights;

			for(; weights != last_weight; weights++)
			{
				*weights = (fann_type) (fann_rand(min_weight, max_weight));
			}
		}

	}
#if 0 /* FIXME */
#ifndef FIXEDFANN
	if(ann->rprop_params->prev_train_slopes != NULL)
	{
		fann_clear_train_arrays(ann);
	}
#endif
#endif
}

#if 0
/* deep copy of the fann structure */
FANN_EXTERNAL struct fann* FANN_API fann_copy(const struct fann* orig)
{
    if(orig->network_type == FANN_NETTYPE_SOM)
        return fann_copy_som(orig);
    else if(orig->network_type == FANN_NETTYPE_SOM)
        return fann_copy_gng(orig);

    struct fann* copy;
    unsigned int num_layers = orig->last_layer - orig->first_layer;
    struct fann_layer *orig_layer_it, *copy_layer_it;
    unsigned int layer_size;
    struct fann_neuron *last_neuron,*orig_neuron_it,*copy_neuron_it;
    unsigned int i;
    struct fann_neuron *orig_first_neuron,*copy_first_neuron;
    unsigned int input_neuron;

    copy = fann_allocate_structure(num_layers);
    if (copy==NULL) {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    copy->error->errno_f = orig->error->errno_f;
    if (orig->error->errstr)
    {
        copy->error->errstr = (char *) malloc(FANN_ERRSTR_MAX);
        if (copy->error->errstr == NULL)
        {
            fann_destroy(copy);
            return NULL;
        }
        strcpy(copy->error->errstr,orig->error->errstr);
    }
    copy->error->error_log = orig->error->error_log;

    copy->backprop_params->learning_rate = orig->backprop_params->learning_rate;
    copy->backprop_params->learning_momentum = orig->backprop_params->learning_momentum;
    copy->connection_rate = orig->connection_rate;
    copy->network_type = orig->network_type;
    copy->training_params->num_MSE								= orig->training_params->num_MSE;
    copy->training_params->MSE_value							= orig->training_params->MSE_value;
    copy->training_params->num_bit_fail							= orig->training_params->num_bit_fail;
    copy->training_params->bit_fail_limit						= orig->training_params->bit_fail_limit;
    copy->training_params->train_error_function					= orig->training_params->train_error_function;
    copy->training_params->train_stop_function					= orig->training_params->train_stop_function;
    copy->training_params->callback								= orig->training_params->callback;
    copy->cascade_params->cascade_output_change_fraction		= orig->cascade_params->cascade_output_change_fraction;
    copy->cascade_params->cascade_output_stagnation_epochs		= orig->cascade_params->cascade_output_stagnation_epochs;
    copy->cascade_params->cascade_candidate_change_fraction		= orig->cascade_params->cascade_candidate_change_fraction;
    copy->cascade_params->cascade_candidate_stagnation_epochs	= orig->cascade_params->cascade_candidate_stagnation_epochs;
    copy->cascade_params->cascade_best_candidate		= orig->cascade_params->cascade_best_candidate;
    copy->cascade_params->cascade_candidate_limit		= orig->cascade_params->cascade_candidate_limit;
    copy->cascade_params->cascade_weight_multiplier		= orig->cascade_params->cascade_weight_multiplier;
    copy->cascade_params->cascade_max_out_epochs		= orig->cascade_params->cascade_max_out_epochs;
    copy->cascade_params->cascade_max_cand_epochs		= orig->cascade_params->cascade_max_cand_epochs;
	copy->user_data = orig->user_data;

   /* copy cascade activation functions */
    copy->cascade_params->cascade_activation_functions_count = orig->cascade_params->cascade_activation_functions_count;
    copy->cascade_params->cascade_activation_functions = (enum fann_activationfunc_enum *)
		realloc(copy->cascade_params->cascade_activation_functions,
				copy->cascade_params->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
    if(copy->cascade_params->cascade_activation_functions == NULL)
    {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(copy);
        return NULL;
    }
    memcpy(copy->cascade_params->cascade_activation_functions,orig->cascade_params->cascade_activation_functions,
            copy->cascade_params->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));

    /* copy cascade activation steepnesses */
    copy->cascade_params->cascade_activation_steepnesses_count = orig->cascade_params->cascade_activation_steepnesses_count;
    copy->cascade_params->cascade_activation_steepnesses = (fann_type *)
		realloc(copy->cascade_params->cascade_activation_steepnesses, 
				copy->cascade_params->cascade_activation_steepnesses_count * sizeof(fann_type));
    if(copy->cascade_params->cascade_activation_steepnesses == NULL)
    {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(copy);
        return NULL;
    }
    memcpy(copy->cascade_params->cascade_activation_steepnesses,
		   orig->cascade_params->cascade_activation_steepnesses,
		   copy->cascade_params->cascade_activation_steepnesses_count * sizeof(fann_type));

    copy->cascade_params->cascade_num_candidate_groups = orig->cascade_params->cascade_num_candidate_groups;

    /* copy candidate scores, if used */
    if (orig->cascade_params->cascade_candidate_scores == NULL)
    {
        copy->cascade_params->cascade_candidate_scores = NULL;
    }
    else
    {
        copy->cascade_params->cascade_candidate_scores =
            (fann_type *) malloc(fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
        if(copy->cascade_params->cascade_candidate_scores == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->cascade_params->cascade_candidate_scores,
			orig->cascade_params->cascade_candidate_scores,
			fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
    }

    copy->rprop_params->quickprop_decay			= orig->rprop_params->quickprop_decay;
    copy->rprop_params->quickprop_mu			= orig->rprop_params->quickprop_mu;
    copy->rprop_params->rprop_increase_factor	= orig->rprop_params->rprop_increase_factor;
    copy->rprop_params->rprop_decrease_factor	= orig->rprop_params->rprop_decrease_factor;
    copy->rprop_params->rprop_delta_min			= orig->rprop_params->rprop_delta_min;
    copy->rprop_params->rprop_delta_max			= orig->rprop_params->rprop_delta_max;
    copy->rprop_params->rprop_delta_zero		= orig->rprop_params->rprop_delta_zero;

    /* user_data is not deep copied.  user should use fann_copy_with_user_data() for that */
    copy->user_data = orig->user_data;

#ifdef FIXEDFANN
    copy->scale_params->decimal_point = orig->scale_params->decimal_point;
    copy->scale_params->multiplier = orig->scale_params->multiplier;
    memcpy(copy->scale_params->sigmoid_results,
		orig->scale_params->sigmoid_results,6*sizeof(fann_type));
    memcpy(copy->scale_params->sigmoid_values,
		orig->scale_params->sigmoid_values,6*sizeof(fann_type));
    memcpy(copy->scale_params->sigmoid_symmetric_results,
		orig->scale_params->sigmoid_symmetric_results,6*sizeof(fann_type));
    memcpy(copy->scale_params->sigmoid_symmetric_values,
		orig->scale_params->sigmoid_symmetric_values,6*sizeof(fann_type));
#endif


    /* copy layer sizes, prepare for fann_allocate_neurons */
    for (orig_layer_it = orig->first_layer, copy_layer_it = copy->first_layer;
            orig_layer_it != orig->last_layer; orig_layer_it++, copy_layer_it++)
    {
        layer_size = orig_layer_it->last_neuron - orig_layer_it->first_neuron;
        copy_layer_it->first_neuron = NULL;
        copy_layer_it->last_neuron = copy_layer_it->first_neuron + layer_size;
        copy->total_neurons += layer_size;
    }
    copy->num_input = orig->num_input;
    copy->num_output = orig->num_output;


    /* copy scale parameters, when used */
#ifndef FIXEDFANN
    if (orig->scale_params->scale_mean_in != NULL)
    {
        fann_allocate_scale(copy);
        for (i=0; i < orig->num_input ; i++) {
            copy->scale_params->scale_mean_in[i]		= orig->scale_params->scale_mean_in[i];
            copy->scale_params->scale_deviation_in[i]	= orig->scale_params->scale_deviation_in[i];
            copy->scale_params->scale_new_min_in[i]		= orig->scale_params->scale_new_min_in[i];
            copy->scale_params->scale_factor_in[i]		= orig->scale_params->scale_factor_in[i];
        }
        for (i=0; i < orig->num_output ; i++) {
            copy->scale_params->scale_mean_out[i]		= orig->scale_params->scale_mean_out[i];
            copy->scale_params->scale_deviation_out[i]	= orig->scale_params->scale_deviation_out[i];
            copy->scale_params->scale_new_min_out[i]	= orig->scale_params->scale_new_min_out[i];
            copy->scale_params->scale_factor_out[i]		= orig->scale_params->scale_factor_out[i];
        }
    }
#endif

    /* copy the neurons */
    fann_allocate_neurons(copy);
    if (copy->error->errno_f == FANN_E_CANT_ALLOCATE_MEM)
    {
        fann_destroy(copy);
        return NULL;
    }
    layer_size = (orig->last_layer-1)->last_neuron - (orig->last_layer-1)->first_neuron;
    memcpy(copy->output,orig->output, layer_size * sizeof(fann_type));

    last_neuron = (orig->last_layer - 1)->last_neuron;
    for (orig_neuron_it = orig->first_layer->first_neuron, copy_neuron_it = copy->first_layer->first_neuron;
            orig_neuron_it != last_neuron; orig_neuron_it++, copy_neuron_it++)
    {
        memcpy(copy_neuron_it,orig_neuron_it,sizeof(struct fann_neuron));
    }
 /* copy the connections */
    copy->total_connections = orig->total_connections;
    fann_allocate_connections(copy);
    if (copy->error->errno_f == FANN_E_CANT_ALLOCATE_MEM)
    {
        fann_destroy(copy);
        return NULL;
    }

    orig_first_neuron = orig->first_layer->first_neuron;
    copy_first_neuron = copy->first_layer->first_neuron;
    for (i=0; i < orig->total_connections; i++)
    {
        copy->weights[i] = orig->weights[i];
        input_neuron = orig->connections[i] - orig_first_neuron;
        copy->connections[i] = copy_first_neuron + input_neuron;
    }


    return copy;
}
#endif

FANN_EXTERNAL void FANN_API fann_print_connections(struct fann *ann)
{
	struct fann_layer *layer_it   = NULL;
	struct fann_neuron *neuron_it = NULL;
	unsigned int i = 0;
	int value = 0;
	char *neurons = NULL;
	unsigned int num_neurons = 0;

	assert(ann != NULL);

	if(fann_get_network_type(ann) == FANN_NETTYPE_SOM) {
		fann_print_connections_som(ann);
		return;
	} else if(fann_get_network_type(ann) == FANN_NETTYPE_GNG) {
		fann_print_connections_gng(ann);
		return;
	} else if(fann_get_network_type(ann) == FANN_NETTYPE_FULLY_RECURRENT) {
		fann_print_connections_fully_recurrent(ann);
		return;
	}

	/* Allocate a connection strength per (max) neuron
		(One extra for bias, one extra for '\0')*/
	neurons = (char *) malloc(ann->first_layer->first_neuron->num_weights + 2);
	if(neurons == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	
	neurons[num_neurons] = 0;


	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		printf("\nLayer / Neuron ");
		for(i = 0; i < layer_it->first_neuron->num_weights; i++)
		{
			printf("%d", i % 10);
		}
		printf("\n");

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
			printf("L %3d / N %4d %s\n", (int) (layer_it - ann->first_layer),
					(int) (neuron_it - layer_it->first_neuron), neurons);
		}
	}

	fann_safe_free(neurons);
}

/* Initialize the weights using Widrow + Nguyen's algorithm.
*/
FANN_EXTERNAL void FANN_API fann_init_weights(struct fann *ann, struct fann_train_data *train_data)
{
	fann_type smallest_inp, largest_inp;
	unsigned int dat = 0, elem, num_connect, num_hidden_neurons;
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *last_neuron;

#ifdef FIXEDFANN
	unsigned int multiplier = ann->fixed_params->multiplier;
#endif
	float scale_factor;

  if(fann_get_network_type(ann) == FANN_NETTYPE_SOM)
    fann_init_weights_som(ann, train_data);
  else if(fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    fann_init_weights_gng(ann, train_data);

	for(smallest_inp = largest_inp = train_data->input[0][0]; dat < train_data->num_data; dat++)
	{
		for(elem = 0; elem < train_data->num_input; elem++)
		{
			if(train_data->input[dat][elem] < smallest_inp)
				smallest_inp = train_data->input[dat][elem];
			if(train_data->input[dat][elem] > largest_inp)
				largest_inp = train_data->input[dat][elem];
		}
	}

	num_hidden_neurons = fann_get_total_neurons(ann) - ann->num_output;
	scale_factor =
		(float) (pow
				 ((double) (0.7f * (double) num_hidden_neurons),
				  (double) (1.0f / (double) ann->num_input)) / (double) (largest_inp -
																		 smallest_inp));

#ifdef DEBUG
	printf("Initializing weights with scale factor %f\n", scale_factor);
#endif
	/* TODO handle special case when there is only one bias neuron (cascade)*/
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;

		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			for(num_connect = 0; num_connect < neuron_it->num_weights; num_connect++)
			{
				if( num_connect && ! ((num_connect +1) % neuron_it->num_inputs)  ) /*connection to bias*/
				{
#ifdef FIXEDFANN
					neuron_it->weights[num_connect] =
						(fann_type) fann_rand(-scale_factor, scale_factor * multiplier);
#else
					neuron_it->weights[num_connect] = (fann_type) fann_rand(-scale_factor, scale_factor);
#endif
				}
				else
				{
#ifdef FIXEDFANN
					neuron_it->weights[num_connect] = (fann_type) fann_rand(0, scale_factor * multiplier);
#else
					neuron_it->weights[num_connect] = (fann_type) fann_rand(0, scale_factor);
#endif
				}
			}
		}
	}

#if 0 /* FIXME */
#ifndef FIXEDFANN
	if(ann->rprop_params->prev_train_slopes != NULL)
	{
		fann_clear_train_arrays(ann);
	}
#endif
#endif
}

FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann)
{
	struct fann_layer *layer_it;
	unsigned int i;
	unsigned int b=0;
	
	if(fann_get_network_type(ann) == FANN_NETTYPE_SOM) {
		fann_print_parameters_som(ann);
		return;
	} else if(fann_get_network_type(ann) == FANN_NETTYPE_GNG) {
		fann_print_parameters_gng(ann);
		return;
	}

	printf("Input layer                          :%4d neurons, 1 bias\n", ann->num_input);
	printf("Hidden layers:\n");
	for (i=0, layer_it=ann->first_layer; layer_it!=ann->last_layer-1; i++, layer_it++)
	{
		b=fann_get_total_layer_neurons(layer_it);
  	printf("         [%d]:                        :%4d MIMO neurons, %4d neurons", 
				i, (int) (layer_it->last_neuron-layer_it->first_neuron), b);
		b=layer_it->num_outputs-b;
		if ( !b )
			printf("\n");
		else
			printf(", %4d bias\n",b);
	}
	printf("Output layer                         :%4d neurons\n", ann->num_output);
	printf("Total neurons                        :%4d\n", fann_get_total_neurons(ann));
	/*printf("Total connections                    :%4d\n", ann->total_connections);*/
	printf("Connection rate                      :%8.3f\n", ann->connection_rate);
	printf("Network type                         :   %s\n", FANN_NETTYPE_NAMES[ann->network_type]);
#ifdef FIXEDFANN
	printf("Decimal point                        :%4d\n", ann->fixed_params->decimal_point);
	printf("Multiplier                           :%4d\n", ann->fixed_params->multiplier);
#else
	printf("Training algorithm                   :   %s\n", FANN_TRAIN_NAMES[ann->training_params->training_algorithm]);
	printf("Training error function              :   %s\n", FANN_ERRORFUNC_NAMES[ann->training_params->train_error_function]);
	printf("Training stop function               :   %s\n", FANN_STOPFUNC_NAMES[ann->training_params->train_stop_function]);
#endif
#ifdef FIXEDFANN
	printf("Bit fail limit                       :%4d\n", ann->training_params->bit_fail_limit);
#else
	printf("Bit fail limit                       :%8.3f\n", ann->training_params->bit_fail_limit);
	printf("Learning rate                        :%8.3f\n", ann->backprop_params->learning_rate);
	printf("Learning momentum                    :%8.3f\n", ann->backprop_params->learning_momentum);
	printf("Quickprop decay                      :%11.6f\n", ann->rprop_params->quickprop_decay);
	printf("Quickprop mu                         :%8.3f\n", ann->rprop_params->quickprop_mu);
	printf("RPROP increase factor                :%8.3f\n", ann->rprop_params->rprop_increase_factor);
	printf("RPROP decrease factor                :%8.3f\n", ann->rprop_params->rprop_decrease_factor);
	printf("RPROP delta min                      :%8.3f\n", ann->rprop_params->rprop_delta_min);
	printf("RPROP delta max                      :%8.3f\n", ann->rprop_params->rprop_delta_max);
	printf("RPROP delta zero                     :%8.3f\n", ann->rprop_params->rprop_delta_zero);
#if 0 /*REMOVED*/
	printf("Cascade output change fraction       :%11.6f\n", ann->cascade_params->cascade_output_change_fraction);
	printf("Cascade candidate change fraction    :%11.6f\n", ann->cascade_params->cascade_candidate_change_fraction);
	printf("Cascade output stagnation epochs     :%4d\n", ann->cascade_params->cascade_output_stagnation_epochs);
	printf("Cascade candidate stagnation epochs  :%4d\n", ann->cascade_params->cascade_candidate_stagnation_epochs);
	printf("Cascade max output epochs            :%4d\n", ann->cascade_params->cascade_max_out_epochs);
	printf("Cascade min output epochs            :%4d\n", ann->cascade_params->cascade_min_out_epochs);
	printf("Cascade max candidate epochs         :%4d\n", ann->cascade_params->cascade_max_cand_epochs);
	printf("Cascade min candidate epochs         :%4d\n", ann->cascade_params->cascade_min_cand_epochs);
	printf("Cascade weight multiplier            :%8.3f\n", ann->cascade_params->cascade_weight_multiplier);
	printf("Cascade candidate limit              :%8.3f\n", ann->cascade_params->cascade_candidate_limit);
	for(i = 0; i < ann->cascade_params->cascade_activation_functions_count; i++)
		printf("Cascade activation functions[%d]      :   %s\n", i,
			FANN_ACTIVATIONFUNC_NAMES[ann->cascade_params->cascade_activation_functions[i]]);
	for(i = 0; i < ann->cascade_params->cascade_activation_steepnesses_count; i++)
		printf("Cascade activation steepnesses[%d]    :%8.3f\n", i,
			ann->cascade_params->cascade_activation_steepnesses[i]);
		
	printf("Cascade candidate groups             :%4d\n", ann->cascade_params->cascade_num_candidate_groups);
	printf("Cascade no. of candidates            :%4d\n", fann_get_cascade_num_candidates(ann));
#endif /*REMOVED*/
	
	/* TODO: dump scale parameters */
#endif
}

FANN_GET(unsigned int, num_input)
FANN_GET(unsigned int, num_output)

FANN_EXTERNAL unsigned int FANN_API fann_get_total_layer_neurons(struct fann_layer *layer)
{
	struct fann_neuron *neuron_it;
	unsigned int res=0;
	for (neuron_it=layer->first_neuron; neuron_it!=layer->last_neuron; neuron_it++)
		res+=neuron_it->num_outputs;
	return res;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons(struct fann *ann)
{
	struct fann_layer *layer_it;
	unsigned int res=0;

  if(fann_get_network_type(ann) == FANN_NETTYPE_SOM)
		fann_get_total_neurons_som(ann);
	else if(fann_get_network_type(ann) == FANN_NETTYPE_GNG)
		fann_get_total_neurons_gng(ann);

	for (layer_it=ann->first_layer; layer_it!=ann->last_layer; layer_it++)
		res+=fann_get_total_layer_neurons(layer_it);
	return res;
}

FANN_EXTERNAL enum fann_nettype_enum FANN_API fann_get_network_type(struct fann *ann)
{
    /* Currently two types: LAYER = 0, SHORTCUT = 1 */
    /* Enum network_types must be set to match the return values  */
    return ann->network_type;
}

FANN_EXTERNAL float FANN_API fann_get_connection_rate(struct fann *ann)
{
    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM || 
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return -1;
    }

    return ann->connection_rate;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_layers(struct fann *ann)
{
    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM || 
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return -1;
    }

    return ann->last_layer - ann->first_layer;
}

FANN_EXTERNAL void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers)
{
    struct fann_layer *layer_it;

    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM ||
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return;
    }

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        *layers++ = layer_it->num_outputs;
    }
}

FANN_EXTERNAL void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias)
{
    struct fann_layer *layer_it;

    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM || 
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return;
    }

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; ++layer_it, ++bias) {
        switch (fann_get_network_type(ann)) {
            case FANN_NETTYPE_LAYER: {
                /* Report one bias in each layer except the last */
                if (layer_it != ann->last_layer-1)
                    *bias = 1;
                else
                    *bias = 0;
                break;
            }
            case FANN_NETTYPE_SHORTCUT: {
                /* The bias in the first layer is reused for all layers */
                if (layer_it == ann->first_layer)
                    *bias = 1;
                else
                    *bias = 0;
                break;
            }
            default: {
                /* Unknown network type, assume no bias present  */
                *bias = 0;
                break;
            }
        }
    }
}

#if 0
FANN_EXTERNAL void FANN_API fann_get_connection_array(struct fann *ann, struct fann_connection *connections)
{
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int index;
    unsigned int source_index;
    unsigned int destination_index;

    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM ||
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return;
    }

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;
    
    /* The following assumes that the last unused bias has no connections */

    /* for each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
        /* for each neuron */
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            /* for each connection */
            for (index = neuron_it->first_con; index < neuron_it->last_con; index++){
                /* Assign the source, destination and weight */
                connections->from_neuron = ann->connections[source_index] - first_neuron;
                connections->to_neuron = destination_index;
                connections->weight = ann->weights[source_index];

                connections++;
                source_index++;
            }
            destination_index++;
        }
    }
}

FANN_EXTERNAL void FANN_API fann_set_weight_array(struct fann *ann,
    struct fann_connection *connections, unsigned int num_connections)
{
    unsigned int index;

    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM ||
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return;
    }

    for (index = 0; index < num_connections; index++) {
        fann_set_weight(ann, connections[index].from_neuron,
            connections[index].to_neuron, connections[index].weight);
    }
}

FANN_EXTERNAL void FANN_API fann_set_weight(struct fann *ann,
    unsigned int from_neuron, unsigned int to_neuron, fann_type weight)
{
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int index;
    unsigned int source_index;
    unsigned int destination_index;

    if(fann_get_network_type(ann) == FANN_NETTYPE_SOM ||
			 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
    {
        fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
        return;
    }

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;

    /* Find the connection, simple brute force search through the network
       for one or more connections that match to minimize datastructure dependencies.
       Nothing is done if the connection does not already exist in the network. */

    /* for each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
        /* for each neuron */
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            /* for each connection */
            for (index = neuron_it->first_con; index < neuron_it->last_con; index++){
                /* If the source and destination neurons match, assign the weight */
                if (((int)from_neuron == ann->connections[source_index] - first_neuron) &&
                    (to_neuron == destination_index))
                {
                    ann->weights[source_index] = weight;
                }
                source_index++;
            }
            destination_index++;
        }
    }
}
#endif /*REMOVED*/

FANN_GET_SET(void *, user_data)
FANN_GET_SETP(enum fann_errno_enum, error, errno_f)
FANN_GET_SETP(FILE *, error, error_log)
FANN_GET_SETP(char *, error, errstr)
FANN_GET_SETP(enum fann_train_enum, training_params, training_algorithm)

#ifdef FIXEDFANN

FANN_GETP(unsigned int, fixed_params, decimal_point)
FANN_GETP(unsigned int, fixed_params, multiplier)

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise(struct fann *ann)
{
	unsigned int i = 0;

	if(fann_get_network_type(ann) == FANN_NETTYPE_SOM ||
		 fann_get_network_type(ann) == FANN_NETTYPE_GNG)
	{
		fann_error(NULL, FANN_E_FUNCTION_NA_FOR_SOM);
		return;
	}

	/* Calculate the parameters for the stepwise linear
	 * sigmoid function fixed point.
	 * Using a rewritten sigmoid function.
	 * results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	 */
	ann->fixed_params->sigmoid_results[0] = fann_max((fann_type) (ann->fixed_params->multiplier / 200.0 + 0.5), 1);
	ann->fixed_params->sigmoid_results[1] = fann_max((fann_type) (ann->fixed_params->multiplier / 20.0 + 0.5), 1);
	ann->fixed_params->sigmoid_results[2] = fann_max((fann_type) (ann->fixed_params->multiplier / 4.0 + 0.5), 1);
	ann->fixed_params->sigmoid_results[3] = fann_min(ann->fixed_params->multiplier - (fann_type) (ann->fixed_params->multiplier / 4.0 + 0.5), ann->fixed_params->multiplier - 1);
	ann->fixed_params->sigmoid_results[4] = fann_min(ann->fixed_params->multiplier - (fann_type) (ann->fixed_params->multiplier / 20.0 + 0.5), ann->fixed_params->multiplier - 1);
	ann->fixed_params->sigmoid_results[5] = fann_min(ann->fixed_params->multiplier - (fann_type) (ann->fixed_params->multiplier / 200.0 + 0.5), ann->fixed_params->multiplier - 1);

	ann->fixed_params->sigmoid_symmetric_results[0] = fann_max((fann_type) ((ann->fixed_params->multiplier / 100.0) - ann->fixed_params->multiplier - 0.5),
				                                 (fann_type) (1 - (fann_type) ann->fixed_params->multiplier));
	ann->fixed_params->sigmoid_symmetric_results[1] =	fann_max((fann_type) ((ann->fixed_params->multiplier / 10.0) - ann->fixed_params->multiplier - 0.5),
				                                 (fann_type) (1 - (fann_type) ann->fixed_params->multiplier));
	ann->fixed_params->sigmoid_symmetric_results[2] =	fann_max((fann_type) ((ann->fixed_params->multiplier / 2.0) - ann->fixed_params->multiplier - 0.5),
                                				 (fann_type) (1 - (fann_type) ann->fixed_params->multiplier));
	ann->fixed_params->sigmoid_symmetric_results[3] = fann_min(ann->fixed_params->multiplier - (fann_type) (ann->fixed_params->multiplier / 2.0 + 0.5),
				 							     ann->fixed_params->multiplier - 1);
	ann->fixed_params->sigmoid_symmetric_results[4] = fann_min(ann->fixed_params->multiplier - (fann_type) (ann->fixed_params->multiplier / 10.0 + 0.5),
				 							     ann->fixed_params->multiplier - 1);
	ann->fixed_params->sigmoid_symmetric_results[5] = fann_min(ann->fixed_params->multiplier - (fann_type) (ann->fixed_params->multiplier / 100.0 + 1.0),
				 							     ann->fixed_params->multiplier - 1);

	for(i = 0; i < 6; i++)
	{
		ann->fixed_params->sigmoid_values[i] =
			(fann_type) (((log(ann->fixed_params->multiplier / (float) ann->fixed_params->sigmoid_results[i] - 1) *
						   (float) ann->fixed_params->multiplier) / -2.0) * (float) ann->fixed_params->multiplier);
		ann->fixed_params->sigmoid_symmetric_values[i] =
			(fann_type) (((log
						   ((ann->fixed_params->multiplier -
							 (float) ann->fixed_params->sigmoid_symmetric_results[i]) /
							((float) ann->fixed_params->sigmoid_symmetric_results[i] +
							 ann->fixed_params->multiplier)) * (float) ann->fixed_params->multiplier) / -2.0) *
						 (float) ann->fixed_params->multiplier);
	}
}
#endif


/* INTERNAL FUNCTION
   Allocates the main structure and sets some default values.
 */
struct fann *fann_allocate_structure(unsigned int num_layers)
{
	struct fann *ann;

	if( num_layers == 0 )
	{
#ifdef DEBUG
		printf("less than 1 non input layer - ABORTING.\n");
#endif
		return NULL;
	}

	/* allocate the main network structure */
	ann = (struct fann *) malloc(sizeof(struct fann));
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	/* allocate space for algorithm parameters */
	ann->error           = (struct fann_error *) malloc(sizeof(struct fann_error));
	if(ann->error == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	ann->backprop_params = (struct fann_backprop_params *) malloc(sizeof(struct fann_backprop_params));
	if(ann->backprop_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	ann->cascade_params  = (struct fann_cascade_params *) malloc(sizeof(struct fann_cascade_params));
	if(ann->cascade_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	ann->rprop_params    = (struct fann_rprop_params *) malloc(sizeof(struct fann_rprop_params));
	if(ann->rprop_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	ann->som_params		 = (struct fann_som_params *) malloc(sizeof(struct fann_som_params));
	if(ann->som_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	ann->gng_params		 = (struct fann_gng_params *) malloc(sizeof(struct fann_gng_params));
	if(ann->gng_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	ann->training_params = (struct fann_training_params *) malloc(sizeof(struct fann_training_params));
	if(ann->training_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->scale_params = (struct fann_scale_params *) malloc(sizeof(struct fann_scale_params));
	if(ann->scale_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

#ifdef FIXEDFANN
	ann->fixed_params    = (struct fann_fixed_params *) malloc(sizeof(struct fann_fixed_params));
	if(ann->fixed_params == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
#endif

	/* initialize all parameters to defaults */
	ann->error->errno_f = FANN_E_NO_ERROR;
	ann->error->error_log = fann_default_error_log;
	ann->error->errstr = NULL;
	ann->backprop_params->learning_rate = 0.7f;
	ann->backprop_params->learning_momentum = 0.0;
	ann->num_input = 0;
	ann->num_output = 0;
	ann->training_params->training_algorithm = FANN_TRAIN_RPROP;
	ann->training_params->num_MSE = 0;
	ann->training_params->MSE_value = 0;
	ann->training_params->num_bit_fail = 0;
	ann->training_params->bit_fail_limit = (fann_type)0.35;
	ann->network_type = FANN_NETTYPE_LAYER;
	ann->training_params->train_error_function = FANN_ERRORFUNC_TANH;
	ann->training_params->train_stop_function = FANN_STOPFUNC_MSE;
	ann->training_params->callback = NULL;
	ann->user_data = NULL; /* User is responsible for deallocation */
	ann->output = NULL;
#ifndef FIXEDFANN
	ann->scale_params->scale_mean_in = NULL;
	ann->scale_params->scale_deviation_in = NULL;
	ann->scale_params->scale_new_min_in = NULL;
	ann->scale_params->scale_factor_in = NULL;
	ann->scale_params->scale_mean_out = NULL;
	ann->scale_params->scale_deviation_out = NULL;
	ann->scale_params->scale_new_min_out = NULL;
	ann->scale_params->scale_factor_out = NULL;
#endif	
	
	/* variables used for cascade correlation (reasonable defaults) */
	ann->cascade_params->cascade_output_change_fraction = 0.01f;
	ann->cascade_params->cascade_candidate_change_fraction = 0.01f;
	ann->cascade_params->cascade_output_stagnation_epochs = 12;
	ann->cascade_params->cascade_candidate_stagnation_epochs = 12;
	ann->cascade_params->cascade_num_candidate_groups = 2;
	ann->cascade_params->cascade_weight_multiplier = (fann_type)0.4;
	ann->cascade_params->cascade_candidate_limit = (fann_type)1000.0;
	ann->cascade_params->cascade_max_out_epochs = 150;
	ann->cascade_params->cascade_max_cand_epochs = 150;
	ann->cascade_params->cascade_min_out_epochs = 50;
	ann->cascade_params->cascade_min_cand_epochs = 50;
	ann->cascade_params->cascade_candidate_scores = NULL;
	ann->cascade_params->cascade_activation_functions_count = 10;
	ann->cascade_params->cascade_activation_functions = 
		(enum fann_activationfunc_enum *)
		calloc(ann->cascade_params->cascade_activation_functions_count, 
			   sizeof(enum fann_activationfunc_enum));
	if(ann->cascade_params->cascade_activation_functions == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}
							   
	ann->cascade_params->cascade_activation_functions[0] = FANN_SIGMOID;
	ann->cascade_params->cascade_activation_functions[1] = FANN_SIGMOID_SYMMETRIC;
	ann->cascade_params->cascade_activation_functions[2] = FANN_GAUSSIAN;
	ann->cascade_params->cascade_activation_functions[3] = FANN_GAUSSIAN_SYMMETRIC;
	ann->cascade_params->cascade_activation_functions[4] = FANN_ELLIOT;
	ann->cascade_params->cascade_activation_functions[5] = FANN_ELLIOT_SYMMETRIC;
	ann->cascade_params->cascade_activation_functions[6] = FANN_SIN_SYMMETRIC;
	ann->cascade_params->cascade_activation_functions[7] = FANN_COS_SYMMETRIC;
	ann->cascade_params->cascade_activation_functions[8] = FANN_SIN;
	ann->cascade_params->cascade_activation_functions[9] = FANN_COS;

	ann->cascade_params->cascade_activation_steepnesses_count = 4;
	ann->cascade_params->cascade_activation_steepnesses = 
		(fann_type *)
		calloc(ann->cascade_params->cascade_activation_steepnesses_count, 
			   sizeof(fann_type));
	if(ann->cascade_params->cascade_activation_steepnesses == NULL)
	{
		fann_safe_free(ann->cascade_params->cascade_activation_functions);
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}
	
	ann->cascade_params->cascade_activation_steepnesses[0] = (fann_type)0.25;
	ann->cascade_params->cascade_activation_steepnesses[1] = (fann_type)0.5;
	ann->cascade_params->cascade_activation_steepnesses[2] = (fann_type)0.75;
	ann->cascade_params->cascade_activation_steepnesses[3] = (fann_type)1.0;

	/* Variables for use with with Quickprop training (reasonable defaults) */
	ann->rprop_params->quickprop_decay = (float) -0.0001f;
	ann->rprop_params->quickprop_mu = 1.75;

	/* Variables for use with with RPROP training (reasonable defaults) */
	ann->rprop_params->rprop_increase_factor = (float) 1.2f;
	ann->rprop_params->rprop_decrease_factor = 0.5f;
	ann->rprop_params->rprop_delta_min = 0.0f;
	ann->rprop_params->rprop_delta_max = 50.0f;
	ann->rprop_params->rprop_delta_zero = 0.1f;
	
 	/* Variables for use with SARPROP training (reasonable defaults) */
 	ann->rprop_params->sarprop_weight_decay_shift = -6.644f;
 	ann->rprop_params->sarprop_step_error_threshold_factor = 0.1f;
 	ann->rprop_params->sarprop_step_error_shift = 1.385f;
 	ann->rprop_params->sarprop_temperature = 0.015f;
 	ann->rprop_params->sarprop_epoch = 0;

	/* Variables for use with Self-Organizing Maps */
	ann->som_params->som_width = 10;
	ann->som_params->som_height = 10;
 	ann->som_params->som_radius = 5;
	ann->som_params->som_learning_rate_constant = 0.01f;
	ann->som_params->som_topology = FANN_SOM_TOPOLOGY_RECTANGULAR;
	ann->som_params->som_neighborhood = FANN_SOM_NEIGHBORHOOD_GAUSSIAN;
	ann->som_params->som_learning_decay	= FANN_SOM_LEARNING_DECAY_LINEAR;

	/* Variables for use with Growing Neural Gas algorithm. These default values are good for many
   types of problems */
	ann->gng_params->gng_max_nodes = 100;
	ann->gng_params->gng_max_age = 100;
	ann->gng_params->gng_iteration_of_node_insert = 300;
	ann->gng_params->gng_local_error_reduction_factor = 0.05f;
	ann->gng_params->gng_global_error_reduction_factor = 0.0006f;
	ann->gng_params->gng_winner_node_scaling_factor = 0.5f;
	ann->gng_params->gng_neighbor_node_scaling_factor = 0.0005f;

	fann_init_error_data((struct fann_error *) ann->error);

#ifdef FIXEDFANN
	/* these values are only boring defaults, and should really
	 * never be used, since the real values are always loaded from a file. */
	ann->fixed_params->decimal_point = 8;
	ann->fixed_params->multiplier = 256;
#endif

	/* allocate room for the layers (the input one is not a proper layer)*/
	ann->first_layer = (struct fann_layer *) calloc(num_layers, sizeof(struct fann_layer));
	if(ann->first_layer == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}

	ann->last_layer = ann->first_layer + num_layers;

	return ann;
}


/* INTERNAL FUNCTION
   Allocates room for the scaling parameters.
 */
int fann_allocate_scale(struct fann *ann)
{
	/* todo this should only be allocated when needed */
#ifdef FIXEDFANN
	unsigned int i = 0;
#define SCALE_ALLOCATE( what, where, default_value )		    			\
		ann->scale_params->what##_##where = (float *)calloc(				\
			ann->num_##where##put,											\
			sizeof( float )													\
			);																\
		if( ann->scale_params->what##_##where == NULL )						\
		{																	\
			fann_error( NULL, FANN_E_CANT_ALLOCATE_MEM );					\
			fann_destroy( ann );                            				\
			return 1;														\
		}																	\
		for( i = 0; i < ann->num_##where##put; i++ )						\
			ann->scale_params->what##_##where[ i ] = ( default_value );

	SCALE_ALLOCATE( scale_mean,			in,		0.0 )
	SCALE_ALLOCATE( scale_deviation,	in,		1.0 )
	SCALE_ALLOCATE( scale_new_min,		in,		-1.0 )
	SCALE_ALLOCATE( scale_factor,		in,		1.0 )

	SCALE_ALLOCATE( scale_mean,			out,	0.0 )
	SCALE_ALLOCATE( scale_deviation,	out,	1.0 )
	SCALE_ALLOCATE( scale_new_min,		out,	-1.0 )
	SCALE_ALLOCATE( scale_factor,		out,	1.0 )
#undef SCALE_ALLOCATE
#endif	
	return 0;
}

/* INTERNAL FUNCTION
   Seed the random function.
 */
void fann_seed_rand()
{
#ifndef _WIN32
	FILE *fp = fopen("/dev/urandom", "r");
	unsigned int foo;
	struct timeval t;

	if(!fp)
	{
		gettimeofday(&t, NULL);
		foo = t.tv_usec;
#ifdef DEBUG
		printf("unable to open /dev/urandom\n");
#endif
	}
	else
	{
		fread(&foo, sizeof(foo), 1, fp);
		fclose(fp);
	}
	srand(foo);
#else
	/* COMPAT_TIME REPLACEMENT */
	srand(GetTickCount());
#endif
}


/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap
 */
