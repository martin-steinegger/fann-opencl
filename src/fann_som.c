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
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "config.h"
#include "fann.h"
#include "fann_som.h"
#include "string.h"
#include "fann_train.h"


/* #define FANN_SOM_DEBUG */
/* #define FANN_SOM_DEBUG_FULL */

#define FANN_SOM_ROUND(X) ((int) ((X) + 0.5))

/* Setters and Getters for parameters */
FANN_GET_SETP(unsigned int, som_params, som_width)
FANN_GET_SETP(unsigned int, som_params, som_height)
FANN_GET_SETP(float, som_params, som_radius)
FANN_GET_SETP(enum fann_som_topology_enum, som_params, som_topology)
FANN_GET_SETP(enum fann_som_neighborhood_enum, som_params, som_neighborhood)
FANN_GET_SETP(enum fann_som_learning_decay_enum, som_params, som_learning_decay)


FANN_EXTERNAL struct fann *FANN_API fann_create_som(unsigned int width, unsigned int height, unsigned int num_input)
{
   	struct fann_layer_descr *layer_descr;
	struct fann *ann;

	/* Create network descriptor */
	struct fann_descr *descr=(struct fann_descr*) calloc(1, sizeof(struct fann_descr));
	if(descr == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	descr->num_layers = 1;
	descr->num_inputs = num_input;

	/* Create layer descriptors */
	descr->layers_descr=(struct fann_layer_descr*) calloc(1, sizeof(struct fann_layer_descr));
	if(descr->layers_descr == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	layer_descr = descr->layers_descr;
	layer_descr->num_neurons = 1;
	layer_descr->constructor = fann_som_layer_constructor;
	layer_descr->private_data = NULL;

	/* Create neuron descriptors */
	layer_descr->neurons_descr = (struct fann_neuron_descr*) calloc(layer_descr->num_neurons, sizeof(struct fann_neuron_descr));
	if(layer_descr->neurons_descr == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	layer_descr->neurons_descr[0].num_outputs = width * height;
	layer_descr->neurons_descr[0].constructor = fann_som_neuron_constructor;
	layer_descr->neurons_descr[0].private_data = NULL;


	ann = fann_create_from_descr(descr);
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	/* free up the memory from all the descriptors */

	fann_safe_free(layer_descr->neurons_descr);
	fann_safe_free(layer_descr);
	fann_safe_free(descr);
	
	/* Set som parameters */
	ann->network_type = FANN_NETTYPE_SOM;
	ann->som_params->som_width = width;
	ann->som_params->som_height = height;
	ann->num_input = num_input;
	ann->training_params->training_algorithm = FANN_TRAIN_SOM;
	
	/* Just a initial value that is not too large */
 	ann->som_params->som_radius = width;
	ann->som_params->som_current_radius = width;

	fann_randomize_weights_som(ann, 0.0, 1.0);

	return ann;
}

FANN_EXTERNAL int FANN_API fann_som_layer_constructor(struct fann *ann, 
		struct fann_layer *layer, struct fann_layer_descr *descr)
{

	struct fann_neuron_descr *neurons_descr = descr->neurons_descr;
	fann_type *free_output;
	struct fann_neuron *n;

	layer->num_neurons = descr->num_neurons;
	layer->num_outputs = neurons_descr[0].num_outputs;
	layer->run = fann_som_layer_run;

	layer->train_errors=NULL;

	/* allocate the outputs array */
	free_output = layer->outputs = (fann_type*) calloc(layer->num_outputs,sizeof(fann_type));
	if( layer->outputs == NULL)
	{
		return 1;
	}

	/* allocate the neurons array */
	layer->first_neuron = (struct fann_neuron *) calloc(layer->num_neurons, sizeof(struct fann_neuron));
	{
	        if (layer->first_neuron == NULL)
		{
		        return 1;
		}
	}
	layer->last_neuron = layer->first_neuron + layer->num_neurons;
	n = layer->first_neuron;
	n->outputs = free_output;
	if(neurons_descr[0].constructor(ann, layer, n, neurons_descr) != 0)
	{
		return 1;
	}

	/* Set the other function pointers */
	layer->destructor = fann_som_layer_destructor;
	layer->initialize_train_errors = fann_som_layer_train_initialize;
	return 0;
}

FANN_EXTERNAL void FANN_API fann_som_layer_run(struct fann *ann, struct fann_layer* layer)
{
        struct fann_neuron * last_neuron = layer->last_neuron;
	struct fann_neuron * neuron_it;
	for(neuron_it = layer->first_neuron; neuron_it != last_neuron; neuron_it++)
    		neuron_it->run(ann, neuron_it);
}

FANN_EXTERNAL void FANN_API fann_som_layer_destructor(struct fann_layer* layer) {
        
        layer->first_neuron->destructor(layer->first_neuron);
	fann_safe_free(layer->first_neuron);
	fann_safe_free(layer->train_errors);
	fann_safe_free(layer->outputs);
}

FANN_EXTERNAL void FANN_API fann_som_layer_train_initialize(struct fann *ann, struct fann_layer *layer) {

  /* Does nothing */
}



FANN_EXTERNAL int FANN_API fann_som_neuron_constructor(struct fann *ann, struct fann_layer *layer, 
		struct fann_neuron *neuron, struct fann_neuron_descr * descr)
{
	neuron->num_outputs = descr->num_outputs;
	neuron->inputs = layer->inputs;
	neuron->num_inputs = layer->num_inputs;

	/* set the error array to null (lazy allocation) */
	neuron->train_errors=NULL;

	/* allocate the model vector */
	neuron->private_data = calloc(1, sizeof(struct fann_som_neuron_private_data));
	if (neuron->private_data == NULL)
	{
	 	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return 1; 
	}
	((struct fann_som_neuron_private_data *)(neuron->private_data))->som_model_vector = (fann_type *)calloc((layer->num_inputs - 1)* descr->num_outputs, sizeof(fann_type));
	if(((struct fann_som_neuron_private_data *)neuron->private_data)->som_model_vector == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return 1;
	}

	neuron->destructor = fann_som_neuron_destructor;
	neuron->run = fann_som_neuron_run;
	neuron->backpropagate = fann_som_neuron_backprop;
	neuron->update_weights = fann_som_neuron_update;
	neuron->compute_error = fann_som_compute_MSE;

	return 0;
}


FANN_EXTERNAL void FANN_API fann_som_neuron_destructor(struct fann_neuron* neuron)
{
	struct fann_som_neuron_private_data *priv;
	priv=(struct fann_som_neuron_private_data *) neuron->private_data;
	fann_safe_free(priv->som_model_vector);
	fann_safe_free(priv);
}


FANN_EXTERNAL struct fann* FANN_API fann_copy_som(const struct fann* orig) 
{
	struct fann *copy = NULL;
	printf("Not currently implemented!\n");
	return copy;
}


/* Randomize the weights of the som */
FANN_EXTERNAL void FANN_API fann_randomize_weights_som(struct fann *ann, fann_type min_weight, fann_type max_weight)
{
        struct fann_neuron *current_neuron;
	unsigned int num_weight_vals, i;
	struct fann_som_neuron_private_data *priv;

	if (fann_get_network_type(ann) != FANN_NETTYPE_SOM)
	{
	          printf("Invalid network type in fann_randomize_weights_som!\n");
        	  exit(1);
	}

        current_neuron = ann->first_layer->first_neuron;
	num_weight_vals = ann->num_input * ann->som_params->som_width * ann->som_params->som_height;
	priv = (struct fann_som_neuron_private_data *) current_neuron->private_data;

	for (i = 0; i < num_weight_vals; i++)
	{
	  
	        priv->som_model_vector[i] = fann_rand(min_weight, max_weight);
	}

}


/* Initializes weight values with training data */
FANN_EXTERNAL void FANN_API fann_init_weights_som(struct fann *ann, struct fann_train_data *train_data) {
        int i, w, h, numinput, j;
	unsigned int cur_data_index = 0;
        struct fann_neuron *current_neuron;
	struct fann_som_neuron_private_data *priv;
	fann_type *cur_modelvector;

	if (fann_get_network_type(ann) != FANN_NETTYPE_SOM)
	{
	          printf("Invalid network type in fann_randomize_weights_som!\n");
        	  exit(1);
	}

	w = ann->som_params->som_width;
	h = ann->som_params->som_height;
	numinput = ann->num_input;
        current_neuron = ann->first_layer->first_neuron;
	priv = (struct fann_som_neuron_private_data *) current_neuron->private_data;

	for (i = 0; i < w * h; i ++)
	{
	        cur_modelvector = &priv->som_model_vector[i * numinput];
		for (j = 0; j < numinput; j++) 
		{
	                cur_modelvector[j] = train_data->input[cur_data_index][j];
		}
		cur_data_index++;
		if (cur_data_index > train_data->num_data)
		{
	                cur_data_index = 0;
		}  
	}

}


FANN_EXTERNAL void FANN_API fann_print_connections_som(struct fann *ann)
{
	printf("No SOM connections!\n");
}

FANN_EXTERNAL void FANN_API fann_print_parameters_som(struct fann *ann)
{
	printf("Printing SOM parameters!\n");
        printf("Self-Organizing Maps width           :%4d\n", fann_get_som_width(ann));
        printf("Self-Organizing Maps height          :%4d\n", fann_get_som_height(ann));
        printf("Self-Organizing Maps radius          :%8.3f\n", fann_get_som_radius(ann));
        printf("Self-Organizing Maps toplogy         :%s\n", FANN_SOM_TOPOLOGY_NAMES[fann_get_som_topology(ann)]);
        printf("Self-Organizing Maps neighborhood    :%s\n", FANN_SOM_NEIGHBORHOOD_NAMES[fann_get_som_neighborhood(ann)]);
        printf("Self-Organizing Maps learning decay  :%s\n", FANN_SOM_LEARNING_DECAY_NAMES[fann_get_som_learning_decay(ann)]);

}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_input_som(struct fann *ann)
{
        return ann->num_input;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_output_som(struct fann *ann)
{
        return ann->num_output;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons_som(struct fann *ann)
{
	return 1;  
}

FANN_EXTERNAL unsigned int FANN_API fann_get_total_connections_som(struct fann *ann)
{
        return 0;
}

FANN_EXTERNAL enum fann_nettype_enum FANN_API fann_get_network_type_som(struct fann *ann)
{
        return ann->network_type;
}


/* Som neuron run callback */
FANN_EXTERNAL void FANN_API fann_som_neuron_run(struct fann * ann, struct fann_neuron *neuron)
{
        fann_type *cur_modelvector;
	double runsum;
	int i, j, numinputs;
	int numcells = ann->som_params->som_width * ann->som_params->som_height;
	double min_dist = FLT_MAX;
	struct fann_som_neuron_private_data *priv;
	priv=(struct fann_som_neuron_private_data *) neuron->private_data;
 
	ann->som_params->som_closest_index = -1;
	numinputs = ann->num_input;

	/* compute the distance at each cell */
	for (i = 0; i < numcells; i++)
	{
                runsum = 0;
		cur_modelvector = &priv->som_model_vector[i * numinputs];
		for (j = 0; j < numinputs; j++)
		{
		  runsum += (cur_modelvector[j] - neuron->inputs[j]) * (cur_modelvector[j] - neuron->inputs[j]);
		}

		/* Just keep track of the squared value to speed things up a bit */
	        neuron->outputs[i] = runsum;

		/* Keep track of the cell with the minimum distance */
		if (neuron->outputs[i] < min_dist)
		{
	                min_dist = neuron->outputs[i];
			ann->som_params->som_closest_index = i;
			ann->som_params->min_dist = min_dist;
		}
	}

}

FANN_EXTERNAL void FANN_API fann_som_compute_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type *desired_output)
{
        /* Not used */
}


FANN_EXTERNAL void FANN_API fann_som_neuron_backprop(struct fann *ann, struct fann_neuron *neuron, fann_type *prev_layer_errors)
{
        /* No backprop in SOMs*/
}

/* Som Neuron update callback */
FANN_EXTERNAL void FANN_API fann_som_neuron_update(struct fann *ann, struct fann_neuron *neuron)
{


        if (ann->som_params->som_topology == FANN_SOM_TOPOLOGY_RECTANGULAR)
        {
                fann_update_rectangular_topology(ann, neuron);
        }
        else if (ann->som_params->som_topology == FANN_SOM_TOPOLOGY_HEXAGONAL)
        {
                fann_update_hexagonal_topology(ann, neuron);
        }
}



/* INTERNAL FUNCTION
   update weights for a som with a rectangular topology
*/
void fann_update_rectangular_topology(struct fann *ann, struct fann_neuron *neuron)
{
        int x, y, cx, cy, i;
	int lx, rx, ty, by, cur_radiusi;
	int somwidth, somheight, numinputs;
	fann_type *cur_modelvector;
	fann_type *modelvector,*inputvector;
	float learningrate;
	float cur_radiusf, dist, neighborhoodweight, radius;
	struct fann_som_neuron_private_data *priv;
	priv = (struct fann_som_neuron_private_data *)neuron->private_data;
	
	somwidth = ann->som_params->som_width;
	somheight = ann->som_params->som_height;
	numinputs = ann->num_input;
	modelvector = priv->som_model_vector;
	inputvector = ann->inputs;
	learningrate = ann->som_params->som_learning_rate;
	radius = ann->som_params->som_radius * ann->som_params->som_radius;
	
	/* calculate the current x/y position of the closest cell */
	cx = ann->som_params->som_closest_index % somwidth;
	cy = ann->som_params->som_closest_index / somwidth;




	/* update the weights using the distance based neighborhood metric */
	if (ann->som_params->som_neighborhood == FANN_SOM_NEIGHBORHOOD_DISTANCE)
	{
              cur_radiusi = FANN_SOM_ROUND(ann->som_params->som_current_radius);

	      /* calculate the boundry conditions ahead of time to speed up weight adjustment */
	      lx = cx - cur_radiusi;
	      if (lx < 0)
	      {
	              lx = 0;
	      }
	      rx = cx + cur_radiusi;
	      if (rx > somwidth - 1)
	      {
		      rx = somwidth - 1;
	      }
	      ty = cy - cur_radiusi;
	      if (ty < 0)
	      {
		      ty = 0;
	      }
	      by = cy + cur_radiusi;
	      if (by > somheight - 1)
	      {
		      by = somheight - 1;
	      }
	      

	      /* main weight adjustment loop */
	      for (y = ty; y <= by; y++)
	      {
	              for (x = lx; x <= rx; x++)
  		      {
		              if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius )
			      {
			              cur_modelvector = &modelvector[(y * somwidth + x) * numinputs];
				      for (i = 0; i < numinputs; i++)
				      {
					      cur_modelvector[i] += (learningrate * (inputvector[i] - cur_modelvector[i]));
				      }
			      }
		      }
	      }
	}

	/* adjust weights using the gaussian neighborhood kernel */
	else if (ann->som_params->som_neighborhood == FANN_SOM_NEIGHBORHOOD_GAUSSIAN)
        {
	        cur_radiusf = ann->som_params->som_current_radius;      

		/* main weight adjustment loop */
		for (y = 0; y < somheight; y++)
		{
		        for (x = 0; x < somwidth; x++) 
			{
			        cur_modelvector = &modelvector[(y * somwidth + x) * numinputs];

				dist = sqrt(pow((cx - x), 2) + pow((cy - y), 2));

				/* Adjust model vector weights */
				neighborhoodweight =  exp(-((dist * dist) / (2 * cur_radiusf * cur_radiusf)));
				for (i = 0; i < numinputs; i++)
				{
				        cur_modelvector[i] += neighborhoodweight * learningrate * (inputvector[i] - cur_modelvector[i]);
				} 
			}
		}
	}

}




/* INTERNAL FUNCTION
   update weights for a som with a hexagonal topology
*/
void fann_update_hexagonal_topology(struct fann *ann, struct fann_neuron *neuron)
{
        int x, y, cx, cy, i;
	int lx, rx, ty, by, cur_radiusi;
	int somwidth, somheight, numinputs;
	fann_type *cur_modelvector;
	fann_type *modelvector,*inputvector;
	float learningrate;
	float radius, tempd, dist, neighborhoodweight;
	struct fann_som_neuron_private_data *priv;
	priv = (struct fann_som_neuron_private_data *)neuron->private_data;	

	somwidth = ann->som_params->som_width;
	somheight = ann->som_params->som_height;
	numinputs = ann->num_input;
	modelvector = priv->som_model_vector;
	inputvector = neuron->inputs;
	learningrate = ann->som_params->som_learning_rate;
	radius = ann->som_params->som_current_radius;

	/* calculate the current x/y position of the closest cell */
	cx = ann->som_params->som_closest_index % somwidth;
	cy = ann->som_params->som_closest_index / somwidth;
	
	if (ann->som_params->som_neighborhood == FANN_SOM_NEIGHBORHOOD_DISTANCE)
	{
	    
		/* calculate the boundry conditions ahead of time to speed up weight adjustment */
	        cur_radiusi = FANN_SOM_ROUND(ann->som_params->som_current_radius);
		
		lx = cx - cur_radiusi;
		if (lx < 0)
	        {
	                lx = 0;
  	        }
		rx = cx + cur_radiusi;
		if (rx > somwidth - 1)
	        {
		        rx = somwidth - 1;
		}
		ty = cy - cur_radiusi;
		if (ty < 0)
	        {
		        ty = 0;
	        }
		by = cy + cur_radiusi;
		if (by > somheight - 1)
		{
		        by = somheight - 1;
		}

		/* main weight adjustment loop */
		for (y = 0; y < somheight; y++)
		{
		        for (x = 0; x < somwidth; x++)
		        {

			        /*  Calculate distances in the hex grid */
			        tempd = cx - x;
				if ((cy - y) % 2) 
				{
				        if (!(cy % 2))
					{
					        tempd -= 0.5;
					}
					else 
					{
					        tempd += 0.5;
					}
				}

				dist = tempd * tempd;
				tempd = cy - y;
				dist += 0.75 * tempd * tempd;
				
				dist = (float)sqrt(dist);

				if (dist <= radius) 
				{
				         cur_modelvector = &modelvector[(y * somwidth + x) * numinputs];

					 /* Adjust the model vector */
      				         for (i = 0; i < numinputs; i++)
				         {
				                 cur_modelvector[i] += learningrate * (inputvector[i] - cur_modelvector[i]);
				         }
				}
			}
		}
	}

	else if (ann->som_params->som_neighborhood == FANN_SOM_NEIGHBORHOOD_GAUSSIAN)
	{

	  /* main weight adjustment loop */
		for (y = 0; y < somheight; y++)
		{
		        for (x = 0; x < somwidth; x++) 
			{
			        cur_modelvector = &modelvector[(y * somwidth + x) * numinputs];

				/*  Calculate distances in the hex grid */
			        tempd = cx - x;
				if ((cy - y) % 2) 
				{
				        if (!(cy % 2))
					{
					        tempd -= 0.5;
					}
					else 
					{
					        tempd += 0.5;
					}
				}

				dist = tempd * tempd;
				tempd = cy - y;
				dist += 0.75 * tempd * tempd;
				
				dist = (float)sqrt(dist);

				/* adjust the model vector */
				neighborhoodweight =  exp((double)-((dist * dist) / (2 * radius * radius)));
				for (i = 0; i < numinputs; i++)
				{
				        cur_modelvector[i] += neighborhoodweight * learningrate * (inputvector[i] - cur_modelvector[i]);
				} 
				
			}
		}
	}
 }


/* Reduce the learning rate and neighborhood radius after each example */
void fann_som_decay(struct fann *ann, int epoch, int max_epoch)
{
        float invconst;

	/* Reduce neighborhood radius linearly */
        ann->som_params->som_current_radius = 1 + (ann->som_params->som_radius - 1) * (float)(max_epoch - epoch) / (float)max_epoch;

	/* Reduce learning rate */
	if (ann->som_params->som_learning_decay == FANN_SOM_LEARNING_DECAY_LINEAR)
	{
	        ann->som_params->som_learning_rate = ann->som_params->som_learning_rate_constant * ((float)(max_epoch - epoch)) / ((float)max_epoch);
	}
	else if (ann->som_params->som_learning_decay == FANN_SOM_LEARNING_DECAY_INVERSE)
	{
	        invconst = (float)epoch  / (float)((float)max_epoch / 100.0f);
	        ann->som_params->som_learning_rate = (float)(ann->som_params->som_learning_rate_constant * invconst / (invconst + epoch));
	}
}

/* Calculate error rates over the entire training set */
FANN_EXTERNAL float FANN_API fann_get_MSE_som(struct fann *ann, struct fann_train_data *data)
{
        unsigned int i;
	double cur_error = 0;

	for (i = 0; i < data->num_data; i++) 
	{
                fann_run(ann, data->input[i]);
		cur_error += sqrt(ann->output[ann->som_params->som_closest_index]);
	}
	return cur_error / (float)(data->num_data);
}

#ifndef FIXEDFANN
FANN_EXTERNAL void FANN_API fann_train_example_som(struct fann *ann, struct fann_train_data *data, unsigned int example_num, unsigned int max_examples)
{

        fann_train(ann, data->input[example_num % data->num_data], NULL);
	fann_som_decay(ann, example_num, max_examples);
}

FANN_EXTERNAL void FANN_API fann_train_example_array_som(struct fann *ann, fann_type *data, unsigned int example_num, unsigned int max_examples) {
        fann_train(ann, data, NULL);
        fann_som_decay(ann, example_num, max_examples);
}


/* INTERNAL FUNCTION
   Train a som using the available data
*/
void fann_train_on_data_som(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error) 
{
        unsigned int epoch_cnt = 0;
	float cur_error = FLT_MAX;
	unsigned int cur_data_index = 0;


	/* In a the SOM, we treat each training example given to the SOM as an epoch */
	while ((epoch_cnt != max_epochs) && (cur_error > desired_error))
	{
                fann_train(ann, data->input[cur_data_index], NULL);
		cur_data_index++;
		epoch_cnt++;
		if (cur_data_index == data->num_data)
		{
	                cur_data_index = 0;
		}

		/* After each training example, the learning rate and neighborhood decay */
		fann_som_decay(ann, epoch_cnt, max_epochs);

		/* Generate reports after a certain number of training examples */
		if ((epoch_cnt % epochs_between_reports) == 0)
		{
		        cur_error = fann_get_MSE_som(ann, data);
			if (ann->training_params->callback == NULL) {
			        printf("Epochs     %8d. Current error: %.10f.\n", epoch_cnt, cur_error);
			}
			else if(((*ann->training_params->callback)(ann, data, max_epochs, epochs_between_reports, desired_error, epoch_cnt)) == -1)
			{
		                break;
			}		
		}
	}
}
#endif


/* INTERNAL FUNCTION
   Debug function to save the som part of the network
*/
 void fann_save_som_to_file(struct fann *ann, FILE *conf)
 {
         struct fann_neuron *current_neuron;
	 unsigned int i, j;
	 fann_type *cur_modelvector;
	 struct fann_som_neuron_private_data *priv;
	 unsigned int num_weight_val = ann->num_input;
	 
	 current_neuron = ann->first_layer->first_neuron;
	 priv = (struct fann_som_neuron_private_data *)current_neuron->private_data;
	 
	 /* Dump the som parameters */
	 fprintf(conf, "%d,%d,%f,%f,%d,%d,%d\n", ann->som_params->som_width, ann->som_params->som_height, ann->som_params->som_radius, ann->som_params->som_current_radius, ann->som_params->som_topology, ann->som_params->som_neighborhood, ann->som_params->som_learning_decay);
	 
	 
	 /* dump the weights */
	 for (j = 0; j < ann->som_params->som_height * ann->som_params->som_width; j++)
	 {
                 cur_modelvector = &priv->som_model_vector[j * num_weight_val];
		 for (i = 0; i < num_weight_val; i++)
                 {
		         fprintf(conf,"%f ", (float)cur_modelvector[i]);
		 }
		 fprintf(conf,"\n");
	 }
 }


FANN_EXTERNAL void FANN_API fann_set_som_config(struct fann *ann, struct fann_som_params *som_params) {

	(*ann->som_params) = *(som_params);
}
	


