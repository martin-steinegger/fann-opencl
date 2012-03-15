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
#include <limits.h>

#include "config.h"
#include "fann.h"
#include "fann_gng.h"
#include "string.h"
#include "fann_train.h"

/* #define FANN_GNG_DEBUG */
/* #define FANN_GNG_DEBUG_FULL */

/* Setters and Getters for parameters */
FANN_GET_SETP(unsigned int, gng_params, gng_max_nodes)
FANN_GET_SETP(unsigned int, gng_params, gng_max_age)
FANN_GET_SETP(unsigned int, gng_params, gng_iteration_of_node_insert)
FANN_GET_SETP(float, gng_params, gng_local_error_reduction_factor)
FANN_GET_SETP(float, gng_params, gng_global_error_reduction_factor)
FANN_GET_SETP(float, gng_params, gng_winner_node_scaling_factor)
FANN_GET_SETP(float, gng_params, gng_neighbor_node_scaling_factor)


FANN_EXTERNAL struct fann *FANN_API fann_create_gng(unsigned int num_input)
{ 
	struct fann_layer_descr *layer_descr;
	struct fann *ann;	
	unsigned int i,j;
	struct fann_neuron *neuron;

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
	layer_descr->constructor = fann_gng_layer_constructor;
	layer_descr->private_data = NULL;

	/* Create neuron descriptors */
	layer_descr->neurons_descr = (struct fann_neuron_descr*) calloc(layer_descr->num_neurons, sizeof(struct fann_neuron_descr));
	if(layer_descr->neurons_descr == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	layer_descr->neurons_descr[0].num_outputs = 2;
	layer_descr->neurons_descr[0].constructor = fann_gng_neuron_constructor;
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

	/* The GNG specific initilization */	
	ann->network_type = FANN_NETTYPE_GNG;
	ann->num_input = num_input;
	ann->gng_params->gng_num_cells = 2;
	ann->training_params->training_algorithm = FANN_TRAIN_GNG;
	ann->gng_params->min_dist = FLT_MAX;

        neuron = ann->first_layer->first_neuron;      

	/* Allocate memory for the data within private data */
	if( (((struct fann_gng_neuron_private_data *)(neuron->private_data))->gng_cell_error = (fann_type*) calloc(ann->gng_params->gng_num_cells,sizeof(fann_type)) ) == NULL)
	{
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    return NULL;
	}

	/* Allocate memory for the cell locations. Cell locations are a table/matrix.The table is cells by dimensions */
	if( (((struct fann_gng_neuron_private_data *)(neuron->private_data))->gng_cell_location = (fann_type**) calloc(ann->gng_params->gng_num_cells,sizeof(fann_type*)) ) == NULL)
	{
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    return NULL;
	}

	for(i=0;i<ann->gng_params->gng_num_cells;i++) 
	{	
	  if( (((struct fann_gng_neuron_private_data *)(neuron->private_data))->gng_cell_location[i] = (fann_type*) calloc(ann->num_input,sizeof(fann_type))) == NULL)
	  {
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    return NULL;
	  }
	}

	/* Allocate memory to record the edges. We use a matrix/table of cells by cells.*/
	if( (ann->gng_params->gng_cell_edges = (int**) calloc(ann->gng_params->gng_num_cells,sizeof(int*))) == NULL)
	{
	  fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	  return NULL;
	}

	for(i=0;i<ann->gng_params->gng_num_cells;i++) 
	{	
	  if( (ann->gng_params->gng_cell_edges[i] = (int*) calloc(ann->gng_params->gng_num_cells,sizeof(int))) == NULL)
	  {
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    return NULL;
	  }
	}

	/* Set the values to zero*/
	for(i=0;i<ann->gng_params->gng_num_cells;i++) 
	{
	  for(j=0;j<ann->gng_params->gng_num_cells;j++)
	  {
	    ann->gng_params->gng_cell_edges[i][j] = -1;
	  }
	}

	((struct fann_gng_neuron_private_data *)(neuron->private_data))->gng_num_cells = ann->gng_params->gng_num_cells;	
	ann->gng_params->gng_current_iteration = 0;
	fann_randomize_weights_gng(ann, 0.0, 1.0);

	return ann;
}

FANN_EXTERNAL int FANN_API fann_gng_layer_constructor(struct fann *ann, 
		struct fann_layer *layer, struct fann_layer_descr *descr)
{

	struct fann_neuron_descr *neurons_descr = descr->neurons_descr;
	fann_type *free_output;
	struct fann_neuron *n;

	layer->num_neurons = descr->num_neurons;
	layer->num_outputs = neurons_descr[0].num_outputs;
	layer->run = fann_gng_layer_run;

	layer->train_errors = NULL;

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
	layer->destructor = fann_gng_layer_destructor;
	layer->initialize_train_errors = fann_gng_layer_train_initialize;
	return 0;
}

FANN_EXTERNAL void FANN_API fann_gng_layer_run(struct fann *ann, struct fann_layer* layer) {
        struct fann_neuron * last_neuron = layer->last_neuron;
        struct fann_neuron * neuron_it;
        for(neuron_it = layer->first_neuron; neuron_it != last_neuron; neuron_it++)
                neuron_it->run(ann, neuron_it);
}


FANN_EXTERNAL void FANN_API fann_gng_layer_destructor(struct fann_layer* layer) 
{        
        layer->first_neuron->destructor(layer->first_neuron);
	fann_safe_free(layer->first_neuron);
	fann_safe_free(layer->train_errors);
	fann_safe_free(layer->outputs);
}

FANN_EXTERNAL void FANN_API fann_gng_layer_train_initialize(struct fann *ann, struct fann_layer *layer) 
{
        /* Not used */
        return;
}

FANN_EXTERNAL int FANN_API fann_gng_neuron_constructor(struct fann *ann, struct fann_layer *layer, 
		struct fann_neuron *neuron, struct fann_neuron_descr * descr)
{
	neuron->num_outputs = descr->num_outputs;
	neuron->inputs = layer->inputs;
	neuron->num_inputs = layer->num_inputs;

	/* set the error array to null (lazy allocation) */
	neuron->train_errors=NULL;

	/* allocate the private data */
	neuron->private_data = calloc(1, sizeof(struct fann_gng_neuron_private_data));
	if (neuron->private_data == NULL)
	{
	 	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return 1; 
	}

	neuron->destructor = fann_gng_neuron_destructor;
	neuron->run = fann_gng_neuron_run;
	neuron->backpropagate = fann_gng_neuron_backprop;
	neuron->update_weights = fann_gng_neuron_update;
	neuron->compute_error = fann_gng_compute_MSE;

	return 0;
}

FANN_EXTERNAL void FANN_API fann_gng_neuron_destructor(struct fann_neuron* neuron)
{
        unsigned int i;
	struct fann_gng_neuron_private_data *priv;
	priv=(struct fann_gng_neuron_private_data *) neuron->private_data;	
	fann_safe_free(priv->gng_cell_error); 
	for(i=0;i<priv->gng_num_cells;i++) 
        {
	  fann_safe_free(priv->gng_cell_location[i]); 	
	}
	fann_safe_free(priv->gng_cell_location); 	
}


FANN_EXTERNAL struct fann* FANN_API fann_copy_gng(const struct fann* orig) 
{
	printf("Not currently implemented!\n");
	return NULL;
}

FANN_EXTERNAL void FANN_API fann_randomize_weights_gng(struct fann *ann, fann_type min_weight, fann_type max_weight)
{
     struct fann_neuron *current_neuron;
     unsigned int i,j;
     struct fann_gng_neuron_private_data *priv;

     if (fann_get_network_type(ann) != FANN_NETTYPE_GNG)
     {
         printf("Invalid network type in fann_init_weights_gng!\n");
         return;
     }

     current_neuron = ann->first_layer->first_neuron;
     priv=(struct fann_gng_neuron_private_data *) current_neuron->private_data;

      for (i = 0; i < ann->gng_params->gng_num_cells; i++)
      {
         for(j= 0; j < ann->num_input;j++ )
         {
	   priv->gng_cell_location[i][j] = fann_rand(min_weight, max_weight);
         }
      }
}

FANN_EXTERNAL void FANN_API fann_init_weights_gng(struct fann *ann, struct fann_train_data *train_data)
{ 
      fann_type* smallest_inp_vec,*largest_inp_vec;
      unsigned int i,j;
      struct fann_gng_neuron_private_data *priv;  
      struct fann_neuron *current_neuron;

      if (fann_get_network_type(ann) != FANN_NETTYPE_GNG)
      {
         printf("Invalid network type in fann_init_weights_gng!\n");
         return;
      }
      
      current_neuron = ann->first_layer->first_neuron;
      priv=(struct fann_gng_neuron_private_data *) current_neuron->private_data;
   
      smallest_inp_vec = NULL;
      largest_inp_vec = NULL;   
    
      if( (smallest_inp_vec = (fann_type*) calloc(train_data->num_input,sizeof(fann_type)) ) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      }

      if( (largest_inp_vec = (fann_type*) calloc(train_data->num_input,sizeof(fann_type)) ) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      }

      for(j=0; j<train_data->num_input; j++)
      {
	smallest_inp_vec[j] = 1000000;
	largest_inp_vec[j] = -1000000;	
      }
   
      /* Iterate throught the data set. We're looking for the largest and smallest value of each dimension
	 of the input space*/
      for(i=0;i<train_data->num_data;i++) 
      {
	for(j=0; j<train_data->num_input; j++)
	{
	  if(train_data->input[i][j] < smallest_inp_vec[j])
	    smallest_inp_vec[j] = train_data->input[i][j];
	  if(train_data->input[i][j] > largest_inp_vec[j])
	    largest_inp_vec[j] = train_data->input[i][j];	
	}		
      }  

      /* Now init the location with a random number for each dimension IN the input space.*/
      for (i = 0; i < ann->gng_params->gng_num_cells; i++)
      {
        for(j= 0; j < ann->num_input;j++ )
        {
	  priv->gng_cell_location[i][j] = fann_rand(smallest_inp_vec[j], largest_inp_vec[j]);
        }
      }  
      fann_safe_free(smallest_inp_vec);   
      fann_safe_free(largest_inp_vec); 
}

FANN_EXTERNAL void FANN_API fann_print_connections_gng(struct fann *ann)
{
        unsigned int i,j;
        printf("Printing GNG edge matrix:\n");
        printf("Values indicate edge age, -1 means no edge\n\n");
	for (i = 0; i < ann->gng_params->gng_num_cells; i++)
        {	  	  
           for(j= 0; j < ann->gng_params->gng_num_cells;j++ )
           {
	      printf("%i\t", ann->gng_params->gng_cell_edges[i][j]);
           } 
	   printf("\n");
        }
}

FANN_EXTERNAL void FANN_API fann_print_parameters_gng(struct fann *ann)
{
        printf("Printing GNG parameters:\n");
        printf("\tGrowing Neural Gas maximium nodes                               :%d\n",ann->gng_params->gng_max_nodes);
        printf("\tGrowing Neural Gas maximium node age (a sub max)                :%d\n",ann->gng_params->gng_max_age);
        printf("\tGrowing Neural Gas scaling winner node (e sub b)                :%1.3f\n",ann->gng_params->gng_winner_node_scaling_factor);
        printf("\tGrowing Neural Gas scaling neighbor nodes (e sub n)             :%1.3f\n",ann->gng_params->gng_neighbor_node_scaling_factor);
        printf("\tGrowing Neural Gas node insertion iteration (lambda)            :%d\n",ann->gng_params->gng_iteration_of_node_insert);
        printf("\tGrowing Neural Gas local error variable decrease factor (d)     :%1.3f\n",ann->gng_params->gng_local_error_reduction_factor);
        printf("\tGrowing Neural Gas global error variable decrease factor (d)    :%1.3f\n",ann->gng_params->gng_global_error_reduction_factor);
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_input_gng(struct fann *ann)
{
        return ann->num_input;
} 

FANN_EXTERNAL unsigned int FANN_API fann_get_num_output_gng(struct fann *ann)
{
	return ann->num_output;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons_gng(struct fann *ann)
{
	return 1; 
}

FANN_EXTERNAL unsigned int FANN_API fann_get_total_connections_gng(struct fann *ann)
{
	return 0;
}

FANN_EXTERNAL enum fann_nettype_enum FANN_API fann_get_network_type_gng(struct fann *ann)
{
	return ann->network_type;
}

FANN_EXTERNAL void FANN_API fann_gng_neuron_run(struct fann * ann, struct fann_neuron *neuron)
{
	double runsum;
	double sec_min_dist = DBL_MAX;
	unsigned int i, j;

	struct fann_gng_neuron_private_data *priv;
	ann->gng_params->min_dist = FLT_MAX;
	priv=(struct fann_gng_neuron_private_data *) neuron->private_data;
 
	ann->gng_params->gng_closest_index = -1;
	ann->gng_params->gng_second_closest_index = -1;

	/* Find the winner node, i.e. the node with the least squared distance */
	for (i = 0; i < ann->gng_params->gng_num_cells; i++)
	{
	  runsum = 0;

	  for (j = 0; j < ann->num_input; j++) {
	    runsum += (priv->gng_cell_location[i][j] - neuron->inputs[j]) * (priv->gng_cell_location[i][j] - neuron->inputs[j]);
	 
	  }
	  if (runsum < ann->gng_params->min_dist)
	  {
	    sec_min_dist = ann->gng_params->min_dist;
	    ann->gng_params->min_dist = runsum;
	    ann->gng_params->gng_second_closest_index = ann->gng_params->gng_closest_index;
	    ann->gng_params->gng_closest_index = i;	
	  }
	  else if (runsum > ann->gng_params->min_dist && runsum < sec_min_dist)
	  {
	    sec_min_dist = runsum;
	    ann->gng_params->gng_second_closest_index = i;
	  }
	} 

	return;
}

FANN_EXTERNAL void FANN_API fann_gng_compute_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type *desired_output)
{
	/* Not used */
        return;
}

FANN_EXTERNAL void FANN_API fann_gng_neuron_backprop(struct fann *ann, struct fann_neuron *neuron, fann_type *prev_layer_errors)
{
	/* No backprop in GNG */
        return;
}

FANN_EXTERNAL void FANN_API fann_gng_neuron_update(struct fann *ann, struct fann_neuron *neuron)
{
	unsigned int i, j, num_node_neighbors;
	unsigned int numcells = ann->gng_params->gng_num_cells;	

	struct fann_gng_neuron_private_data *priv;
	int* node_neighbors;
	priv=(struct fann_gng_neuron_private_data *) neuron->private_data;
 
	/* Iterate iteration */
	ann->gng_params->gng_current_iteration++;

    /* Termination conditions. These are the only term conditions perscribed by the alg. 
	but you can add additional term. conditions as needed */
	if( numcells >= ann->gng_params->gng_max_nodes ) 
	  return;
       
	/*
	  Update the winner node's error
	*/
	priv->gng_cell_error[ann->gng_params->gng_closest_index] += ann->gng_params->min_dist;

	/*
	  Move the winner node. 
	*/ 			
	for (j = 0; j <ann->num_input; j++)
	{
	  priv->gng_cell_location[ann->gng_params->gng_closest_index][j] = priv->gng_cell_location[ann->gng_params->gng_closest_index][j] + 
	    ann->gng_params->gng_winner_node_scaling_factor* (neuron->inputs[j] - priv->gng_cell_location[ann->gng_params->gng_closest_index][j]);						 
	}		
	     
	/*
	  Get the neighbors of the winner node and move them.
	*/
	num_node_neighbors = fann_get_number_neighbors_gng(ann, ann->gng_params->gng_closest_index );
	if( (node_neighbors = (int*) calloc(num_node_neighbors,sizeof(int)) ) == NULL)
	{
	    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	    return;
	}
	fann_get_neighbors_gng(ann, ann->gng_params->gng_closest_index, node_neighbors );	

	for(i=0;i<num_node_neighbors;i++) 
	{
	  for (j = 0; j < ann->num_input; j++)
	  {
	    priv->gng_cell_location[node_neighbors[i]][j] = priv->gng_cell_location[node_neighbors[i]][j] + 
	      ann->gng_params->gng_neighbor_node_scaling_factor * (neuron->inputs[j] - priv->gng_cell_location[node_neighbors[i]][j]);
	  }
	}

	/*
	  Increment the edge age of all neighbors.
	*/
	fann_increment_edges_of_neighbors( ann, ann->gng_params->gng_closest_index);
	fann_safe_free( node_neighbors );

	/* If closest and second closest are connected by an edge set it to zero, 
	   If not,connect them with an edge. */
	ann->gng_params->gng_cell_edges[ann->gng_params->gng_closest_index][ann->gng_params->gng_second_closest_index]=0;
	ann->gng_params->gng_cell_edges[ann->gng_params->gng_second_closest_index][ann->gng_params->gng_closest_index]=0;

	/* Remove cell edges greater than age max */
	fann_remove_edges( ann );

	/* Remove cells with no edges */
	fann_remove_cell_with_no_edges( ann, neuron );

	/* 
	   If current iteration is multiple if lambda and max node count has not been meet,
	   then insert a new node.
	*/
	if((int)fmod((float)ann->gng_params->gng_current_iteration,(float)ann->gng_params->gng_iteration_of_node_insert) == 0) 
	   fann_gng_insert_node( ann, neuron );
	
	
	/* Decrease all error-variables of all nodes */
	for (i = 0; i < ann->gng_params->gng_num_cells; i++)
	{
	  priv->gng_cell_error[i] = priv->gng_cell_error[i] - (ann->gng_params->gng_global_error_reduction_factor * priv->gng_cell_error[i]);  
	}
	
	return;
}

/* Internal Functions used by GNG algorithm */
int fann_get_number_neighbors_gng(struct fann * ann, unsigned int node_of_interest) 
{
   unsigned int i;
   unsigned int num_neighbors=0;

   for(i = 0; i < ann->gng_params->gng_num_cells; i++) 
   {
     if(ann->gng_params->gng_cell_edges[node_of_interest][i] > -1)
     {
       num_neighbors++;
     }
   }
   return num_neighbors;
}

void fann_get_neighbors_gng(struct fann * ann, unsigned int node_of_interest, int* node_neighbors) 
{
   unsigned int i;
   int current_neighbor=0;
   for(i = 0; i < ann->gng_params->gng_num_cells; i++) 
   {
     if(ann->gng_params->gng_cell_edges[node_of_interest][i] > -1)
     {
       node_neighbors[ current_neighbor ] =i;
       current_neighbor++;
     }
   }
}

/* This code could be used in place of the current fann_get_neighbors_gng. It would be fast and more efficient, 
   but has not been tested yet.

 int* neighbor_vector;

  if((neighbor_vector = (int*)calloc())== NULL)

  for((i = 0; i < ann->gng_params->gng_num_cells; i++) 
  {
    if(cell_edges[node_of_interest][i] > -1)
    {
      num_neighbors++;
      neighbor_vector[] = i;                       
      cell_edges[node_of_interest][i]++;          
      cell_edges[i][node_of_interest]++;
      if(cell_edges[i][node_of_interest] > gng_max_age)
      {
	cell_edges[node_of_interest][i]=-1;        
	cell_edges[i][node_of_interest]=-1;
      }
    }
  }

*/


void fann_gng_insert_node(struct fann * ann, struct fann_neuron *neuron) 
{	
  unsigned int i,j,k;	
  double max_error = 0;
  int node_with_largest_error=0;
  int neighbor_with_largest_error=0;	
  struct fann_gng_neuron_private_data *priv;
  unsigned int num_node_neighbors;
  int* node_neighbors;	
  fann_type* temp_error_vector;
  fann_type** temp_location_matrix;
  int** temp_edge_matrix;
  priv=(struct fann_gng_neuron_private_data *) neuron->private_data;

  /* This time find the node with the LARGEST error */
  for (i = 0; i < ann->gng_params->gng_num_cells; i++)
  {
    if (priv->gng_cell_error[i] > max_error)
    {
      max_error = priv->gng_cell_error[i];
      node_with_largest_error = i;
    }
  }

  /* Now find the neighbor with the LARGEST error */
  num_node_neighbors = fann_get_number_neighbors_gng(ann, node_with_largest_error );

  if( (node_neighbors = (int*) calloc(num_node_neighbors,sizeof(int)) ) == NULL)
  {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }
 
  fann_get_neighbors_gng(ann, node_with_largest_error, node_neighbors );	
  max_error =0;
  for (i = 0; i < num_node_neighbors; i++)
  {
     if (priv->gng_cell_error[node_neighbors[i]] > max_error)
     {
       max_error = priv->gng_cell_error[node_neighbors[i]];
       neighbor_with_largest_error = node_neighbors[i];
     }
   }
 
   /* Insert a new node. First need to allocate new memory for a new error vector, location matrix and the edge
       matrix. 
   */
   ann->gng_params->gng_num_cells = priv->gng_num_cells = ann->gng_params->gng_num_cells + 1; 
 
   temp_error_vector = NULL;
   temp_location_matrix = NULL; 
   temp_edge_matrix = NULL;
    
   if( (temp_error_vector = (fann_type*) calloc(ann->gng_params->gng_num_cells,sizeof(fann_type)) ) == NULL)
   {
      fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
      return;
   }
    
    /* Allocate new memory to record the edges*/
   if( (temp_edge_matrix = (int**) calloc(ann->gng_params->gng_num_cells,sizeof(int*))) == NULL)
   {
      fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
      return;
   }
 
    for(i=0;i<ann->gng_params->gng_num_cells;i++) 
    {	
      if( (temp_edge_matrix[i] = (int*) calloc(ann->gng_params->gng_num_cells,sizeof(int))) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      }
    }

    /* Allocate new memory to record the cell locations */
    if( (temp_location_matrix = (fann_type**) calloc(ann->gng_params->gng_num_cells,sizeof(fann_type*))) == NULL)
    {
      fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
      return;
    } 

    for(i=0;i<ann->gng_params->gng_num_cells;i++) 
    {	
      if( (temp_location_matrix[i] = (fann_type*) calloc(ann->num_input,sizeof(fann_type))) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      }
    }
  
    /* Init the edge matrix*/
    for(j = 0; j < ann->gng_params->gng_num_cells; j++) 
      for(k=0;k<ann->gng_params->gng_num_cells;k++)
	temp_edge_matrix[j][k] = -1;
   
    /* Next we need to copy the contents from the old vector. The values for the 
       added cell are altered a couple of lines below. */ 
    for(j = 0; j < ann->gng_params->gng_num_cells-1; j++) 
    {
      temp_error_vector[j] = priv->gng_cell_error[j];
      for(k=0;k<ann->num_input;k++) 
	temp_location_matrix[j][k] = priv->gng_cell_location[j][k];

      for(k=0;k<ann->gng_params->gng_num_cells-1;k++)
	temp_edge_matrix[j][k] = ann->gng_params->gng_cell_edges[j][k];
    }

    /* Release the old memory */
    fann_safe_free(priv->gng_cell_error); 
    for(i=0;i<ann->gng_params->gng_num_cells-1;i++) 
    {
       fann_safe_free(ann->gng_params->gng_cell_edges[i]); 	
    }
    fann_safe_free(ann->gng_params->gng_cell_edges); 

    for(i=0;i<ann->gng_params->gng_num_cells-1;i++) 
      fann_safe_free(priv->gng_cell_location[i]); 	
   
    fann_safe_free(priv->gng_cell_location); 
   
    priv->gng_cell_error = temp_error_vector;
    priv->gng_cell_location = temp_location_matrix;   	
    ann->gng_params->gng_cell_edges = temp_edge_matrix;
 
    /* Change the location of the new node. Put it between 
       the node and neighbor with largest error. */
    for (j = 0; j < ann->num_input; j++)
    { 
       priv->gng_cell_location[ann->gng_params->gng_num_cells-1][j] = 
       (priv->gng_cell_location[ node_with_largest_error][j] + priv->gng_cell_location[neighbor_with_largest_error][j])/2;
    }
    ann->gng_params->gng_cell_edges[ann->gng_params->gng_num_cells-1][ann->gng_params->gng_num_cells-1] = -1;
    
    /* Create new edges */
    ann->gng_params->gng_cell_edges[ann->gng_params->gng_num_cells-1][node_with_largest_error] = 0;
    ann->gng_params->gng_cell_edges[node_with_largest_error][ann->gng_params->gng_num_cells-1] = 0;
    ann->gng_params->gng_cell_edges[ann->gng_params->gng_num_cells-1][neighbor_with_largest_error] = 0;
    ann->gng_params->gng_cell_edges[neighbor_with_largest_error][ann->gng_params->gng_num_cells-1] = 0;

    /* Remove old edges */
    ann->gng_params->gng_cell_edges[node_with_largest_error][neighbor_with_largest_error] = -1;
    ann->gng_params->gng_cell_edges[neighbor_with_largest_error][node_with_largest_error] = -1;

   /* Set error of new node, largest error node, and largest neighbor */
    priv->gng_cell_error[node_with_largest_error] = ann->gng_params->gng_local_error_reduction_factor * priv->gng_cell_error[node_with_largest_error];
    priv->gng_cell_error[neighbor_with_largest_error] = ann->gng_params->gng_local_error_reduction_factor * priv->gng_cell_error[neighbor_with_largest_error];
    priv->gng_cell_error[ann->gng_params->gng_num_cells-1] = priv->gng_cell_error[node_with_largest_error];
}

void fann_remove_edges(struct fann * ann)
{
  unsigned int i,j;
  for(i = 0; i < ann->gng_params->gng_num_cells; i++) 
  {
    for(j = 0; j < ann->gng_params->gng_num_cells; j++) 
    {
      if(ann->gng_params->gng_cell_edges[i][j] > (int)ann->gng_params->gng_max_age)
      {
	ann->gng_params->gng_cell_edges[i][j]=ann->gng_params->gng_cell_edges[j][i]=-1;
      }
    }
  }
}

void fann_remove_cell_with_no_edges(struct fann * ann, struct fann_neuron *neuron)
{  
  unsigned int i,j,k;
  struct fann_gng_neuron_private_data *priv; 
  fann_type* temp_error_vector;
  fann_type** temp_location_matrix;
  int** temp_edge_matrix;
  unsigned int hasEdge;
  unsigned int cell_to_delete;

  priv=(struct fann_gng_neuron_private_data *) neuron->private_data;

  for(i = 0; i < ann->gng_params->gng_num_cells; i++) 
  {
    hasEdge = 0;

    /* for a particular cell, check all of the edges */
    for(j = 0; j < ann->gng_params->gng_num_cells; j++) 
    {
      if(ann->gng_params->gng_cell_edges[i][j] > -1 ) {
	hasEdge = 1;
	break;
      }
    }

    if(hasEdge == 0 )
    {
      /* 
       Remove the cell if it has no edges. We first need to allocate memory for new vectors and the edge
       matrix. 
      */ 
      cell_to_delete =i;
      ann->gng_params->gng_num_cells = priv->gng_num_cells = ann->gng_params->gng_num_cells - 1; 
      temp_error_vector = NULL;
      temp_location_matrix = NULL;   
      temp_edge_matrix = NULL;
   
      if( (temp_error_vector = (fann_type*) calloc(ann->gng_params->gng_num_cells,sizeof(fann_type)) ) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      }

      /* Allocate new memory to record the cell locations */
      if( (temp_location_matrix = (fann_type**) calloc(ann->gng_params->gng_num_cells,sizeof(fann_type*))) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      } 

      for(i=0;i<ann->gng_params->gng_num_cells;i++) 
      {	
	if( (temp_location_matrix[i] = (fann_type*) calloc(ann->num_input,sizeof(fann_type))) == NULL)
        {
	  fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	  return;
	}
      }

      /* Allocate new memory to record the edges*/
      if( (temp_edge_matrix = (int**) calloc(ann->gng_params->gng_num_cells,sizeof(int*))) == NULL)
      {
	fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	return;
      }

      for(i=0;i<ann->gng_params->gng_num_cells;i++) 
      {	
	if( (temp_edge_matrix[i] = (int*) calloc(ann->gng_params->gng_num_cells,sizeof(int))) == NULL)
	{
	  fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
	  return;
	}
      }
 
      /* Next we need to copy the contents from the old vector, minus the deleted cell. */ 
      for(i = 0; i < ann->gng_params->gng_num_cells+1; i++) 
      {
	if(i < cell_to_delete )
	{
	  temp_error_vector[i] = priv->gng_cell_error[i];

	  for(k=0;k<ann->num_input;k++) 
	    temp_location_matrix[i][k] = priv->gng_cell_location[i][k];

	  for(k=0;k<ann->gng_params->gng_num_cells+1;k++)
	  {
	    if(k < cell_to_delete )
	      temp_edge_matrix[i][k] =  ann->gng_params->gng_cell_edges[i][k];
	    else if(k == cell_to_delete)
	      ;
	    else if(k > cell_to_delete) 
	      temp_edge_matrix[i][k-1] =  ann->gng_params->gng_cell_edges[i][k];
	  }
	}
	else if(i == cell_to_delete )
	  ;
	else if(i > cell_to_delete )
	{
	  temp_error_vector[i-1] = priv->gng_cell_error[i];	
	  for(k=0;k<ann->num_input;k++)
	    temp_location_matrix[i-1][k] = priv->gng_cell_location[i][k];
	
	  for(k=0;k<ann->gng_params->gng_num_cells+1;k++)
	  {
	    if(k < cell_to_delete )
	      temp_edge_matrix[i-1][k] =  ann->gng_params->gng_cell_edges[i][k];
	    else if(k == cell_to_delete)
	     ;
	    else if(k > cell_to_delete) 
	      temp_edge_matrix[i-1][k-1] =  ann->gng_params->gng_cell_edges[i][k];
	  }  
	}
      }
     
      /* Finally release the old memory and set the pointers to the new memory. */ 
      fann_safe_free(priv->gng_cell_error);  
      for(i=0;i<ann->gng_params->gng_num_cells+1;i++) 
      {
	fann_safe_free(priv->gng_cell_location[i]); 	
      }
      fann_safe_free(priv->gng_cell_location); 
      priv->gng_cell_error = temp_error_vector;
      priv->gng_cell_location = temp_location_matrix;
   
      for(i=0;i<ann->gng_params->gng_num_cells+1;i++) 
      {
	fann_safe_free(ann->gng_params->gng_cell_edges[i]); 	
      }
      fann_safe_free(ann->gng_params->gng_cell_edges); 	
      ann->gng_params->gng_cell_edges = temp_edge_matrix;
    }
  }
}

void fann_increment_edges_of_neighbors(struct fann * ann, unsigned int winner)
{
  unsigned int i;
  for(i = 0; i < ann->gng_params->gng_num_cells; i++) 
  {
      if(ann->gng_params->gng_cell_edges[i][winner] > -1)
      {
	ann->gng_params->gng_cell_edges[i][winner]++;
	ann->gng_params->gng_cell_edges[winner][i]++;
      }
  }
}

  /*
int* fann_get_neighbors_gng(struct fann * ann, unsigned int node_of_interest, unsigned int* num_neighbors) 
{

  int i;
  int* neighbor_vector;

  if((neighbor_vector = (int*)calloc())== NULL)

  for((i = 0; i < ann->gng_params->gng_num_cells; i++) 
  {
    if(cell_edges[node_of_interest][i] > -1)
    {
      num_neighbors++;
      neighbor_vector[] = i;                       
      cell_edges[node_of_interest][i]++;          
      cell_edges[i][node_of_interest]++;
      if(cell_edges[i][node_of_interest] > gng_max_age)
      {
	cell_edges[node_of_interest][i]=-1;        
	cell_edges[i][node_of_interest]=-1;
      }
    }
  }

 
}
*/

/* Calculate error rates over the entire training set */
FANN_EXTERNAL float FANN_API fann_get_MSE_gng(struct fann *ann, struct fann_train_data *data)
{
        unsigned int i;
	double cur_error;  
	struct fann_neuron *neuron;
	struct fann_gng_neuron_private_data *priv; 
	neuron = ann->first_layer->first_neuron;
	priv=(struct fann_gng_neuron_private_data *) neuron->private_data;

	cur_error=0;

	for (i = 0; i < data->num_data; i++) 
	{
                fann_run(ann, data->input[i]);
		cur_error += sqrt(priv->gng_cell_error[ann->gng_params->gng_closest_index]);
	}
	return cur_error / (float)(data->num_data);
}

#ifndef FIXEDFANN
FANN_EXTERNAL void FANN_API fann_train_example_gng(struct fann *ann, struct fann_train_data *data, unsigned int example_num, unsigned int max_examples)
{
        fann_train(ann, data->input[example_num % data->num_data], NULL);
}

/* INTERNAL FUNCTION  
   Train a som using the available data
*/
void fann_train_on_data_gng(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error) 
{
        unsigned int epoch_cnt = 0;
	float cur_error = FLT_MAX;
	unsigned int cur_data_index = 0;

	/* In the GNG, we treat each training example given to the GNG as an epoch */
	while ((epoch_cnt != max_epochs) && (cur_error > desired_error))
	{
                fann_train(ann, data->input[cur_data_index], NULL);
	
		cur_data_index++;
		epoch_cnt++;
		if (cur_data_index == data->num_data)
		{
	                cur_data_index = 0;
		}
 
		/* Generate reports after a certain number of training examples */
		if ((epoch_cnt % epochs_between_reports) == 0)
		{
		        cur_error = fann_get_MSE_gng(ann, data);
		        if (ann->training_params->callback == NULL) 
		        {
			        printf("Epochs     %8d. Current error: %.10f\n", epoch_cnt, cur_error);
			}
			else if(((*ann->training_params->callback)(ann, data, max_epochs, epochs_between_reports, desired_error, epoch_cnt)) == -1)
			{
		                break;
			}		
		}
	}
}


FANN_EXTERNAL void FANN_API fann_train_example_array_gng(struct fann *ann, fann_type *data, unsigned int example_num, unsigned int max_examples) {
	fann_train(ann, data, NULL);
}
#endif 

/* INTERNAL FUNCTION 
   Debug function to save the gng part of the network
*/
 void fann_save_gng_to_file(struct fann *ann, FILE *conf)
 {
         struct fann_neuron *current_neuron;
	 unsigned int i, j;
	 struct fann_gng_neuron_private_data *priv;
	 current_neuron = ann->first_layer->first_neuron;
	 priv = (struct fann_gng_neuron_private_data *)current_neuron->private_data;
   	 
	 /* Dump the gng parameters */
	 fprintf(conf, "%d,%d,%d,%f,%f,%f,%f\n", ann->gng_params->gng_max_nodes,ann->gng_params->gng_max_age,ann->gng_params->gng_iteration_of_node_insert,ann->gng_params->gng_local_error_reduction_factor,ann->gng_params->gng_global_error_reduction_factor,ann->gng_params->gng_winner_node_scaling_factor,ann->gng_params->gng_neighbor_node_scaling_factor);
	 
    
	 /* dump the weights */
	 for (i = 0; i < ann->gng_params->gng_num_cells; i++)
	 {
	   fprintf(conf,"%f ", (float)priv->gng_cell_error[i]);
	   for(j=0;j<ann->num_input;j++)
	     fprintf(conf,"%f ", (float)priv->gng_cell_location[i][j]);
	   
	   fprintf(conf,"\n ");
	 }
 }

FANN_EXTERNAL void FANN_API fann_set_gng_config(struct fann *ann, struct fann_gng_params *gng_params) {

  gng_params->gng_cell_edges = ann->gng_params->gng_cell_edges;
  gng_params->gng_num_cells = 2;
  gng_params->gng_current_iteration = ann->gng_params->gng_current_iteration;
  gng_params->gng_iteration_of_node_insert = ann->gng_params->gng_iteration_of_node_insert;
  *(ann->gng_params) = *gng_params;
}
