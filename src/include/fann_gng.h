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

/* This file contains manipulations for Self-Organizing Maps */

#ifndef __fann_gng_h__
#define __fann_gng_h__

#include <stdio.h>

#define PI 3.14159265


FANN_EXTERNAL void FANN_API fann_destroy_gng(struct fann *ann);
FANN_EXTERNAL struct fann* FANN_API fann_copy_gng(const struct fann* orig);
void fann_save_gng_to_file(struct fann *ann, FILE *conf);
fann_type *fann_run_gng(struct fann *ann, fann_type *input);
void fann_update_weights_gng(struct fann *ann);


FANN_EXTERNAL int FANN_API fann_gng_layer_constructor(struct fann *ann, struct fann_layer *layer, struct fann_layer_descr *descr);
FANN_EXTERNAL void FANN_API fann_gng_layer_destructor(struct fann_layer* layer);
FANN_EXTERNAL void FANN_API fann_gng_layer_train_initialize(struct fann *ann, struct fann_layer *layer);
FANN_EXTERNAL void FANN_API fann_gng_layer_run(struct fann *ann, struct fann_layer* layer);

FANN_EXTERNAL int FANN_API fann_gng_neuron_constructor(struct fann *ann, struct fann_layer *layer, 
						       struct fann_neuron *neuron, struct fann_neuron_descr * descr);
FANN_EXTERNAL void FANN_API fann_gng_neuron_destructor(struct fann_neuron* neuron);
FANN_EXTERNAL void FANN_API fann_gng_compute_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type *desired_output);
FANN_EXTERNAL void FANN_API fann_gng_neuron_backprop(struct fann *ann, struct fann_neuron *neuron, fann_type *prev_layer_errors);
FANN_EXTERNAL void FANN_API fann_gng_neuron_update(struct fann *ann, struct fann_neuron *neuron);
FANN_EXTERNAL void FANN_API fann_gng_neuron_run(struct fann * ann, struct fann_neuron *neuron);

FANN_EXTERNAL float FANN_API fann_get_MSE_gng(struct fann *ann, struct fann_train_data *data);


FANN_EXTERNAL void FANN_API fann_randomize_weights_gng(struct fann *ann, fann_type min_weight, fann_type max_weight);
FANN_EXTERNAL void FANN_API fann_init_weights_gng(struct fann *ann, struct fann_train_data *train_data);
FANN_EXTERNAL void FANN_API fann_print_connections_gng(struct fann *ann);
FANN_EXTERNAL void FANN_API fann_print_parameters_gng(struct fann *ann);
FANN_EXTERNAL unsigned int FANN_API fann_get_num_input_gng(struct fann *ann);
FANN_EXTERNAL unsigned int FANN_API fann_get_num_output_gng(struct fann *ann);
FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons_gng(struct fann *ann);
FANN_EXTERNAL unsigned int FANN_API fann_get_total_connections_gng(struct fann *ann);
FANN_EXTERNAL enum fann_nettype_enum FANN_API fann_get_network_type_gng(struct fann *ann);

FANN_EXTERNAL void FANN_API fann_gng_decay(struct fann *ann, int epoch, int max_epoch);
FANN_EXTERNAL void FANN_API fann_dump_weights_gng(struct fann *ann);
FANN_EXTERNAL void FANN_API fann_dump_weights_file_gng(struct fann *ann);

void fann_train_on_data_gng(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error);
FANN_EXTERNAL void FANN_API fann_train_example_gng(struct fann *ann, struct fann_train_data *data, unsigned int example_num, unsigned int max_examples);
FANN_EXTERNAL void FANN_API fann_train_example_array_gng(struct fann *ann, fann_type *data, unsigned int example_num, unsigned int max_examples);

void fann_gng_insert_node( struct fann * ann, struct fann_neuron *neuron);
void fann_remove_edges(struct fann * ann);
void fann_increment_edges_of_neighbors(struct fann *ann,unsigned int winner);
void fann_get_neighbors_gng(struct fann *ann, unsigned int node_of_interest,int* neighbors);
int fann_get_number_neighbors_gng(struct fann *ann, unsigned int node_of_interest);
void fann_remove_cell_with_no_edges(struct fann * ann, struct fann_neuron *neuron);

FANN_EXTERNAL void FANN_API fann_set_gng_config(struct fann *ann, struct fann_gng_params *gng_params);


struct fann_gng_neuron_private_data
{
        /* Used for gng neurons */
        fann_type *gng_cell_error;             /* Error values for the individual cells */
        fann_type **gng_cell_location;         /* The positional values for the individual cells */  
        unsigned int gng_num_cells;            /* The number of cells in the GNG */
};

#endif
