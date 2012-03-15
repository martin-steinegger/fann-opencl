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

#ifndef _FANN_RECURRENT_H_
#define _FANN_RECURRENT_H_

#include "fann.h"






/**************************************************
 REAL-TIME RECURRENT LEARNING

 Williams and Zipser, "A Learning Algorithm for
   Continually Running Fully Recurrent Neural
   Networks," Neural Computation, 1. (1989)

 NOTE: This function is still being debugged.
       MSE does not decrease properly.
 *************************************************/
FANN_EXTERNAL void FANN_API fann_train_rtrl(struct fann *ann, struct fann_train_data *pattern, 
											float max_MSE, unsigned int max_iters, float rate);


/**************************************************
 HOPFIELD NETWORK

 Fully recurrent N neuron network which is
 used for content-addressable memories. Uses
 one-shot learning for input patterns of length
 N. Then, given an input vector of length N,
 it will recall one of the initial patterns
 (or other fixed points such as their inverses).
 *************************************************/

FANN_EXTERNAL fann_type *FANN_API fann_run_hopfield(struct fann *ann, fann_type *input);

FANN_EXTERNAL void FANN_API fann_train_hopfield(struct fann *ann, struct fann_train_data *pattern);

/* A Hopfield network will be implemented as
   a fully recurrent network with 0 inputs and 'num_neurons'
   outputs! */
FANN_EXTERNAL struct fann *FANN_API fann_create_hopfield(
	unsigned int num_neurons);

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
	unsigned int num_outputs);


/* Only computes the MSE for the last 'num_output'
     neurons, unlike the standard which computes
	 it for all of them. 'desired_output' must
	 be of length 'num_output'. */
FANN_EXTERNAL void FANN_API fann_compute_MSE_fully_recurrent(
	struct fann *ann, fann_type *desired_output);


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
FANN_EXTERNAL struct fann *FANN_API fann_create_unrolled_recurrent(unsigned int num_weights, fann_type *weights, unsigned int time_steps);



/* Displays the connections between the input layer
     and the internal neurons of a fully recurrent
	 network. Specifically, it prints the number
	 of connections correctly and separately specifies
	 the inputs and outputs. */
FANN_EXTERNAL void FANN_API fann_print_connections_fully_recurrent(struct fann *ann);


FANN_EXTERNAL void FANN_API fann_neuron_destructor_fully_recurrent(struct fann_neuron* neuron);
FANN_EXTERNAL int FANN_API fann_neuron_constructor_fully_recurrent(struct fann *ann, struct fann_layer *layer, 
		struct fann_neuron *neuron, struct fann_neuron_descr * descr);
FANN_EXTERNAL void FANN_API fann_layer_destructor_fully_recurrent(struct fann_layer* layer);
FANN_EXTERNAL int FANN_API fann_layer_constructor_fully_recurrent(struct fann *ann, 
		struct fann_layer *layer, struct fann_layer_descr *descr);
FANN_EXTERNAL void FANN_API fann_layer_run_fully_recurrent(
	struct fann *ann, struct fann_layer* layer);
FANN_EXTERNAL void FANN_API fann_neuron_train_initialize_fully_recurrent(
	struct fann *ann, 
	struct fann_layer *layer, 
	struct fann_neuron *neuron);
FANN_EXTERNAL void FANN_API fann_layer_train_initialize_fully_recurrent(struct fann *ann, struct fann_layer *layer);
FANN_EXTERNAL void FANN_API fann_neuron_run_fully_recurrent(struct fann *ann, struct fann_neuron *neuron);
FANN_EXTERNAL void FANN_API fann_fully_recurrent_neuron_compute_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type *desired_output);

#endif
