#ifndef FANN_SPARSE
#define FANN_SPARSE
/* File: fann_sparse.h
   The fann sparse implementation is a backward compatible MIMO neuron
   implementation for sparse networks.
	 Using the MIMO Neuron defined by theese functions one gets a MIMO neuron that
	 behaves like an old FANN Neuron i.e. it gets its working parameters from the 
	 struct fann.
	 This MIMO implementation is a kind of compatibilty layer to allow old FANN 
	 using applications to run unmodified.
	 This code is not fast: the new VFANN implementations will be much smarter
	 (specilized) and faster.
	 The code in here uses swithches everywhere there is need to select something
	 like activation functions and algorithms.
 */


/* Function: fann_sparse_layer_constructor
   Allocates room inside the layer for neurons and connections. 
	 Adds one bias output set to 1.
 */
#define fann_sparse_layer_constructor fann_layer_constructor_connected_any_any

/* Function: fann_sparse_neuron_constructor
   Allocates room inside the neuron for the connections.
   Creates a random sparsely connected neuron acconrding to the 
	 connection rate specified in descr->private_data as a float value.
 */
FANN_EXTERNAL int FANN_API fann_sparse_neuron_constructor(struct fann *ann,
	 	struct fann_layer *layer, struct fann_neuron *neuron, struct fann_neuron_descr * descr);

/* Function: fann_sparse_neuron_run
   Does a forward iteration on the neuron. 
   We can use the generic version code because we keep the 
	 masked weights to 0 
*/
#define fann_sparse_neuron_run fann_neuron_run_connected_any_any

/* Function: fann_sparse_compute_MSE
   Compute the error on a MIMO Neuron after forward propagation of 
    a certain input vector i.e. after neuron->run().
    The error is the sum, over all the outputs, of the squared difference
    between the computed output and the desired target
    also increments a counter because MSE is an average of such errors
   
    After calling this function on a neuron the train_errors array is set to:
    (desired_output - neuron_value)
		Compute the error at the MIMO neuron output.
 		We can use the generic version code because we keep the 
 		masked weights to 0 
*/
#define fann_sparse_neuron_compute_MSE fann_neuron_compute_MSE_connected_any_any 

/*
   Function: fann_sparse_neuron_backprop
   Train the MIMO neuron: this function backpropagates the error to the 
	 previous layer and computes the weight update. The weight update is not 
	 applied here see <fann_neuron_update_connected_any_any>
   We can use the generic version code because we keep the 
   masked weights to 0 
*/
#define fann_sparse_neuron_backprop fann_neuron_backprop_connected_any_any

/* 
   Function: fann_sparse_neuron_update
   Apply the training on the MIMO neuron according to the globally selected algorithm:
	 the weight update stored in the neuron is applied. 
	 This function is called once per input pattern when online training
	 is carried on and once per epoch when batch algoritms are selected.
*/
FANN_EXTERNAL void FANN_API fann_sparse_neuron_update(struct fann *ann,
	 	struct fann_neuron *neuron);

/* Struct: fann_sparse_neuron_private_data
   The structure where the connection mask is tored.
   This structure also stores the arrays needed to use all the algorims 
	 supported in old fann code (inside the fann_neuron_private_data_connected_any_any).
 */
struct fann_sparse_neuron_private_data
{
	struct fann_neuron_private_data_connected_any_any *generic;
	fann_type *mask;
};

#endif

/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap noet
 */
