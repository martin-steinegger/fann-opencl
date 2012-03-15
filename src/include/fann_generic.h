#ifndef FANN_GENERIC
#define FANN_GENERIC
/* File: fann_generic.h
   The fann generic implementation is a backward compatible MIMO neuron
   implementation.
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
#define activation_function_name any
#define algorithm any
#define implementation connected

/* Function: MAKE_NAME(layer_destructor)
 * destroy the layer allocated data
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(layer_destructor)(struct fann_layer* layer);

/* Function: MAKE_NAME(layer_constructor)
   Allocates room inside the layer for neurons and connections. 
	 Adds one bias output set to 1.
 */
FANN_EXTERNAL int FANN_API MAKE_NAME(layer_constructor)(struct fann *ann, 
		struct fann_layer *layer, struct fann_layer_descr *descr);

/*Function: MAKE_NAME(layer_run)
  Iterates over all neurons and calls the respective run functions.
	The implementation is responsible for setting up the input array.
  If data is not already there this function must copy it into the 
	inpuit array before calling neuron_it->run().
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(layer_run)(struct fann *ann, struct fann_layer* layer);

/* Function: MAKE_NAME(layer_train_initialize)
   Allocates room inside the layer for training data structures.
	 This function is called just when it is necessary to allocate the space
	 for training. If a network is not going to be trainied (when using an already
	 trainied network) all this stuff is not allocated.
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(layer_train_initialize)(struct fann *ann, struct fann_layer *layer);


/* Function: MAKE_NAME(neuron_destructor)
 * destroy the neuron allocated data
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_destructor)(struct fann_neuron* neuron);

/* Function: MAKE_NAME(neuron_constructor)
   Allocates room inside the neuron for the connections.
   Creates a fully connected neuron.
 */
FANN_EXTERNAL int FANN_API MAKE_NAME(neuron_constructor)(struct fann *ann,
	 	struct fann_layer *layer, struct fann_neuron *neuron, struct fann_neuron_descr * descr);

/* Function: MAKE_NAME(neuron_run)
   Does a forward iteration on the neuron. 
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_run)(struct fann *ann, 
		struct fann_neuron *neuron);

/* Function: MAKE_NAME(neuron_train_initialize)
   Allocates room inside the neuron for training data structures.
	 This function is called just when it is necessary to allocate the space
	 for training. If a network is not going to be trainied (when using an already
	 trainied network) all this stuff is not allocated.
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_train_initialize)(struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron);

/* Function: MAKE_NAME(neuron_compute_MSE)
   Compute the error on a MIMO Neuron after forward propagation of 
    a certain input vector i.e. after neuron->run().
    The error is the sum, over all the outputs, of the squared difference
    between the computed output and the desired target
    also increments a counter because MSE is an average of such errors
   
    After calling this function on a neuron the train_errors array is set to:
    (desired_output - neuron_value)
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_compute_MSE)(struct fann *ann, 
		struct fann_neuron *neuron, fann_type *desired_output);

/* 
   Function: MAKE_NAME(neuron_backprop)
   Train the MIMO neuron: this function backpropagates the error to the 
	 previous layer and computes the weight update. The weight update is not 
	 applied here see <MAKE_NAME(neuron_update>)
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_backprop)(struct fann *ann,
	 	struct fann_neuron *neuron, fann_type *prev_layer_errors);

/* 
   Function: MAKE_NAME(neuron_update)
   Apply the training on the MIMO neuron: the weight update stored in the neuron
	 is applied. This function is called once per input pattern when online training
	 is carried on and once per epoch when batch algoritms are selected
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_update)(struct fann *ann,
	 	struct fann_neuron *neuron);

/* Struct: MAKE_NAME(neuron_private_data)
   This structure just stores the arrays needed to use all the algorims 
	 supported in old fann code.
 */
struct MAKE_NAME(neuron_private_data)
{
	fann_type * prev_steps;
	fann_type * prev_weights_deltas;
};

FANN_EXTERNAL int FANN_API MAKE_NAME(neuron_load) (struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron, FILE *conf);
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_save) (struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron, FILE *conf);
FANN_EXTERNAL int FANN_API MAKE_NAME(layer_load) (struct fann *ann, struct fann_layer *layer, FILE *conf);
FANN_EXTERNAL void FANN_API MAKE_NAME(layer_save) (struct fann *ann, struct fann_layer *layer, FILE *conf);


/*#define fann_activation_any(neuron, out ) \*/
static __inline void fann_activation_any(struct fann_neuron *neuron, unsigned int out ) 
{
	switch(neuron->activation_function) 
	{ 
		case FANN_LINEAR: 
			neuron->outputs[out] = (fann_type)neuron->sums[out]; 
			break; 
		case FANN_LINEAR_PIECE: 
			neuron->outputs[out] = (fann_type)((neuron->sums[out] < 0) ? 0 : (neuron->sums[out] > 1) ? 1 : neuron->sums[out]); 
			break; 
		case FANN_LINEAR_PIECE_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)((neuron->sums[out] < -1) ? -1 : (neuron->sums[out] > 1) ? 1 : neuron->sums[out]); 
			break; 
		case FANN_SIGMOID: 
			neuron->outputs[out] = (fann_type)fann_sigmoid_real(neuron->sums[out]); 
			break; 
		case FANN_SIGMOID_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)fann_sigmoid_symmetric_real(neuron->sums[out]); 
			break; 
		case FANN_SIGMOID_SYMMETRIC_STEPWISE: 
			neuron->outputs[out] = (fann_type)fann_stepwise(-2.64665293693542480469e+00, -1.47221934795379638672e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, -9.90000009536743164062e-01, -8.99999976158142089844e-01, -5.00000000000000000000e-01, 5.00000000000000000000e-01, 8.99999976158142089844e-01, 9.90000009536743164062e-01, -1, 1, neuron->sums[out]); 
			break; 
		case FANN_SIGMOID_STEPWISE: 
			neuron->outputs[out] = (fann_type)fann_stepwise(-2.64665246009826660156e+00, -1.47221946716308593750e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, 4.99999988824129104614e-03, 5.00000007450580596924e-02, 2.50000000000000000000e-01, 7.50000000000000000000e-01, 9.49999988079071044922e-01, 9.95000004768371582031e-01, 0, 1, neuron->sums[out]); 
			break; 
		case FANN_THRESHOLD: 
			neuron->outputs[out] = (fann_type)((neuron->sums[out] < 0) ? 0 : 1); 
			break; 
		case FANN_THRESHOLD_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)((neuron->sums[out] < 0) ? -1 : 1); 
			break; 
		case FANN_GAUSSIAN: 
			neuron->outputs[out] = (fann_type)fann_gaussian_real(neuron->sums[out]); 
			break; 
		case FANN_GAUSSIAN_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)fann_gaussian_symmetric_real(neuron->sums[out]); 
			break; 
		case FANN_ELLIOT: 
			neuron->outputs[out] = (fann_type)fann_elliot_real(neuron->sums[out]); 
			break; 
		case FANN_ELLIOT_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)fann_elliot_symmetric_real(neuron->sums[out]); 
			break; 
		case FANN_SIN_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)fann_sin_symmetric_real(neuron->sums[out]); 
			break; 
		case FANN_COS_SYMMETRIC: 
			neuron->outputs[out] = (fann_type)fann_cos_symmetric_real(neuron->sums[out]); 
			break; 
		case FANN_SIN: 
			neuron->outputs[out] = (fann_type)fann_sin_real(neuron->sums[out]); 
			break; 
		case FANN_COS: 
			neuron->outputs[out] = (fann_type)fann_cos_real(neuron->sums[out]); 
			break; 
		case FANN_GAUSSIAN_STEPWISE: 
			neuron->outputs[out] = 0; 
			break; 
	}
}

#define fann_activation_derived_any(neuron,out,res)\
{\
	fann_type value=neuron->outputs[out];\
	switch (neuron->activation_function)\
	{\
		case FANN_LINEAR:\
		case FANN_LINEAR_PIECE:\
		case FANN_LINEAR_PIECE_SYMMETRIC:\
			res = fann_linear_derive(neuron->activation_steepness, value);\
			break; \
		case FANN_SIGMOID:\
		case FANN_SIGMOID_STEPWISE:\
			value = fann_clip(value, 0.01f, 0.99f);\
			res = fann_sigmoid_derive(neuron->activation_steepness, value);\
			break; \
		case FANN_SIGMOID_SYMMETRIC:\
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:\
			value = fann_clip(value, -0.98f, 0.98f);\
			res = fann_sigmoid_symmetric_derive(neuron->activation_steepness, value);\
			break; \
		case FANN_GAUSSIAN:\
			/* value = fann_clip(value, 0.01f, 0.99f); */\
			res= fann_gaussian_derive(neuron->activation_steepness, value, neuron->sums[out]);\
			break; \
		case FANN_GAUSSIAN_SYMMETRIC:\
			/* value = fann_clip(value, -0.98f, 0.98f); */\
			res = fann_gaussian_symmetric_derive(neuron->activation_steepness, value, neuron->sums[out]);\
			break; \
		case FANN_ELLIOT:\
			value = fann_clip(value, 0.01f, 0.99f);\
			res = fann_elliot_derive(neuron->activation_steepness, value, neuron->sums[out]);\
			break; \
		case FANN_ELLIOT_SYMMETRIC:\
			value = fann_clip(value, -0.98f, 0.98f);\
			res = fann_elliot_symmetric_derive(neuron->activation_steepness, value, neuron->sums[out]);\
			break; \
		case FANN_SIN_SYMMETRIC:\
			res = fann_sin_symmetric_derive(neuron->activation_steepness, neuron->sums[out]);\
			break; \
		case FANN_COS_SYMMETRIC:\
			res = fann_cos_symmetric_derive(neuron->activation_steepness, neuron->sums[out]);\
			break; \
		case FANN_SIN:\
			res = fann_sin_derive(neuron->activation_steepness, neuron->sums[out]);\
			break; \
		case FANN_COS:\
			res = fann_cos_derive(neuron->activation_steepness, neuron->sums[out]);\
			break; \
		case FANN_THRESHOLD_SYMMETRIC:\
		case FANN_THRESHOLD:\
			fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);\
			break; \
		case FANN_GAUSSIAN_STEPWISE: /*FIXME*/\
			break; \
	}\
}

#undef activation_function_name
#undef algorithm
#undef implementation

#endif

/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap
 */
