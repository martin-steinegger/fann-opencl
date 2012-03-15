#include "fann.h"
#define activation_function_name any
#define algorithm any
#define implementation connected

static __inline void fann_update_MSE_any(struct fann *ann, struct fann_neuron* neuron, fann_type *neuron_diff);

#include "fann_base.c"

/*
 * LAYER STUFF
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(layer_destructor)(struct fann_layer* layer)
{
	fann_base_layer_destructor(layer);
}

FANN_EXTERNAL int FANN_API MAKE_NAME(layer_constructor)(struct fann *ann, 
		struct fann_layer *layer, struct fann_layer_descr *descr)
{
	if (fann_base_layer_constructor(ann, layer, descr))
		return 1;
	/* Set the other function pointers */
	layer->destructor = MAKE_NAME(layer_destructor);
	layer->initialize_train_errors = MAKE_NAME(layer_train_initialize);
	layer->run = MAKE_NAME(layer_run);
	layer->load = MAKE_NAME(layer_load);
	layer->save = MAKE_NAME(layer_save);
	layer->type = strdup(MAKE_TYPE());
	return 0;
}

FANN_EXTERNAL void FANN_API MAKE_NAME(layer_run)(struct fann *ann, struct fann_layer* layer)
{
	fann_base_layer_run(ann,layer);
}

FANN_EXTERNAL void FANN_API MAKE_NAME(layer_train_initialize)(struct fann *ann, struct fann_layer *layer)
{	
	fann_base_layer_train_initialize(ann,layer);
}

/*
 * NEURON STUFF
 */
FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_destructor)(struct fann_neuron* neuron)
{
	struct MAKE_NAME(neuron_private_data ) *priv = (struct MAKE_NAME(neuron_private_data) *) neuron->private_data;
	fann_safe_free(priv->prev_weights_deltas);
	fann_safe_free(priv->prev_steps);
	fann_safe_free(neuron->private_data);
	fann_base_neuron_destructor(neuron);
}

FANN_EXTERNAL int FANN_API MAKE_NAME(neuron_constructor)(struct fann *ann, struct fann_layer *layer, 
		struct fann_neuron *neuron, struct fann_neuron_descr * descr)
{
	struct MAKE_NAME(neuron_private_data) *priv;

	if (fann_base_neuron_constructor(ann, layer, neuron, descr))
		return 1;

	/*allocate neuron private data*/
	if (( neuron->private_data = priv =
				(struct MAKE_NAME(neuron_private_data) * ) malloc (sizeof(struct MAKE_NAME(neuron_private_data))) ) == NULL)
		return 1;

	priv->prev_steps=NULL;
	priv->prev_weights_deltas=NULL;

	/* set the function pointers */
	neuron->destructor = MAKE_NAME(neuron_destructor);
	neuron->run = MAKE_NAME(neuron_run);
	neuron->backpropagate = MAKE_NAME(neuron_backprop);
	neuron->update_weights = MAKE_NAME(neuron_update);
	neuron->compute_error = MAKE_NAME(neuron_compute_MSE);
	neuron->train_initialize = MAKE_NAME(neuron_train_initialize);
	neuron->load = MAKE_NAME(neuron_load);
	neuron->save = MAKE_NAME(neuron_save);
	neuron->type = strdup(MAKE_TYPE());
	return 0;
}


FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_train_initialize)(struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron)
{
	unsigned int i;
	struct MAKE_NAME(neuron_private_data) * priv = (struct MAKE_NAME(neuron_private_data) *) neuron->private_data;
	fann_base_neuron_train_initialize(ann, layer, neuron);

	/* if no room allocated for the variabels, allocate it now */
	if(priv->prev_steps == NULL)
	{
		priv->prev_steps = (fann_type *) malloc(neuron->num_weights * sizeof(fann_type));
		if(priv->prev_steps == NULL)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}

	if(ann->training_params->training_algorithm == FANN_TRAIN_RPROP)
	{
		for(i = 0; i < neuron->num_weights; i++)
			priv->prev_steps[i] = ann->rprop_params->rprop_delta_zero;
	}
	else
	{
		memset(priv->prev_steps, 0, neuron->num_weights * sizeof(fann_type));
	}

	/* if no room allocated for the variabels, allocate it now */
	if(priv->prev_weights_deltas == NULL)
	{
		priv->prev_weights_deltas =
			(fann_type *) calloc(neuron->num_weights, sizeof(fann_type));
		if(priv->prev_weights_deltas == NULL)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	else
	{
		memset(priv->prev_weights_deltas, 0, (neuron->num_weights) * sizeof(fann_type));
	}
}

FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_run)(struct fann * ann, struct fann_neuron * neuron)
{
	fann_base_neuron_run(ann, neuron);
}

FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_compute_MSE)(struct fann *ann, struct fann_neuron *neuron, fann_type *desired_output)
{
	fann_base_neuron_compute_MSE(ann,neuron,desired_output);
}

FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_backprop)(struct fann *ann, struct fann_neuron *neuron, fann_type *prev_layer_errors)
{
	fann_base_neuron_backprop(ann,neuron,prev_layer_errors);
}

void MAKE_NAME(neuron_standard_backprop_update)(struct fann *ann, struct fann_neuron *neuron)
{
	unsigned int o, j;
	fann_type *weights, *deltas;
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

	for (o = 0; o < num_outputs; o++)
	{
		for (j = 0; j < num_inputs; j++)
		{
			/* adjust the weight */
			weights[j] += deltas[j] * learning_rate; /* FIXME add the learning momentum here */
			deltas[j]=0;
		}
		weights += num_inputs;
		deltas += num_inputs;
	}
	neuron->num_backprop_done=0;
}

/* INTERNAL FUNCTION
	 The iRprop- algorithm
	 */
void MAKE_NAME(neuron_irpropm_update)(struct fann *ann, struct fann_neuron *neuron)
{
	struct MAKE_NAME(neuron_private_data) *priv = (struct MAKE_NAME(neuron_private_data) *) neuron->private_data;

	fann_type *weights = neuron->weights;
	fann_type *weights_deltas = neuron->weights_deltas;
	fann_type *prev_weights_deltas = priv->prev_weights_deltas;
	fann_type *prev_steps = priv->prev_steps;

	const unsigned int num_outputs = neuron->num_outputs;
	const unsigned int num_inputs = neuron->num_inputs;
	float increase_factor = ann->rprop_params->rprop_increase_factor;	/*1.2; */
	float decrease_factor = ann->rprop_params->rprop_decrease_factor;	/*0.5; */
	float delta_min = ann->rprop_params->rprop_delta_min;	/*0.0; */
	float delta_max = ann->rprop_params->rprop_delta_max;	/*50.0; */

	unsigned int o;
	
	if (neuron->num_backprop_done==0)
	{
		fann_error(NULL, FANN_E_CANT_USE_TRAIN_ALG);
		return;
	}

	for (o = 0; o < num_outputs; o++)
	{
		int i;
        for (i = 0; i < num_inputs; i++)
		{
			fann_type next_step;
            fann_type prev_step = fann_max(prev_steps[i], (fann_type) 0.000001);	/* prev_step may not be zero because then the training will stop */
			/* does 0.0001 make sense????*/
			fann_type delta = weights_deltas[i];
			fann_type prev_delta = prev_weights_deltas[i];

			fann_type same_sign = prev_delta * delta;
            
			if (delta*delta == 0.0) /* Prevent Arithmetic underflow */
                next_step = 0.0;
            else if(same_sign >= 0.0)
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
	}
	neuron->num_backprop_done=0;
}

FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_update)(struct fann *ann, struct fann_neuron *neuron)
{
    /* This function does nothing, why was it called?
     fann_base_neuron_update(ann,neuron);
    */
		switch (ann->training_params->training_algorithm)
	{
		case FANN_TRAIN_INCREMENTAL:
		case FANN_TRAIN_SARPROP:
		case FANN_TRAIN_QUICKPROP:
		case FANN_TRAIN_BATCH:
			MAKE_NAME(neuron_standard_backprop_update)(ann, neuron);
			return;
		case FANN_TRAIN_RPROP:
			MAKE_NAME(neuron_irpropm_update)(ann, neuron);
			return;
		default:
			return;
	}
}

/* INTERNAL FUNCTION
   Helper function to update the MSE value and return a diff which takes symmetric functions into account
i*/
static __inline void fann_update_MSE_any(struct fann *ann, struct fann_neuron* neuron, fann_type *neuron_diff)
{
	float neuron_diff2;
	
	switch (neuron->activation_function)
	{
		case FANN_LINEAR_PIECE_SYMMETRIC:
		case FANN_THRESHOLD_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
		case FANN_ELLIOT_SYMMETRIC:
		case FANN_GAUSSIAN_SYMMETRIC:
		case FANN_SIN_SYMMETRIC:
		case FANN_COS_SYMMETRIC:
			*neuron_diff /= (fann_type)2.0;
			break;
		case FANN_THRESHOLD:
		case FANN_LINEAR:
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
		case FANN_GAUSSIAN:
		case FANN_GAUSSIAN_STEPWISE:
		case FANN_ELLIOT:
		case FANN_LINEAR_PIECE:
		case FANN_SIN:
		case FANN_COS:
			break;
	}

#ifdef FIXEDFANN
		neuron_diff2 =
			(*neuron_diff / (float) ann->fixed_params->multiplier) * (*neuron_diff / (float) ann->fixed_params->multiplier);
#else
		neuron_diff2 = (float) (*neuron_diff * *neuron_diff);
#endif

	ann->training_params->MSE_value += neuron_diff2;

	/*printf("neuron_diff %f = (%f - %f)[/2], neuron_diff2=%f, sum=%f, MSE_value=%f, num_MSE=%d\n", neuron_diff, *desired_output, neuron_value, neuron_diff2, last_layer_begin->sum, ann->MSE_value, ann->num_MSE); */
	
	if(fann_abs(*neuron_diff) >= ann->training_params->bit_fail_limit)
	{
		ann->training_params->num_bit_fail++;
	}
}
#if DEBUG && 0
#define fscanf_guard(cmd) \
{ \
	long pos=ftell(conf);\
	long size=0;\
	char *s;\
	int ret=(cmd);\
	if(ret == EOF || ret==0) \
	{\
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, #cmd ); \
		return 1;\
	}\
	size=ftell(conf)-pos;\
	fseek(conf,pos, SEEK_SET);\
	s=(char*)malloc(size+1); \
	fread(s, 1, size, conf); \
	s[size]='\0';\
	printf("pos=%lu, size=%lu, now=%lu\n", pos, size, pos+size );\
	printf("cmd was:%s\n", #cmd );\
	printf("I found the following:\n%s\n", s);\
	free(s);\
}
#else
#define fscanf_guard(cmd) \
{ \
	int ret=(cmd);\
	if(ret == EOF || ret==0) \
	{\
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, #cmd ); \
		return 1;\
	}\
}
#endif

FANN_EXTERNAL int FANN_API MAKE_NAME(neuron_load) (struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron, FILE *conf)
{
	unsigned int i;
	fscanf_guard(fscanf(conf, "\t\tNeuron Data\n\t\t\tactivation_function=%u\n\t\t\tactivation_steepness=" FANNSCANF "\n", 
			&neuron->activation_function, &neuron->activation_steepness));

	/* save the connection "(source weight) " */
	fscanf(conf, "\t\t\tweights="); 
	for (i = 0; i<neuron->num_weights; i++)
		fscanf_guard(fscanf(conf, FANNSCANF " ", &(neuron->weights[i])));

	fscanf(conf, "\n");

	return 0;
}

FANN_EXTERNAL void FANN_API MAKE_NAME(neuron_save) (struct fann *ann, struct fann_layer *layer, struct fann_neuron *neuron, FILE *conf)
{
	unsigned int i; /*FIXME: FIX_NETS won't work!*/
#ifndef FIXEDFANN
	unsigned int save_as_fixed=0, fixed_multiplier=0; /*FIXME: FIX_NETS won't work!*/
#endif
	fprintf(conf, "\t\tNeuron Data\n");
#ifndef FIXEDFANN
	if(save_as_fixed)
	{
		fprintf(conf, "\t\t\tactivation_function=%u\n\t\t\tactivation_steepness=%u\n", 
				neuron->activation_function,
				(int) floor((neuron->activation_steepness * fixed_multiplier) + 0.5));

		fprintf(conf, "\t\t\tweights="); 
		for (i = 0; i<neuron->num_weights; i++)
		{
			/* save the connection "(source weight) " */
			fprintf(conf, "%d ",
					(int) floor((neuron->weights[i] * fixed_multiplier) + 0.5));
		}
		fprintf(conf, "\n"); 
	}
	else
	{
		fprintf(conf, "\t\t\tactivation_function=%u\n\t\t\tactivation_steepness=" FANNPRINTF "\n", 
				neuron->activation_function, neuron->activation_steepness);

		/* save the connection "(source weight) " */
		fprintf(conf, "\t\t\tweights="); 
		for (i = 0; i<neuron->num_weights; i++)
		{
			fprintf(conf, FANNPRINTF " ", neuron->weights[i]);
		}
		fprintf(conf, "\n"); 
	}
#else
	fprintf(conf, "\t\t\tactivation_function=%u\n\t\t\tactivation_steepness=%u\n", 
			neuron->activation_function, neuron->activation_steepness);

	/* save the connection "(source weight) " */
	fprintf(conf, "\t\t\tweights="); 
	for (i = 0; i<neuron->num_weights; i++)
	{
		fprintf(conf, FANNPRINTF " ", neuron->weights[i]);
	}
	fprintf(conf, "\n"); 
#endif
}

FANN_EXTERNAL int FANN_API MAKE_NAME(layer_load) (struct fann *ann, struct fann_layer *layer, FILE *conf)
{
	struct fann_neuron *neuron_it=layer->first_neuron;
	fscanf(conf, "\tLayer data:\n");
	for (;neuron_it<layer->last_neuron; neuron_it++)
		if (neuron_it->load(ann, layer, neuron_it, conf))
			return 1;
	return 0;
}

FANN_EXTERNAL void FANN_API MAKE_NAME(layer_save) (struct fann *ann, struct fann_layer *layer, FILE *conf)
{
	struct fann_neuron *neuron_it=layer->first_neuron;
	fprintf(conf, "\tLayer data:\n");
	for (;neuron_it<layer->last_neuron; neuron_it++)
		neuron_it->save(ann, layer, neuron_it, conf);
}

#undef activation_function_name
#undef algorithm
#undef implementation

/*
 * vim: ts=2 smarttab smartindent shiftwidth=2 nowrap noet
 */
