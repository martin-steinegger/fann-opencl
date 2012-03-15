#ifdef __cplusplus
extern "C" {
#endif
#ifndef _fann_optimized_opencl_h_
#define _fann_optimized_opencl_h_

/* Function: fann_run_many
 Will run many inputs through the neural network, returning many arrays of outputs. The structure of the input value
 arrays is N sets of M input variables, where N equals num_runs and M equals the number of input neurons. The output
 array is also organized as N sets of J outputs, where J equals the number of output neurons.
 
 See also:
 <fann_test>
 
 This function appears in FANN >= 2.2.0.
 */ 
FANN_EXTERNAL void FANN_API fann_run_many(struct fann **ann, fann_type * input, fann_type **output, int num_anns, int num_runs);

FANN_EXTERNAL void FANN_API fann_train_on_data_cl(struct fann *ann, struct fann_train_data *data,
                                                  unsigned int max_epochs,
                                                  unsigned int epochs_between_reports,
                                                  float desired_error);

FANN_EXTERNAL float FANN_API fann_train_epoch_cl(struct fann *ann, struct fann_train_data *data);

#endif
#ifdef __cplusplus
}
#endif
