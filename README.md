# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

[<img src="./images/01.png">](#)

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I chose ResNet50 for its ease of use, light weight, and computational power.

To hyperparameter tuning, I tried different values for the following hyperparameters:
- lr: ContinuousParameter(0.001, 0.1)
- batch_size: CategoricalParameter([16, 64])
- epochs: IntegerParameter(5, 10)


[<img src="./images/02.png">](#)

And finally, the best config is:

[<img src="./images/03.png">](#)


After hyperparamater tuning phase, the model will be trained with best hyperparameters in a training job. We can see a
part of the logs inside the job corresponding to training & testing phase of the model.

[<img src="./images/04.png">](#)

## Debugging and Profiling

Since we have a training job with best hyperparameters, we directly debug and profile that job with the following
configuration:

```
rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
]

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, framework_profile_params=FrameworkProfile(num_steps=10)
)

debugger_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)
```

and put them in estimator instance:

[<img src="./images/05.png">](#)


### Results

#### Operators

For both CPU and GPU operators, the three most expensive operations were:
1. copy_
2. to
3. contiguous

which makes sense because these operations deal with memory transfers and allocations.

#### Rules

```LowGPUUtilization``` rule was the most frequently triggered one. It can happen due to bottlenecks, blocking calls
for synchronizations, or a small batch size.

Since the batch size is 16 in our experiment, it's worth to try bigger 
numbers for batch_size hyperparameter because ```BatchSize``` rule was triggered twelve times in the experiment.


## Model Deployment

The model deployment is implemented using a stand-alone script([inference.py](./inference.py) in our project). This
script should at least all the things for inference of the model which is ```model_fn```.

In notebook, we use the script as shown below:

[<img src="./images/06.png">](#)

[<img src="./images/07.png">](#)

With having ```predictor``` instance, we can invoke the endpoint by some predictions:

[<img src="./images/08.png">](#)

As we can see, the model has a successful prediction on this sample case.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
