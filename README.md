# Homework

The task was named entity recognition. I was provided with a small dataset from German wikipedia with location, person and organisation entities tagged. My task was to build a model that could do this automatically.

The dataset is small and my computer is too weak to use pre-trained models like BERT or even publicly available FastText binaries. Therefore I felt that I would be unable to optimize for performance, and instead chose to optimize my code towards modularity and scalability. In my experience, the most important part of a research project is the ability to quickly change the models, and the ability to compare them.

Lastly, this repository represents roughly 12 hours of work. This is not nearly the finished version. However, I feel that my intentions are clear enough, even in this unfinished state.

## Running the code
All python package requirements are in `requirements.txt`. You know what to do ;)

There are two runable scripts: `main.py` and `k-fold.py`. The first is capable of training and saving a single model. Alternatively, it can grid search through hyperparameters to find the best models per each metric. It logs all training processes into csv formatted files, for model comparability. The second one demonstrates a k-fold cross-validation on the given data. Given more time, the grid search and cross-validation should be integrated together, since the tiny amount of data leads to huge variance in results.

## Models
As mentioned above, the models are intentionaly kept very simple. In fact, the only model available currently is the bidirectional single-layer LSTM. That is because it adequately demonstrates my intentions with modularity, and is lightweight enough that my machine can handle it. Model can be found in `models.py`.

## Evaluation
The generally accepted form of NER labelling is to include the information about starts and ends of the entity. Therefore our aim here is to accurately predict not only the entity of a given token (here we tokenize by words), but also its position in the whole entity.
Example:
 * **Deutchland** has the label [_loc-U_] where U means that the token is the whole entity.
 * **Otto von Bismarck** has labels [_per-S_. _per-I_, _per-E_], as in Start, Inside, End.
 
We evaluate exact match, binary accuracy and binary f1. For further information see `eval.py`.

## Further work, or "If I had more time":
The user experience is as of now very bare-bones. The hyperparameters must be changed in code, same for the names of the log files and model files. There is as yet no capability for checkpointing or loading models. There is also no end-to-end module for inference. The project structure is also a bit messy, and could use some improvement. Deployment is very basic, and could use a docker. This repo also contains the dataset, which is considered bad practice (however it is tiny, so it doesn't matter too much in this case).
If given enough time, I would also add a place-agnostic evaluator, i.e. an evaluator that only looks at what type of entity the model predicts, but ignores the positional part of the label.
I would also add some utility to visualise and compare the training processes using graphs.

## Strong points, or "what I am proud of"
I think that the strongest point of my solution is its **scalability**. The training and evaluation are all very modular, and will work with any new model or evaluator. Furthermore, I think that **logging** is very important and often overlooked part of ML research, and I think that my logs could be very useful for getting the most information possible.
