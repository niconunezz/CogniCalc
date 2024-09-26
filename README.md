# CogniCalc

![CalcIm](calc.jpeg)


> The most exciting phrase to hear in science, the one that heralds new discoveries, is not 'Eureka!' but 'That's funny...' - Isaac Asimov

This project is inspired by the exercises proposed in Karpathy's GPT video. The main idea is to create a GPT model that functions as a calculator. The project is evidently not useful in a practical sense and has as its sole purpose learning and understanding the GPT architecture in more depth.

**Main files**

- **data.py**: Contains the get_batch function that provides x and y data for training.
- **model.py**: Contains all the components of the final architecture, almost identical to the one in Karpathy's video.
- **train.py**: Main file, imports get_batch and the model, and performs the learning.

**Current state:**
Right now the model only supports addition on positive numbers, more operations are coming soon...

**Trying the model:**
Start by creating and activating a virtual environment, then run:
```bash
pip install -r requirements.txt
```
And that's it!
Change the hyperparameters as you wish in the Config class, and use the generate function to see the model calculate!