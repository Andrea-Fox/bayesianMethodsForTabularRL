# Bayesian Methods for Tabular Reinforcement Learning
This repository contains all the code I wrote to create the examples (and the corresponding plots) contained in my master's thesis, which can be found in the file called `masterThesisAndreaFox.pdf`


## Abstract
In recent years Reinforcement Learning has become an important field of interest in Artificial Intelligence. Due to its flexibility, it has gained importance in several areas of scientific research and its applications in robotics or finance proved to be very effective. Reinforcement Learning differentiates from the other major paradigms of machine learning (supervised and unsupervised learning) for its ability to learn from repeated interactions with the environment, with the goal of maximizing an appropriately defined reward.

Most of the applications to the real world require some kind of approximation due to the complexity of the problems studied. In the following pages, the focus will be on more simple environments, which can be studied using methods that do not involve any kind of approximation. These methods, called tabular methods, represent also the building block for all the approximated ones, and are thus very relevant to understand even the more complex problems.
One of the main issues in the field is the so called exploration-exploitation dilemma, that arises as soon as we have to define a strategy to select the actions to execute. This can be summarized by the question we face at each step: is it more convenient to explore and therefore improve our knowledge of the environment, or exploit the available information and choose only those actions that are currently thought to be optimal? The following chapters will introduce some of the most important methods to train an agent to find an optimal strategy that copes with simple problems, as well as an introduction to some of the exploration strategies that try to deal with the exploration-exploitation dilemma. In particular, of mathematical interest will be the duality between the model-free methods and the Bayesian ones, with the former that build their optimal strategy through point estimates and the latter that make use of appropriate distributions and the Bayesian framework to improve the performances.





