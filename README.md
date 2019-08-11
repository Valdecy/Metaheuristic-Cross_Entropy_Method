# Metaheuristic-Cross_Entropy_Method
Cross Entropy Method (or Kullback–Leibler divergence) to Minimize Functions with Continuous Variables. The function returns: 1) An array containing the used value(s) for the target function and the output of the target function f(x). For example, if the function f(x1, x2) is used, then the array would be [x1, x2, f(x1, x2)].  

* n = Population size.The default value is  5.

* min_values = The minimum value that the variable(s) from a list can have. The Default Value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The Default Value is  5.

* iterations = The total number of iterations. The Default Value is 1000.

* learning_rate = Value that adjusts the Mean and Standard Deviation (of a normal distribution) during search. The Default Value is 0.7.

* k_samples = Number of samples generated in each interation. The Default Value is  2. 

* target_function = Function to be minimized.
