# ínitiate gOPython to call gPROMS
import gopython
import numpy as np
import csv
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from scipy.stats.qmc import Sobol, LatinHypercube

# function to suppress output
def generate_sobol_points(lower_bounds, upper_bounds, num_points):
    """
    generate Sobol points within given upper and lower bounds using scipy.stats.qmc.Sobol.
    :param lower_bounds: A list of lower bounds for each dimension.
    :param upper_bounds: A list of upper bounds for each dimension.
    :param num_points: Number of Sobol points to generate.
    :return: Array of Sobol points within the specified bounds.
    """
    # flatten the bounds lists
    flat_lower_bounds = np.concatenate(([lower_bounds[0], lower_bounds[1]], lower_bounds[2]))
    flat_upper_bounds = np.concatenate(([upper_bounds[0], upper_bounds[1]], upper_bounds[2]))

    # number of dimensions
    dim = len(flat_lower_bounds)

    # initialize Sobol sampler
    sampler = Sobol(d=dim)

    # generate Sobol points in [0,1]^dim
    sobol_points_unit = sampler.random(n=num_points)

    # scale points to the desired range [lower_bounds, upper_bounds]
    scaled_points = np.zeros_like(sobol_points_unit)
    for i in range(dim):
        scaled_points[:, i] = flat_lower_bounds[i] + sobol_points_unit[:, i] * (flat_upper_bounds[i] - flat_lower_bounds[i])
    return scaled_points

# process specification ranges
pressureUpperBound = 2.5*101325
pressureLowerBound = 101325
temperatureUpperBound = 450
temperatureLowerBound = 298
compositionUpperBounds = [0.085, 0.085, 0.4]
compositionLowerBounds = [0.025, 0.015, 0.2]
composition = ["ABSORBENT", "CO2", "N2"]

upperBounds = [temperatureUpperBound, pressureUpperBound, compositionUpperBounds]
lowerBounds = [temperatureLowerBound, pressureLowerBound, compositionLowerBounds]

# this is for ...
groupsGC = np.array([1,     0,     0,    0,  0,  0, 0, 0,   0,          1,  0, 0, 1])

# order: [T,P,amine,co2,n2]
numberOfPoints = pow(2, 14)
sobol_points = generate_sobol_points(lowerBounds,upperBounds,numberOfPoints)
print(upperBounds)
print(lowerBounds)
print(numberOfPoints)
print(sobol_points.shape[0])
print(sobol_points[0,])

# call gproms to ...?
gopython.start_only()
gopython.select("gOPythonSamanVariableGroupInput", "gOPythonSamanVariableGroupInput")
gopython.simulate("gOPythonSamanVariableGroupInput")

# this is doing ...?
null = io.StringIO()
i = 1
results = []
x = 1000
for lineInput in sobol_points:
    # Redirect stdout and stderr to the null object
    with redirect_stdout(null), redirect_stderr(null):
        #composition_scaled = lineInput[2:] / np.sum(np.abs(lineInput[2:]))
        waterComposition = 1-np.sum(lineInput[2:])
        compositionInput = np.concatenate((np.array(waterComposition, ndmin=1), lineInput[2:]))
        gPROMSInput = np.concatenate((lineInput[:2], compositionInput, groupsGC))
        status, result = gopython.evaluate(gPROMSInput)
        pass
    results.append(np.concatenate((result[:2], result[2:6], result[6:10], result[10:14], result[14:16])))
    i = i + 1
    if i % x == 0:
        print(f'Iteration {i}: This is every {x}th iteration.')
gopython.stop()

strCompositions = []
compositionTypes = ['z', 'x', 'y']
compositionPrint = ["water", "ABSORBENT", "CO2", "N2"]
for compositionType in compositionTypes:
    for species in compositionPrint:
        strCompositions.append(compositionType + '_' + species)

header = ['Temperature', 'Pressure'] + strCompositions + ['LiquidPhaseFraction', 'VapourPhaseFraction']
print(header)
with open('output_7.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(results)

