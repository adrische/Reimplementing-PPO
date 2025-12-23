import multiprocessing as mp
from itertools import product
import pprint


def wrapper_function_with_printing(id_params_tuple):
        id, fun, params = id_params_tuple

        print(f"Starting run {id} with input: {params}")

        output = fun(**params)

        print(f"Output with params {params} is : {output}")

        return  {"id": id, "params": params, "output": output}


def trainparallel(fun, params, processes=None):
    """Run fun in parallel for all elements in params.

    Call "fun" for all combinations of parameters in a parallel fashion.

    -- params is a dictionary of the form {"alpha": [0.01, 0.001], "seed": [42]}
    -- processes is passed to Pool: processes is the number of worker processes to use. 
       If processes is None (default) then the number returned by os.process_cpu_count() is used.
    """
    all_params_grid = []

    for i, t in enumerate(product(*params.values())):
        all_params_grid.append([i, fun, dict(zip(params.keys(), t))])

    print(f"Starting {len(all_params_grid)} runs!\n")

    with mp.Pool(processes=processes) as p:
        results = p.map(wrapper_function_with_printing, all_params_grid)

    print(f"\nCompleted {len(results)} runs!\n")

    return(results)