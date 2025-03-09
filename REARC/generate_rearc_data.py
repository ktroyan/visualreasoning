# NOTE: currently, the script would overwrite the data if an already existing file is used.

# Personal codebase dependencies
from main import generate_dataset

def main(base_data_dir, seed, n_tasks, n_examples, difficulties_lb, difficulties_ub, specific_tasks):
    """
    Generate REARC data.

    Args:
        path (str): where to save the generated data
        seed (int): random seed
        n_tasks (int): number of tasks to generate
        n_examples (int): number of examples to generate per task
        difficulties_lb (list): list of lower bounds of the sampled level of difficulty for the examples
        difficulties_ub (list): list of upper bounds of the sampled level of difficulty for the examples
        specific_tasks (list): list of specific tasks (specified by their name without the file extension .json) to generate (if empty, generate n_tasks tasks randomly)
    """

    for index, (diff_lb, diff_ub) in enumerate(zip(difficulties_lb, difficulties_ub)):
        diff_lb = float(diff_lb)
        diff_ub = float(diff_ub)
        diff_lb_name = str(diff_lb).replace('.', '') if diff_lb != 0.0 else '0'
        diff_ub_name = str(diff_ub).replace('.', '') if diff_ub != 1.0 else '1'
        path = f'{base_data_dir}_lb{diff_lb_name}_ub{diff_ub_name}'

        print(f"Generating data with level of difficulty in the range [{diff_lb}, {diff_ub}]")
        generate_dataset(path=path, seed=seed, n_tasks=n_tasks, n_examples=n_examples, diff_lb=diff_lb, diff_ub=diff_ub, specific_tasks=specific_tasks)
        print(f"Generation done! Data saved at: {path}")
        print(f"Performed {index+1}/{len(difficulties_lb)} generations of data")

    # generate_dataset(path='./generated_data_lb0_ub04', seed=1997, n_tasks=n_tasks, n_examples=n_examples, diff_lb=0, diff_ub=0.4, specific_tasks=[])
    # generate_dataset(path='./generated_data_lb0_ub07', seed=1997, n_tasks=n_tasks, n_examples=n_examples, diff_lb=0, diff_ub=0.7, specific_tasks=[])
    # generate_dataset(path='./generated_data_lb07_ub1', seed=1997, n_tasks=n_tasks, n_examples=n_examples, diff_lb=0.7, diff_ub=1, specific_tasks=[])


if __name__ == '__main__':

    # Parameters
    base_data_dir = './generated_data'
    seed = 1997
    n_tasks = 10    # max is 400; tasks are randomly selected unless specific_tasks is provided
    n_examples = 10000
    difficulties_lb = [0.0, 0.0, 0.7]
    difficulties_ub = [0.4, 0.7, 1.0]
    specific_tasks = []

    assert len(difficulties_lb) == len(difficulties_ub), "The number of lower bounds and upper bounds of the level of difficulty must be the same as they define a range."

    # Generate data
    main(base_data_dir, seed, n_tasks, n_examples, difficulties_lb, difficulties_ub, specific_tasks)