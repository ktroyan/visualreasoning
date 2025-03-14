"""
This file can be used as a script or its main function can be called. 
In both cases, its purpose is to generate CVR data.
"""

import os

# Personal codebase dependencies
from external.tasks import TASKS
from external.generalization_tasks import TASKS as TASKS_GEN
from external.generate_dataset import generate_dataset

ALL_TASKS_ID_TO_NAME = {
    # elementary
    0: "task_shape",
    1: "task_pos",
    2: "task_size",
    3: "task_color",
    4: "task_rot",
    5: "task_flip",
    6: "task_count",
    7: "task_inside",
    8: "task_contact",
    # compositions
    9: "task_sym_rot",
    10: "task_sym_mir",
    11: "task_pos_pos_1",
    12: "task_pos_pos_2",
    13: "task_pos_count_2",
    14: "task_pos_count_1",
    15: "task_pos_pos_4",
    16: "task_pos_count_3",
    17: "task_inside_count_1",
    18: "task_count_count",
    19: "task_shape_shape",
    20: "task_shape_contact_2",
    21: "task_contact_contact_1",
    22: "task_inside_inside_1",
    23: "task_inside_inside_2",
    24: "task_pos_inside_3",
    25: "task_pos_inside_1",
    26: "task_pos_inside_2",
    27: "task_pos_inside_4",
    28: "task_rot_rot_1",
    29: "task_flip_flip_1",
    30: "task_rot_rot_3",
    31: "task_pos_pos_3",
    32: "task_pos_count_4",
    33: "task_size_size_1",
    34: "task_size_size_2",
    35: "task_size_size_3",
    36: "task_size_size_4",
    37: "task_size_size_5",
    38: "task_size_sym_1",
    39: "task_size_sym_2",
    40: "task_color_color_1",
    41: "task_color_color_2",
    42: "task_sym_sym_1",
    43: "task_sym_sym_2",
    44: "task_shape_contact_3",
    45: "task_shape_contact_4",
    46: "task_contact_contact_2",
    47: "task_pos_size_1",
    48: "task_pos_size_2",
    49: "task_pos_shape_1",
    50: "task_pos_shape_2",
    51: "task_pos_rot_1",
    52: "task_pos_rot_2",
    53: "task_pos_color_1",     # Note that we changed the name of the task from "task_pos_col_1" to "task_pos_color_1" compared to the original CVR dataset
    54: "task_pos_color_2",     # Note that we changed the name of the task from "task_pos_col_2" to "task_pos_color_2" compared to the original CVR dataset
    55: "task_pos_contact",
    56: "task_size_shape_1",
    57: "task_size_shape_2",
    58: "task_size_rot",
    59: "task_size_inside_1",
    60: "task_size_contact",
    61: "task_size_count_1",
    62: "task_size_count_2",
    63: "task_shape_color",
    64: "task_shape_color_2",
    65: "task_shape_color_3",
    66: "task_shape_inside",
    67: "task_shape_inside_1",
    68: "task_shape_count_1",
    69: "task_shape_count_2",
    70: "task_rot_color",
    71: "task_rot_inside_1",
    72: "task_rot_inside_2",
    73: "task_rot_count_1",
    74: "task_color_inside_1",
    75: "task_color_inside_2",
    76: "task_color_contact",
    77: "task_color_count_1",
    78: "task_color_count_2",
    79: "task_inside_contact",
    80: "task_contact_count_1",
    81: "task_contact_count_2",
    82: "task_size_color_1",
    83: "task_size_color_2",
    84: "task_color_sym_1",
    85: "task_color_sym_2",
    86: "task_shape_rot_1",
    87: "task_shape_contact_5",
    88: "task_rot_contact_1",
    89: "task_rot_contact_2",
    90: "task_inside_sym_mir",
    91: "task_flip_count_1",
    92: "task_flip_inside_1",
    93: "task_flip_inside_2",
    94: "task_flip_color_1",
    95: "task_shape_flip_1",
    96: "task_rot_flip_1",
    97: "task_size_flip_1",
    98: "task_pos_rot_3",
    99: "task_pos_flip_1",
    100: "task_pos_flip_2",
    101: "task_flip_contact_1",
    102: "task_flip_contact_2",
}

# Create the dict task name to task id
ALL_TASKS_NAME_TO_ID = {v: k for k, v in ALL_TASKS_ID_TO_NAME.items()}

def main(data_dir, tasks, train_size, val_size, test_size, test_gen_size, image_size, seed):
    """
    Generate CVR data.

    Args:
        data_dir (str): where to save the generated data
        tasks (str, list, int): tasks to generate
        train_size (int): number of training examples to generate per task
        val_size (int): number of validation examples to generate per task
        test_size (int): number of test examples to generate per task
        test_gen_size (int): number of sys-gen test examples to generate per task
        image_size (int): size of the images
        seed (int): random seed
    """

    print(f"""Starting generation of tasks {tasks} with:
          seed {seed},
          train_size {train_size},
          val_size {val_size},
          test_size {test_size},
          test_gen_size {test_gen_size}
          image_size {image_size}
        """)

    # Create folder if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # To generate all the tasks (103)
    if tasks == 'all':
        for i in range(0, 103):
            print(f"Generating task {i}")
            tn, tfn, _ = TASKS[i]
            _, tfn_g, _ = TASKS_GEN[i]
            generate_dataset(task_name=tn, 
                             task_fn=tfn, 
                             task_fn_gen=tfn_g, 
                             data_path=data_dir, 
                             image_size=image_size, 
                             seed=seed, 
                             train_size=train_size, 
                             val_size=val_size, 
                             test_size=test_size, 
                             test_gen_size=test_gen_size)

            print(f"Generated {i+1}/103 tasks")

    elif isinstance(tasks, list):
        num_tasks = len(tasks)
        for i, task_id in enumerate(tasks):
            print(f"Generating task {task_id}")
            tn, tfn, _ = TASKS[task_id]
            _, tfn_g, _ = TASKS_GEN[task_id]
            generate_dataset(task_name=tn,
                             task_fn=tfn, 
                             task_fn_gen=tfn_g, 
                             data_path=data_dir, 
                             image_size=image_size, 
                             seed=seed, 
                             train_size=train_size, 
                             val_size=val_size, 
                             test_size=test_size, 
                             test_gen_size=test_gen_size)

            print(f"Generated {i+1}/{num_tasks} tasks\n")
    
    # To generate a single specific task
    elif isinstance(tasks, int):
        print(f"Generating task {tasks}")
        tn, tfn, _ = TASKS[tasks]
        _, tfn_g, _ = TASKS_GEN[tasks]
        generate_dataset(task_name=tn,
                         task_fn=tfn,
                         task_fn_gen=tfn_g, 
                         data_path=data_dir, 
                         image_size=image_size, 
                         seed=seed, 
                         train_size=train_size, 
                         val_size=val_size, 
                         test_size=test_size, 
                         test_gen_size=test_gen_size)

        print(f"Generated task with task id: {tasks}")

    else:
        raise ValueError("tasks must be 'all' (to generate all tasks), a list of integers (to generate a subset of tasks) or an integer (to generate a specific task)")

    print(f"Generation done! Data saved at: {data_dir}")

if __name__ == '__main__':

    # Parameters
    train_size = 1000  # originally: 10000
    val_size = 500  # originally: 500
    test_size = 1000    # originally: 1000
    test_gen_size = 1000    # originally: 500
    image_size = 64 # originally: 128
    seed = 1997
    data_dir = f"./generated_data_{image_size}x{image_size}"


    # Tasks to generate
    elem_tasks_considered = ["task_pos", "task_rot", "task_count", "task_color"]
    comp_tasks_considered = ["task_rot_rot_1", "task_count_count", "task_color_color_1", "task_pos_rot_1", "task_color_count_1", "task_rot_count_1", "task_rot_color", "task_pos_pos_1", "task_pos_color_1", "task_pos_count_1"]
    task_considered = elem_tasks_considered + comp_tasks_considered
    
    tasks = [ALL_TASKS_NAME_TO_ID[task_name] for task_name in task_considered]  # list of tasks
    # tasks = 'all' # all tasks
    # tasks = list(range(0,9)) # elementary tasks
    # tasks = 0 # single task

    # Generate data
    main(data_dir, tasks, train_size, val_size, test_size, test_gen_size, image_size, seed)