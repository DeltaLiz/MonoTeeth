import os
import json



def load_specifications(specs_file_name):

    if not os.path.isfile(specs_file_name):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
        )

    return json.load(open(specs_file_name))