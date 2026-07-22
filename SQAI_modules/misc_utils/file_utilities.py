import json


class input_reader:
    def __init__(self, file_name):
        json_open = open(file_name, "r")
        self.input_data = json.load(json_open)

    def load_data(self, section, key):
        data = self.input_data[section][key]
        print(f"{section}:{key} = {data}")
        return data
