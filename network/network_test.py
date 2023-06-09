import numpy as np

#def read_input(input):
    #raw_data = input.strip().replace("[", "").replace("]", "").split()
    #for i in range(len(raw_data)):
      #  print(raw_data)
    # return ""

def read_output(output):
    raw_data = output.strip().replace("[", "").replace("]", "")
    return [float(raw_data.split()[0].strip()), float(raw_data.split()[1].strip())]

#with open("../controllers/controller_pre_trainer/inputs.txt") as input_data:
    #input_stripped = ''.join(input_data.readlines())
    #input_list = list(map(read_input, input_stripped))
 #   input_list = np.fromstring(''.join(input_data.readlines()), dtype=list)
  #  print(input_list)

# Specify the file path
file_path = '../controllers/controller_pre_trainer/inputs.txt'

# Read the contents of the file
with open(file_path, 'r') as file:
    contents = file.read()

# Parse the nested list using eval()
nested_list = eval(contents)

# Convert the nested list to a NumPy array
numpy_array = np.array(nested_list)

# Print the NumPy array
print(numpy_array)

with open("../controllers/controller_pre_trainer/outputs.txt") as output_data:
    output_stripped = output_data.readline().split("][")
    output_list = list(map(read_output, output_stripped))
    print(output_list)