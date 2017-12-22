import numpy as np
import sys
import os.path

if len(sys.argv) != 2:
    print("args is not valid")
    sys.exit()

# Read in data 
if not os.path.isfile(sys.argv[1]):
    print("Data file doesn't exist")
    sys.exit()

my_data = np.genfromtxt(sys.argv[1], delimiter=',', dtype='float64')

# Split instances into classes
incomplete_instances = []
negative_instances = []
positive_instances = []
for row in my_data:
    if -1 in row:
        incomplete_instances.append(row)
    elif row[-1] == 0:
        negative_instances.append(row)
    elif row[-1] == 1:
        positive_instances.append(row)
incomplete_instances = np.array(incomplete_instances)
negative_instances = np.array(negative_instances)
positive_instances = np.array(positive_instances)

# Fill in missing attribute data
if incomplete_instances.shape[0] > 0:
    for idx, elem in np.ndenumerate(incomplete_instances):
        if elem == -1:
            if incomplete_instances[idx[0],-1] == 0:
                random_attr_val = np.random.choice(negative_instances[:,idx[1]])
            elif incomplete_instances[idx[0],-1] == 1:
                random_attr_val = np.random.choice(positive_instances[:,idx[1]])
            incomplete_instances[idx] = random_attr_val

    my_data = np.concatenate((incomplete_instances, positive_instances, negative_instances))
np.random.shuffle(my_data)
if not 'voting' in sys.argv[1]:
    np.savetxt('test.out', my_data, delimiter=',', fmt='%f')
else:
    np.savetxt('test.out', my_data, delimiter=',', fmt='%d')


# Convert discrete attributes into boolean ones
converted_result = ""
file_in = open('test.out', 'r')
file_out = open('input_data_encoded.txt', 'w')
if 'cancer' in sys.argv[1]:
    for line in file_in:
        split_line = line.split(',')
        for attr in split_line[:-1]:
            converted_result += \
                (",".join("1" if int(attr) == i + 1 else "0" for i in range(10))) + ","
        converted_result += split_line[-1]
elif 'glass_data' in sys.argv[1] or 'iris' in sys.argv[1]:
    if 'glass_data' in sys.argv[1]:
        max_vals = [1.533931, 17.381, 4.491, 3.51, 75.411, 6.211, 16.191, 3.151, 0.511]
        min_vals = [1.51115, 10.73, 0, 0.29, 69.81, 0, 5.43, 0, 0]
    elif 'iris' in sys.argv[1]:
        max_vals = [7.91, 4.41, 6.91, 2.51]
        min_vals = [4.3, 2.0, 1.0, 0.1]
    bin_widths = [(max_vals[idx] - min_vals[idx])/10 for idx, val in enumerate(max_vals)]
    for line in file_in:
        split_line = line.split(',')
        for idx,val in enumerate(split_line[:-1]):
            bin_num = int((float(val) - min_vals[idx]) / bin_widths[idx])
            converted_result += \
                (",".join("1" if bin_num + 1 == i + 1 else "0" for i in range(10))) + ","
        converted_result += str(int(float(split_line[-1]))) + '\n'
elif 'soybean' in sys.argv[1]:
    for line in file_in:
        split_line = line.split(',')
        for idx, val in enumerate(split_line[:-1]):
            if idx == 0:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(7))) + ","
            elif idx == 2:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 3:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 5:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(4))) + ","
            elif idx == 6:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(4))) + ","
            elif idx == 7:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 9:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 13:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 14:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 17:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 20:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(4))) + ","
            elif idx == 21:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(4))) + ","
            elif idx == 24:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(3))) + ","
            elif idx == 26:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(4))) + ","
            elif idx == 28:
                converted_result += \
                    (",".join("1" if int(float(val)) == i else "0" for i in range(5))) + ","
            else:
                converted_result += ("1" if int(float(val)) == 1 else "0") + ","
        converted_result += str(int(float(split_line[-1]))) + '\n'
file_out.write(converted_result)
if not 'voting' in sys.argv[1]:
    os.remove('test.out')
