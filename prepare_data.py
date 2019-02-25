import os
import shutil
import random

data_ture = sorted(os.listdir("crackSubImageForTraining/100BY100/accept"))
data_false = sorted(os.listdir("crackSubImageForTraining/100BY100/nonaccpet"))

random.shuffle(data_ture)
random.shuffle(data_ture)
random.shuffle(data_ture)

random.shuffle(data_false)
random.shuffle(data_false)
random.shuffle(data_false)

# print(len(data_ture), len(data_false))

# for i in range(6300):
#     shutil.copyfile("crackSubImageForTraining/100BY100/accept/" + data_ture[i],
#                     "data/train_true/" + data_ture[i])
#     shutil.copyfile(
#         "crackSubImageForTraining/100BY100/nonaccpet/" + data_false[i],
#         "data/train_false/" + data_false[i])
#
# for i in range(6301, 7001):
#     shutil.copyfile("crackSubImageForTraining/100BY100/accept/" + data_ture[i],
#                     "data/test_true/" + data_ture[i])
#     shutil.copyfile(
#         "crackSubImageForTraining/100BY100/nonaccpet/" + data_false[i],
#         "data/test_false/" + data_false[i])
