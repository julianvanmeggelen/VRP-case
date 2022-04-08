from validator.InstanceCO22 import InstanceCO22

testfile = "./Instances/Instance_1-10/Instance_1.txt"
instance = InstanceCO22(inputfile = testfile, filetype = 'txt')
print(instance.Requests)
