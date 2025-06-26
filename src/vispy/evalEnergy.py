import os
import matplotlib.pyplot as plt

testname = "SampleTest1"

def loadEnergyStr(filedir):
    with open(filedir, 'r') as f:
        data = f.readlines()
    return data

def extractEnergyData(strData):
    base_energy:float = 0
    steps: list[int] = []
    energy_data: list[float] = []
    num_values = int(len(strData) - 1) # dont count header
    for i in range(1, num_values): # start after header
        step, energy_value = strData[i].split(",")
        if (i == 1):
            base_energy = float(energy_value)
        steps.append(int(step))
        energy_data.append(float(energy_value) - base_energy)
    return steps, energy_data


workdir = os.getcwd()
strData = loadEnergyStr(workdir + f"\\testresults\\{testname}\\energy.txt")
steps, energy_data = extractEnergyData(strData)

fig = plt.figure()

plt.plot(
    steps,
    energy_data,
    marker = "+",
    linestyle = "-",
    color = "blue"
)

# TODO: Set title

plt.show()