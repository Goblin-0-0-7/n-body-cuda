import os
import json
import matplotlib.pyplot as plt

gpu = "A1000"
testtitle = "density_test"

def readTestFile(filedir):
    with open(filedir, 'r') as f:
        data = json.load(f)
    return data

def getTestNames(json_data):
    test_names: list[str] = []
    for test in json_data["test_cases"]:
        test_names.append(test)
    return test_names

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
        energy_change_perc = (float(energy_value) - base_energy) / base_energy
        energy_data.append(energy_change_perc)
    return steps, energy_data

def plotTest(ax, steps, energy_data, name:str):

    ax.plot(
        steps,
        energy_data,
        marker = "+",
        linestyle = "-",
        label = name
    )

def savePlot(ax, name, workdir):
    ax.set_xlabel(r"Steps", fontsize=14) # '$$' enables latex-math_mode
    ax.set_ylabel(r"$\frac{\mathrm{\Delta}E}{E}$", fontsize=14)
    plt.legend()
    plt.savefig(workdir + "\\plots\\" + name + ".png")
    plt.close()
    print(f"Saved {name} plot.")
    
workdir = os.getcwd()
testData = readTestFile(workdir + f"\\tests\\{testtitle}.json")
testnames = getTestNames(testData)

fig, ax = plt.subplots(figsize= (15, 8.2))
plt.grid(True)
for name in testnames:
    strData = loadEnergyStr(workdir + f"\\testresults\\{name}\\energy.txt")
    steps, energy_data = extractEnergyData(strData)
    plotTest(ax, steps, energy_data, name)
savePlot(ax, f"{testtitle}-{gpu}-energy", workdir)

