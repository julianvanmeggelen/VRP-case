import streamlit as st
from util import *
from validator.Validate import SolutionCO22
import seaborn as sns 
sns.set()
i = st.selectbox(label="instance nr", options = list(range(1,31)))
i = int(i)
if i != 0:
    instance = loadInstance(i)
    fig = getPlotInstance(instance)
    st.pyplot(fig)

def plotSolution(sol: SolutionCO22, day):
    fig, ax = plt.subplots()
    for route in sol.Days[day-1].TruckRoutes:
        locDepotX = instance.Locations [0].X
        locDepotY = instance.Locations [0].Y
        locX = [instance.Locations[i].X for i in route.Route]
        locY = [instance.Locations[i].Y for i in route.Route]
        locX = [locDepotX] + locX + [locDepotX]
        locY = [locDepotY] + locY + [locDepotY]
        ax.plot(locX, locY)

    for route in sol.Days[day-1].VanRoutes:
        locHubX = instance.Locations [route.HubID].X
        locHubY = instance.Locations [route.HubID].Y
        locX = [instance.Locations[instance.Requests[i-1].customerLocID-1].X for i in route.Route]
        locY = [instance.Locations[instance.Requests[i-1].customerLocID-1].Y for i in route.Route]
        locX = [locHubX] + locX + [locHubX]
        locY = [locHubY] + locY + [locHubY]
        ax.plot(locX, locY)

    locX = [_.X for _ in instance.Locations]
    locY = [_.Y for _ in instance.Locations]
    nHubs = len(instance.Hubs)
    ax.scatter(locX[0], locY[0], marker=",", label="Depot")
    ax.scatter(locX[1:1+nHubs], locY[1:1+nHubs],marker="^", label="Hub")
    ax.scatter(locX[1+nHubs:], locY[1+nHubs:],marker='.', alpha=0.3)
    fig.legend()
    return fig

solutionfile = st.file_uploader("Upload solution")
if solutionfile:
    temppath = "tempsol.txt"
    with open(temppath, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(solutionfile.getbuffer())
    solution = SolutionCO22(temppath, instance, 'txt')
    day = st.selectbox(label="Day", options = list(range(1,instance.Days+1)))
    st.pyplot(plotSolution(solution, day))
    st.write(solution.calcCost.__dict__)






    


