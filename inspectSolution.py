import streamlit as st
from util import *
from validator.Validate import SolutionCO22
import seaborn as sns 
sns.set()

#
# Streamlit app to inspect solution file routes
#



i = st.selectbox(label="instance nr", options = list(range(1,31)))
i = int(i)
if i != 0:
    instance = loadInstance(i)
    fig = getPlotInstance(instance)
    st.pyplot(fig)

def plotSolution(sol: SolutionCO22, day):
    fig, ax = plt.subplots()

    colors = ['b','g','y','b']

    for i, route in enumerate(sol.Days[day-1].VanRoutes):
        locHubX = instance.Locations [route.HubID].X
        locHubY = instance.Locations [route.HubID].Y
        locX = [instance.Locations[instance.Requests[i-1].customerLocID-1].X for i in route.Route]
        locY = [instance.Locations[instance.Requests[i-1].customerLocID-1].Y for i in route.Route]
        locX = [locHubX] + locX + [locHubX]
        locY = [locHubY] + locY + [locHubY]
        color = colors[i%len(colors)]
        ax.plot(locX, locY, c=color)

    for route in sol.Days[day-1].TruckRoutes:
        locDepotX = instance.Locations [0].X
        locDepotY = instance.Locations [0].Y
        locX = [instance.Locations[i].X for i in route.Route]
        locY = [instance.Locations[i].Y for i in route.Route]
        locX = [locDepotX] + locX + [locDepotX]
        locY = [locDepotY] + locY + [locDepotY]
        ax.plot(locX, locY, c="red")

    reqs = [_ for _ in instance.Requests if _.desiredDay is day]
    locX = [instance.Locations[req.customerLocID-1].X for req in reqs]
    locY = [instance.Locations[req.customerLocID-1].Y for req in reqs]
    ax.scatter(locX, locY,marker='.', alpha=1)

    locX = [_.X for _ in instance.Locations]
    locY = [_.Y for _ in instance.Locations]
    nHubs = len(instance.Hubs)
    ax.scatter(locX[0], locY[0], marker=",", label="Depot")
    ax.scatter(locX[1:1+nHubs], locY[1:1+nHubs],marker="^", label="Hub")
    ax.scatter(locX[1+nHubs:], locY[1+nHubs:],marker='.', alpha=0.3)
    fig.legend()
    #fig.suptitle(f"Instance {i} day {day}")
    return fig

solutionfile = st.file_uploader("Upload solution")
if solutionfile:
    try:
        temppath = "tempsol.txt"
        with open(temppath, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(solutionfile.getbuffer())
        solution = SolutionCO22(temppath, instance, 'txt')
        day = st.selectbox(label="Day", options = list(range(1,instance.Days+1)))
        st.pyplot(plotSolution(solution, day))
        st.write(solution.calcCost.__dict__)
    except Exception as e:
        st.write("Make sure the solution is for the selected instance")
        print(e)






    


