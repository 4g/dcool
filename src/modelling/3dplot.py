# library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.animation as animation

url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)

fig = plt.figure()

def animate(i):
    # Get the data (csv file is hosted on the web)
    # Transform it to a long format
    df = data.unstack().reset_index()
    df.columns = ["X", "Y", "Z"]
    print(i)

    # And transform the old column name in something numeric
    df['X'] = pd.Categorical(df['X'])
    df['X'] = df['X'].cat.codes

    # Make the plot
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'].values, df['X'].values, df['Z'].values, cmap=plt.cm.viridis, linewidth=.4)
    plt.show()



ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()