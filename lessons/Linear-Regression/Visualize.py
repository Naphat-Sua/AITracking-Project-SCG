import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize(datasetX, datasetY, fxList, accuracyList=None, lossList=None,
              title='Regression problems'):
    """Animate the fitted line converging to the data.

    accuracyList/lossList are optional: when given, two extra panels show
    the training accuracy and loss over time.
    """
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Regression problems')

    show_metrics = accuracyList is not None and lossList is not None
    if show_metrics:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)
        lineAccuracy, = ax2.plot([], [], 'b-', label="training accuracy")
        lineLoss, = ax3.plot([], [], 'g-', label="training loss")
    else:
        ax1 = fig.add_subplot(1, 1, 1)

    linePlot, lineFx, lineFinal = ax1.plot([], [], 'bo', [], [], 'g-', [], [], 'r-',
                                           animated=True)
    totalFx = np.shape(fxList)[0]  # all fx

    def init():
        ax1.set_title(title)
        ax1.set_xlim(min(datasetX), max(datasetX))
        ax1.set_ylim(min(datasetY), max(datasetY))

        artists = [linePlot, lineFx, lineFinal]
        if show_metrics:
            ax2.set_title("Accuracy")
            ax2.set_xlim(0, len(accuracyList))  # initial value only
            ax2.set_ylim(0, max(accuracyList) + 5)  # not autoscaled

            ax3.set_title("Loss")
            ax3.set_xlim(0, len(lossList))  # initial value only
            ax3.set_ylim(0, max(lossList) + 0.1)  # not autoscaled
            artists += [lineAccuracy, lineLoss]
        plt.tight_layout()
        return artists

    def update(step):
        # for ax1
        linePlot.set_data(datasetX, datasetY)
        fx = fxList[step]
        lineFx.set_data(datasetX, fx)

        artists = [linePlot, lineFx, lineFinal]
        if show_metrics:
            # for ax2 and ax3
            lineAccuracy.set_data(range(0, step), accuracyList[0:step])
            lineLoss.set_data(range(0, step), lossList[0:step])
            artists += [lineAccuracy, lineLoss]

        if step == totalFx - 1:  # display last line
            print("plot graph finish")
            lineFinal.set_data(datasetX, fx)
        elif step == 0:
            linePlot.set_data([], [])
            lineFx.set_data([], [])
            lineFinal.set_data([], [])
            if show_metrics:
                lineAccuracy.set_data([], [])
                lineLoss.set_data([], [])
            return init()

        return artists

    steps = range(0, totalFx)
    print("Plotting graph\nwaiting.........")
    ani = FuncAnimation(fig, update,
                        init_func=init, frames=steps, repeat=True, blit=True)
    plt.show()
    return ani
