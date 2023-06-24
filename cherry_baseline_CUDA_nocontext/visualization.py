import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(epochs,train_losses,val_losses,exp_dir):
    # plot lines
    plt.plot(epochs, train_losses, label="training loss",linestyle="--",color='green')
    plt.plot(epochs, val_losses, label="validation loss",linestyle="-",color='red')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.xticks(epochs)
    plt.yticks(np.arange(0.0, 1.1, 0.1))  # Y axis ticks start deom 0 to 1 with a step of 0.1
    plt.legend()
    plt.savefig(exp_dir+'loss.png')
    plt.show()