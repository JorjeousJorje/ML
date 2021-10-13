import matplotlib.pyplot as plt

def plot_curve(x, y,  title: str, label='PR Curve', xlabel='TPR', ylabel='PPV'):
    with plt.style.context(['science']): 
        plt.figure(dpi=100)
        plt.title(title) 
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.plot(x, y, label=label)
        plt.legend()
        plt.show()