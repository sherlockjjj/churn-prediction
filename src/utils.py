import matplotlib.pyplot as plt
import numpy as np

def plot_bar(features, importance):
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize = (8,6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('importance')
    ax.set_title('Feature Importance')
    # plt.savefig('../images/Feature Importance.png')
    plt.show()
