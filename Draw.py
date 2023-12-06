import matplotlib.pyplot as plt


def drawPolicy(policy,name,title,text={"G":[], "E":[]}):
    arrows = {3: (1, 0), 2: (-1, 0), 0: (0, 1), 1: (0, -1)}
    scale = 0.25
    ar = policy
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for row_index, row in enumerate(ar):
        for col_index, direction in enumerate(row):
            start_x = col_index + 0.5
            start_y = len(ar) - 0.5 - row_index
            
            if (row_index, col_index) in text["G"]:
                ax.text(start_x, start_y, 'G', horizontalalignment='center', verticalalignment='center')
            elif (row_index, col_index) in text["E"]:
                ax.text(start_x, start_y, '-', horizontalalignment='center', verticalalignment='center')
            else:
                dx, dy = [scale * i for i in arrows[direction]]
                ax.arrow(start_x, start_y, dx, dy, head_width=0.1)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.grid(which='both')
    plt.title(title)
    plt.savefig(name)
    plt.close()




def drawValue(value,name,title):
    arrows = {3: (1, 0), 2: (-1, 0), 0: (0, 1), 1: (0, -1)}
    scale = 0.25
    ar = value
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for row_index, row in enumerate(ar):
        for col_index, direction in enumerate(row):
            start_x = col_index + 0.5
            start_y = len(ar) - 0.5 - row_index
            ax.text(start_x, start_y, str(value[row_index,col_index]), horizontalalignment='center', verticalalignment='center')
            
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.grid(which='both')
    plt.title(title)
    plt.savefig(name)
    plt.close()
