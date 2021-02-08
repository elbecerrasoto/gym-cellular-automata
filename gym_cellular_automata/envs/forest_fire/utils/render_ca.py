def render(self, title='Forest Fire Automaton'):
    grid = self.grid_to_rgba()
    """Automaton visualization"""
    # Plot style
    sns.set_style('whitegrid')
    # Main Plot
    plt.imshow(grid, aspect='equal')
    # Title showing Reward
    plt.title(title, **self.title_font)
    # Modify Axes
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, self.n_col, 1))
    ax.set_yticks(np.arange(0, self.n_row, 1))
    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, self.n_col, 1), **self.axes_font)
    ax.set_yticklabels(np.arange(0, self.n_row, 1), **self.axes_font)
    # Minor ticks
    ax.set_xticks(np.arange(-.5, self.n_col, 1), minor=True)
    ax.set_yticks(np.arange(-.5, self.n_row, 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=2)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    fig = plt.gcf()
    plt.show()
    return fig
    




# Render Info
title_font = {'fontname':'Comfortaa'}
axes_font = {'fontname':'Comfortaa'}
alpha = int(1.0*255)
color_tree = np.array([15, 198, 43, alpha]) # Green RGBA
color_empty = np.array([255, 245, 166, alpha]) # Beige RGBA
color_fire = np.array([255, 106, 58, alpha]) # Red RGBA
color_rock = np.array([179, 139, 109, alpha]) # Brown RGBA
color_lake = np.array([131, 174, 255, alpha]) # Blue RGBA



def grid_to_rgba(self):
    rgba_mat = self.grid.tolist()
    for row in range(self.n_row):
        for col in range(self.n_col):
            if rgba_mat[row][col] == self.tree:
                rgba_mat[row][col] = self.color_tree
            elif rgba_mat[row][col] == self.empty:
                rgba_mat[row][col] = self.color_empty
            elif rgba_mat[row][col] == self.fire:
                rgba_mat[row][col] = self.color_fire
            elif rgba_mat[row][col] == self.rock:
                rgba_mat[row][col] = self.color_rock
            elif rgba_mat[row][col] == self.lake:
                rgba_mat[row][col] = self.color_lake
            else:
                raise ValueError('Error: Unidentified cell')
    rgba_mat = np.array(rgba_mat)
    return rgba_mat