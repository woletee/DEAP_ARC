import numpy as np
from typing import Tuple, List

# Define Count as an alias for int
Count = int
from scipy.ndimage import binary_dilation
from scipy import ndimage
from typing import Callable

class GridList(list):
    pass

class PrimitiveException(Exception):
    pass

# Simple Grid class
class Grid:
    def __init__(self, grid: np.ndarray, position: Tuple[int, int]=(0, 0)):
        self.grid = grid
        self.position = position

    @property
    def size(self):
        return self.grid.shape

    def newgrid(self, grid: np.ndarray, position=None):
        if position is None:
            position = self.position
        return Grid(grid, position)

    def count(self):
        return np.count_nonzero(self.grid)

Colour = int

def primitive_assert(condition, message="Assertion failed"):
    if not condition:
        raise ValueError(message)

# DSL primitives
def rot90(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid))

def rot180(g: Grid) -> Grid:
    return g.newgrid(np.rot90(g.grid, 2))

def ic_compress2(g: Grid) -> Grid:
    keep_rows = np.any(g.grid, axis=1)
    keep_cols = np.any(g.grid, axis=0)
    return g.newgrid(g.grid[keep_rows][:, keep_cols])

def flipy(g: Grid) -> Grid:
    return g.newgrid(np.flip(g.grid, axis=1))

def mirrorX(g: Grid) -> Grid:
    return Grid(np.hstack((g.grid, np.fliplr(g.grid))))

def mirrorY(g: Grid) -> Grid:
    return Grid(np.vstack((g.grid, np.flipud(g.grid))))

def overlay(g: Grid, h: Grid) -> Grid:
    if g.size == h.size:
        newgrid = Grid(g.grid.copy())
        newgrid.grid[h.grid != 0] = h.grid[h.grid != 0]
        return newgrid
    else:
        xpos = min(g.position[0], h.position[0])
        ypos = min(g.position[1], h.position[1])
        xsize = max(g.position[0]+g.size[0], h.position[0]+h.size[0]) - xpos
        ysize = max(g.position[1]+g.size[1], h.position[1]+h.size[1]) - ypos

        newgrid = Grid(np.zeros((xsize, ysize)), position=(xpos, ypos))
        newgrid.grid[g.position[0]-xpos:g.position[0]-xpos+g.size[0], 
                     g.position[1]-ypos:g.position[1]-ypos+g.size[1]] = g.grid

        mask = np.nonzero(h.grid)
        slice = newgrid.grid[h.position[0]-xpos:h.position[0]-xpos+h.size[0],
                             h.position[1]-ypos:h.position[1]-ypos+h.size[1]]
        slice[mask] = h.grid[mask]
        return newgrid

def set_bg(c: Colour, g: Grid) -> Grid:
    primitive_assert(c != 0, "background with 0 has no effect")
    grid = np.copy(g.grid)
    grid[grid == 0] = c
    return g.newgrid(grid)
def ic_composegrowing(l: List[Grid]) -> Grid:
    xpos = min([g.position[0] for g in l])
    ypos = min([g.position[1] for g in l])
    xsize = max([g.position[0]+g.size[0] for g in l]) - xpos
    ysize = max([g.position[1]+g.size[1] for g in l]) - ypos

    newgrid = Grid(np.zeros((xsize, ysize)), position=(xpos, ypos))
    sorted_list = sorted(l, key=lambda g: g.count(), reverse=True)

    for g in sorted_list:
        xstart = g.position[0] - xpos
        ystart = g.position[1] - ypos
        slice = newgrid.grid[xstart:xstart+g.size[0], ystart:ystart+g.size[1]]
        mask = np.nonzero(g.grid)
        slice[mask] = g.grid[mask]

    return newgrid

def ic_splitall(g: Grid) -> GridList:
    colours = np.unique(g.grid)
    ret = []
    for colour in colours:
        if colour:
            labeled_grid, _ = ndimage.label(g.grid == colour)
            objects = ndimage.find_objects(labeled_grid)
            ret += [g.newgrid(g.grid[obj], position=(obj[0].start, obj[1].start)) for obj in objects]
    return ret

def ic_connect_kernel(g: Grid, x: bool, y: bool) -> Grid:
    """
    Implements a generic connect (not a primitive)
    x and y control whether it is enabled in the horizontal and vertical directions
    
    Connect works as follows:
    Two cells in the same row/column are connected iff:
    - they are both non-zero
    - they have the same value c
    - there are only zeros in between them
    Connect fills in the zero values in-between with c.
    Note that vertical connection happens after horizontal connection and "overrides" it.

    This function is SLOW
    TODO: Verify this actually works because I'm not confident
    """
    ret = np.zeros_like(g.grid)

    if x:
        for row in range(g.grid.shape[0]):
            last = last_value = -1
            for col in range(g.grid.shape[1]):
                if g.grid[row, col]:
                    if g.grid[row, col] == last_value:
                        ret[row, last+1:col] = last_value
                    last_value = g.grid[row, col]
                    last = col
    
    if y:
        for col in range(g.grid.shape[1]):
            last = last_value = -1
            for row in range(g.grid.shape[0]):
                if g.grid[row, col]:
                    if g.grid[row, col] == last_value:
                        ret[last+1:row, col] = last_value
                    last_value = g.grid[row, col]
                    last = row

    return g.newgrid(ret)

def ic_connectY(g: Grid) -> Grid:
    return ic_connect_kernel(g, False, True)

def ic_connectX(g: Grid) -> Grid:
    return ic_connect_kernel(g, True, False)

def ic_compress2(g: Grid) -> Grid:
    """Deletes any black rows/columns in the grid"""
    keep_rows = np.any(g.grid, axis=1)
    keep_cols = np.any(g.grid, axis=0)

    return g.newgrid(g.grid[keep_rows][:, keep_cols])

def ic_compress3(g: Grid) -> Grid:
    """
    Keep any rows/columns which differ in any way from the previous row/column
    The first row/column is always kept.
    """
    keep_rows = np.ones(g.grid.shape[0], dtype=bool)
    keep_cols = np.ones(g.grid.shape[1], dtype=bool)

    for row in range(1, g.grid.shape[0]):
        if np.all(g.grid[row] == g.grid[row-1]):
            keep_rows[row] = False

    for col in range(1, g.grid.shape[1]):
        if np.all(g.grid[:, col] == g.grid[:, col-1]):
            keep_cols[col] = False

    return g.newgrid(g.grid[keep_rows][:, keep_cols])

def ic_erasecol(c: Colour, g: Grid) -> Grid:
    "Remove a specified colour from the grid, keeping others intact"
    primitive_assert(c != 0, "erasecol with 0 has no effect")
    grid = np.copy(g.grid)
    grid[grid == c] = 0
    return g.newgrid(grid)
def rarestcol(g: Grid) -> Colour:
    """
    Returns the least common colour, excluding black. 
    Excludes any colours with zero count.
    """
    counts = np.bincount(g.grid.ravel())[1:]
    counts[counts == 0] = 9999
    return np.argmin(counts)+1

def left_half(g: Grid) -> Grid:
    primitive_assert(g.size[1] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:, :g.grid.shape[1]//2])

def right_half(g: Grid) -> Grid:
    primitive_assert(g.size[1] > 1, "Grid is too small to crop")
    new_position = (g.position[0], g.position[1] + g.grid.shape[1]//2 + g.grid.shape[1]%2)
    return g.newgrid(g.grid[:, -g.grid.shape[1]//2:], position=new_position)

def top_half(g: Grid) -> Grid:
    primitive_assert(g.size[0] > 1, "Grid is too small to crop")
    return g.newgrid(g.grid[:g.grid.shape[0]//2])
def repeatX(g: Grid) -> Grid:
    """
    Repeat the grid g horizontally, with no gaps
    """
    return Grid(np.tile(g.grid, (1, 2)), position=g.position)
def flipx(g: Grid) -> Grid:
    return g.newgrid(np.flip(g.grid, axis=0))
# Your custom primitive function
def ic_pickunique(l: List[Grid]) -> Grid:
    """
    Given a list of grids, return the one which has a unique colour unused by any other grid.
    If there are no such grids or more than one, terminate.
    """
    counts = np.zeros(10)
    uniques = [np.unique(g.grid) for g in l]
    for u in uniques:
        counts[u] += 1
    
    colour_mask = counts == 1
    ccount = np.sum(colour_mask)
    if not ccount:
        raise PrimitiveException("pickunique: no unique grids")
    
    for g, u in zip(l, uniques):
        if np.sum(colour_mask[u]) == ccount:
            return g
        
    raise PrimitiveException("pickunique: no unique grids (2)")
def countToXY(c: int, col: Colour) -> Grid:
    """
    Given a count (integer) and a colour, create a square grid of size c×c filled with the given colour.
    """
    return Grid(np.full((c, c), col, dtype=int))
# Primitive definitions
def gravity(g: Grid, dx=False, dy=False) -> Grid:
    assert dx or dy

    pieces = ic_splitall(g)

    # Sort pieces by gravity direction
    pieces = sorted(pieces, 
                    key=lambda g: -(g.position[0]*dy + g.position[1]*dx))
    
    # Start with empty grid
    newgrid = Grid(np.zeros(g.size))

    # Iterate over pieces
    for p in pieces:
        while True:
            # Move piece by gravity direction
            p.position = (p.position[0]+dy, p.position[1]+dx)

            # Check bounds
            if (p.position[0] < 0 or p.position[0]+p.size[0] > g.size[0] or 
                p.position[1] < 0 or p.position[1]+p.size[1] > g.size[1]):
                p.position = (p.position[0]-dy, p.position[1]-dx)
                break

            # Collision check
            slice = newgrid.grid[
                p.position[0]:p.position[0]+p.size[0],
                p.position[1]:p.position[1]+p.size[1]
            ]

            if slice[p.grid != 0].any():
                p.position = (p.position[0]-dy, p.position[1]-dx)
                break

        # Composite piece
        newgrid = overlay(newgrid, p)
    
    return newgrid

# Wrapper primitive: gravity to the right
def gravity_right(g: Grid) -> Grid:
    return gravity(g, dx=1)
struct8 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]], dtype=int)

def split8(g: Grid) -> List[Grid]:
    """
    Find all objects using 8-connected structuring element.
    Each colour is separated.
    """
    colours = np.unique(g.grid)
    ret = []
    for colour in colours:
        if colour:
            objects = ndimage.find_objects(ndimage.label(g.grid == colour, structure=struct8)[0])
            ret += [g.newgrid(g.grid[obj], offset=(obj[0].start, obj[1].start)) for obj in objects]
    return ret
def ic_makeborder(g: Grid) -> Grid:
    """
    Return a new grid which is the same as the input, but with a border of 1s around it.
    Only elements which are 0 in the original grid are set to 1 in the new grid.
    8-connected structuring element used to determine border positions.
    """

    binary_grid = g.grid > 0
    output_grid = np.zeros_like(g.grid)

    grown_binary_grid = binary_dilation(binary_grid, structure=np.ones((3, 3)))
    output_grid[grown_binary_grid & ~binary_grid] = 1

    return g.newgrid(output_grid)
def ic_filtercol(c: Colour, g: Grid) -> Grid:
    "Remove all colours except the selected colour"
    primitive_assert(c != 0, "filtercol with 0 has no effect")

    grid = np.copy(g.grid) # Do we really need to copy? old one thrown away anyway
    grid[grid != c] = 0
    return g.newgrid(grid)
def ic_invert(g: Grid) -> Grid:
    """
    Replaces all colours with zeros, and replaces zeros with the most common colour.
    """
    mode = np.argmax(np.bincount(g.grid.ravel())[1:]) + 1  # skip counting 0
    grid = np.zeros_like(g.grid)
    grid[g.grid == 0] = mode
    return g.newgrid(grid)
def logical_and(g: Grid, h: Grid) -> Grid:
    """
    Logical AND between two grids. Use the colour of the first argument.
    Logical OR is given by overlay.
    """
    primitive_assert(g.size == h.size, "logical_and: grids must be the same size")

    mask = np.logical_and(g.grid != 0, h.grid != 0)
    return g.newgrid(np.where(mask, g.grid, 0))

struct4 = np.array([[0,1,0],
                    [1,1,1],
                    [0,1,0]], dtype=int)

def fillobj(c: Colour, g: Grid) -> Grid:
    """
    Fill in any closed objects in the grid with a specified colour.
    Uses 4-connectedness to determine closed objects.
    """
    primitive_assert(c != 0, "fill with 0 has no effect")

    binhole = ndimage.binary_fill_holes(g.grid != 0, structure=struct4)
    newgrid = np.copy(g.grid)
    newgrid[binhole & (g.grid == 0)] = c

    return g.newgrid(newgrid)
def topcol(g: Grid) -> Colour:
    """
    Returns the most common colour in the grid, excluding black (0).
    Equivalent to majCol in icecuber.
    """
    return np.argmax(np.bincount(g.grid.ravel())[1:]) + 1
def rarestcol(g: Grid) -> Colour:
    """
    Returns the least common colour in the grid, excluding black.
    Colours with zero occurrences are ignored.
    """
    counts = np.bincount(g.grid.ravel())[1:]
    counts[counts == 0] = 9999
    return np.argmin(counts) + 1
def gravity_down(g: Grid) -> Grid:
    return gravity(g, dy=1)
def setcol(c: Colour, g: Grid) -> Grid:
    """
    Set all pixels in the grid to the specified colour.
    Originally named colShape in icecuber.
    """
    primitive_assert(c != 0, "setcol with 0 has no effect")

    grid = np.zeros_like(g.grid)
    grid[np.nonzero(g.grid)] = c
    return g.newgrid(grid)
def ic_embed(img: Grid, shape: Grid) -> Grid:
    """
    Embeds a grid into a larger shape defined by a second argument (zero-padded).
    If the image is larger than the shape, it is cropped.
    """
    ret = np.zeros_like(shape.grid)
    
    xoffset = shape.position[0] - img.position[0]
    yoffset = shape.position[1] - img.position[1]

    xsize = min(img.grid.shape[0], shape.grid.shape[0] - xoffset)
    ysize = min(img.grid.shape[1], shape.grid.shape[1] - yoffset)

    ret[xoffset:xoffset+xsize, yoffset:yoffset+ysize] = img.grid[:xsize, :ysize]
    return shape.newgrid(ret)

def rot270(g: Grid) -> Grid:
    """
    Rotate a grid by 270 degrees.
    """
    return g.newgrid(np.rot90(g.grid, k=3))
def mapSplit8(f: Callable[[Grid], Grid], g: Grid) -> Grid:
    """
    Split grid g into objects using 8-connectedness,
    apply function f to each object, and then reassemble.
    """
    pieces = split8(g)
    processed_pieces = [f(piece) for piece in pieces]
    return ic_composegrowing(processed_pieces)
from collections import Counter
def pickcommon(l: List[Grid]) -> Grid:
    """
    Given a list of grids, return the grid that appears most frequently (by exact match).
    """
    primitive_assert(len(l) > 0, "pickcommon: list is empty")
    hashes = [hash(g.grid.data.tobytes()) for g in l]
    most_common_hash = Counter(hashes).most_common(1)[0][0]
    return l[hashes.index(most_common_hash)]
def swapxy(g: Grid) -> Grid:
    return g.newgrid(g.grid.T)

def topcol(g: Grid) -> Colour:
    """
    Returns the most common colour, excluding black.
    majCol in icecuber.
    """
    return np.argmax(np.bincount(g.grid.ravel())[1:])+1
def setcol(c: Colour, g: Grid) -> Grid:
    """
    Set all pixels in the grid to the specified colour.
    This was named colShape in icecuber. 
    """
    primitive_assert(c != 0, "setcol with 0 has no effect")

    grid = np.zeros_like(g.grid)
    grid[np.nonzero(g.grid)] = c
    return g.newgrid(grid)
def get_bg(c: Colour, g: Grid) -> Grid:
    """
    Return a grid of all the background pixels in g, coloured c
    Essentially same as invert.
    """ 
    return Grid(np.where(g.grid == 0, c, 0))
def rarestcol(g: Grid) -> Colour:
    """
    Returns the least common colour, excluding black. 
    Excludes any colours with zero count.
    """
    counts = np.bincount(g.grid.ravel())[1:]
    counts[counts == 0] = 9999
    return np.argmin(counts)+1
def ic_fill(g: Grid) -> Grid:
    """
    Returns a grid with all closed objects filled in with the most common colour
    Note that like Icecuber, this also colours everything not connected to the border with the most common colour
    i.e. the result is a single colour
    """
    return setcol(topcol(g), fillobj(1, g))
def ic_center(g: Grid) -> Grid:
    # TODO: Figure out why this is useful
    w,h = g.size
    
    newsize = ((w + 1) % 2 + 1, (h + 1) % 2 + 1)
    newgrid = np.ones(newsize)
    newpos = (
        g.position[0] + (newsize[0] - w) / 2,
        g.position[1] + (newsize[1] - h) / 2
    )

    return Grid(newgrid, newpos)
def countToY(c: Count, col: Colour) -> Grid:
    """
    Create a vertical (1×c) grid filled with the given colour.
    """
    return Grid(np.full((1, c), col, dtype=int))

def countPixels(g: Grid) -> Count:
    """
    Count the number of non-zero pixels in the grid.
    """
    return np.count_nonzero(g.grid)
def ic_splitcols(g: Grid) -> List[Grid]:
    """
    Split a grid into multiple grids, each with a single colour.
    """
    ret = []
    for colour in np.unique(g.grid):
        if colour:
            ret.append(g.newgrid(g.grid == colour))
    return ret
def grid_split(g: Grid) -> GridList:
    # Get rows and columns that are filled with a single color
    row_colors = [row[0] for row in g.grid if np.all(row == row[0])]
    col_colors = [col[0] for col in g.grid.T if np.all(col == col[0])]

    colors = row_colors + col_colors
    primitive_assert(len(colors) > 0, "No uniform rows or columns found")

    # Find the most common such color
    color = np.argmax(np.bincount(colors))

    # Now split along rows and columns with this color
    horizontal_splits = []
    current = []
    for row in g.grid:
        if np.all(row == color):
            if current:
                horizontal_splits.append(np.stack(current))
                current = []
        else:
            current.append(row)
    if current:
        horizontal_splits.append(np.stack(current))

    # Convert each resulting piece to Grid
    result = []
    for h in horizontal_splits:
        if h.shape[0] > 0:
            result.append(Grid(h))

    return GridList(result)



def arc_assert(boolean, message=None):
    if not boolean: 
        # print('ValueError')
        raise ValueError(message)

def _get(l):
    def get(l, i):
        arc_assert(i >= 0 and i < len(l))
        return l[i]

    return lambda i: get(l, i)

def stack_no_crop(l: GridList) -> Grid:
    """
    Stack same-size grids with masking — first grids drawn underneath later ones.
    """
    primitive_assert(len(l) > 0, "Empty GridList in stack_no_crop")

    shape = l[0].grid.shape
    for g in l:
        primitive_assert(g.grid.shape == shape, "Mismatched grid shapes in stack_no_crop")

    stackedgrid = np.zeros(shape, dtype=int)
    for g in l:
        stackedgrid += g.grid * (stackedgrid == 0)

    return Grid(stackedgrid)

def overlay(g1: Grid, g2: Grid) -> Grid:
    """
    Mask overlay of g2 on top of g1.
    """
    primitive_assert(g1.grid.shape == g2.grid.shape, "Overlay shape mismatch")
    result = g1.grid.copy()
    result[g2.grid != 0] = g2.grid[g2.grid != 0]
    return Grid(result)
def _objects2(g: Grid) -> Callable[[bool], Callable[[bool], GridList]]:
    """
    Extract connected components (objects) from grid.
    Options:
    - connect_diagonals: whether to connect diagonally
    - separate_colors: whether to segment by color
    """

    def inner(connect_diagonals: bool):
        def inner2(separate_colors: bool):
            structure = ndimage.generate_binary_structure(2, 2 if connect_diagonals else 1)
            components = []

            if separate_colors:
                colors = np.unique(g.grid)
                colors = colors[colors != 0]
                for c in colors:
                    mask = (g.grid == c).astype(int)
                    labeled, num = ndimage.label(mask, structure)
                    for i in range(1, num + 1):
                        submask = (labeled == i).astype(int) * c
                        components.append(Grid(submask))
            else:
                mask = (g.grid != 0).astype(int)
                labeled, num = ndimage.label(mask, structure)
                for i in range(1, num + 1):
                    submask = (labeled == i).astype(int) * g.grid
                    components.append(Grid(submask))

            return GridList(components)

        return inner2
    return inner

def _objects(g: Grid) -> GridList:
    connect_diagonals = False
    separate_colors = True
    return _objects2(g)(connect_diagonals)(separate_colors)
def move_down(g: Grid) -> Grid:
    """
    Moves the first extracted object down by one row and overlays it back.
    """
    objects = _objects(g)
    primitive_assert(len(objects) > 0, "No objects found to move.")

    obj = objects[0]
    newg = Grid(np.copy(g.grid))
    newg.grid[obj.grid != 0] = 0

    moved_obj = Grid(np.roll(obj.grid, 1, axis=0), position=obj.position)

    return overlay(newg, moved_obj)
def draw_line(g: Grid, angle: int) -> Grid:
    """
    Draw a line from some starting condition in the grid in the given direction.
    Supported angles: 0 (right), 90 (up), 180 (left), 270 (down)
    """
    # For example, draw a line from top-left corner in given direction
    new_grid = np.copy(g.grid)
    h, w = new_grid.shape

    if angle == 0:  # Right
        new_grid[0, :] = 1
    elif angle == 90:  # Up
        new_grid[:, 0] = 1
    elif angle == 180:  # Left
        new_grid[-1, :] = 1
    elif angle == 270:  # Down
        new_grid[:, -1] = 1
    else:
        primitive_assert(False, f"Unsupported angle: {angle}")

    return Grid(new_grid)
def draw_line_slant_up(g: Grid) -> Grid:
    """
    Extracts first object and draws a 45-degree line from it.
    """
    objects = _objects(g)
    primitive_assert(len(objects) > 0, "No objects found in draw_line_slant_up.")
    obj = objects[0]
    return draw_line(g)(obj)(45)
def draw_line_slant_down(g: Grid) -> Grid:
    """
    Automatically extracts the first object from the grid and draws a 315-degree slant.
    """
    objects = _objects(g)
    primitive_assert(len(objects) > 0, "No objects found in draw_line_slant_down.")
    obj = objects[0]
    return draw_line(g)(obj)(315)

def _place_into_grid(objects):
    grid = np.zeros(objects[0].input_grid.shape, dtype=int)
    # print('grid: {}'.format(grid))
    for obj in objects:
        # print('obj: {}'.format(obj))
        # note: x, y, w, h should be flipped in reality. just go with it
        y, x = obj.position
        # print('x, y: {}'.format((x, y)))
        h, w = obj.grid.shape
        g_h, g_w = grid.shape
        # may need to crop the grid for it to fit
        # if negative, crop out the first parts
        o_x, o_y = max(0, -x), max(0, -y)
        # if negative, start at zero instead
        x, y = max(0, x), max(0, y)
        # this also affects the width/height
        w, h = w - o_x, h - o_y
        # if spills out sides, crop out the extra
        w, h = min(w, g_w - x), min(h, g_h - y)
        # print('x, y = {}, {}, o_x, o_y = {}, {}, w, h = {}, {}'.format(x, y,
            # o_x, o_y, w, h))

        grid[y:y+h, x:x+w] = obj.grid[o_y: o_y + h, o_x: o_x + w]

    return Grid(grid)






###########################################################
# ──────────────────────────────────────────────────────────────────────────
# ★★  EXTRA UTILITIES – copied from the full DSL or re-implemented ★★
#     Paste this block once into your second file.
# ──────────────────────────────────────────────────────────────────────────

# ── simple meta-info helpers ─────────────────────────────────────────────
def getpos(g: Grid) -> Tuple[int, int]:
    """Return the stored (row, col) offset of the grid."""
    return g.position


def getsize(g: Grid) -> Tuple[int, int]:
    """Return (height, width)."""
    return g.size


def ic_toorigin(g: Grid) -> Grid:
    """Return a copy of the grid reset to position (0,0)."""
    return Grid(np.copy(g.grid))


# ── interior / border / compress variants ────────────────────────────────
def ic_interior(g: Grid) -> Grid:
    """
    Fill holes with the majority colour, then keep only those new pixels
    (interior) coloured with that majority.
    """
    c = topcol(g)
    filled = fillobj(c, g).grid
    interior = np.where((filled != 0) & (g.grid == 0), c, 0)
    return g.newgrid(interior)


# trivial alias so nothing breaks
ic_interior2 = ic_interior


def ic_border(g: Grid) -> Grid:
    """Return only the 4-connected border of existing shapes (colour=1)."""
    inner = ndimage.binary_erosion(g.grid != 0, structure=np.ones((3, 3)))
    border_mask = (g.grid != 0) & (~inner)
    return g.newgrid(border_mask.astype(int))


def ic_compress(g: Grid) -> Grid:
    """
    Remove fully-zero rows and columns, preserving position.
    """
    keep_r = np.any(g.grid, axis=1)
    keep_c = np.any(g.grid, axis=0)
    cropped = g.grid[keep_r][:, keep_c]
    dr = np.argmax(keep_r)          # first kept row index
    dc = np.argmax(keep_c)          # first kept col index
    return Grid(cropped, (g.position[0] + dr, g.position[1] + dc))


# ── counting helpers ─────────────────────────────────────────────────────
def countColours(g: Grid) -> Count:
    """Number of distinct non-zero colours."""
    return np.count_nonzero(np.bincount(g.grid.ravel())[1:])


def countComponents(g: Grid) -> Count:
    """Number of 8-connected components, colour-blind."""
    lbl, n = ndimage.label(g.grid != 0, structure=np.ones((3, 3)))
    return int(n)


# ── count→grid constructors ──────────────────────────────────────────────
def countToX(c: Count, col: Colour) -> Grid:
    """Horizontal 1×c bar of colour."""
    return Grid(np.full((c, 1), col, dtype=int))


# ── hull helper ──────────────────────────────────────────────────────────
def colourHull(c: Colour, g: Grid) -> Grid:
    """Same size grid filled with colour c."""
    return Grid(np.full(g.size, c, dtype=int))


# ── smear / spread family ────────────────────────────────────────────────
def ic_smear_kernel(g: Grid, kernels: List[Tuple[int, int]]) -> Grid:
    """
    For each pixel in `g`, copy its colour to neighbours given by kernels
    (list of (dy, dx)).
    """
    out = np.copy(g.grid)
    for (dy, dx) in kernels:
        shifted = np.zeros_like(out)
        if dy >= 0:
            ys_src = slice(0, -dy or None)
            ys_dst = slice(dy, None)
        else:
            ys_src = slice(-dy, None)
            ys_dst = slice(0, dy)
        if dx >= 0:
            xs_src = slice(0, -dx or None)
            xs_dst = slice(dx, None)
        else:
            xs_src = slice(-dx, None)
            xs_dst = slice(0, dx)
        shifted[ys_dst, xs_dst] = g.grid[ys_src, xs_src]
        mask = (out == 0) & (shifted != 0)
        out[mask] = shifted[mask]
    return g.newgrid(out)


# build smear_* wrappers exactly like the first file
smear_functions = {}
for name, vecs in {
    "R":  [(0, 1)],  "L":  [(0, -1)], "D":  [(1, 0)],  "U":  [(-1, 0)],
    "RL": [(0, 1), (0, -1)],
    "DU": [(1, 0), (-1, 0)],
    "RLDU": [(0, 1), (0, -1), (1, 0), (-1, 0)],
    "X":  [(1, 1)],  "Y": [(-1, -1)], "Z": [(1, -1)], "W": [(-1, 1)],
    "XY": [(1, 1), (-1, -1)],
    "ZW": [(1, -1), (-1, 1)],
    "XYZW": [(1, 1), (-1, -1), (1, -1), (-1, 1)],
    "RLDUXYZW": [(0, 1), (0, -1), (1, 0), (-1, 0),
                 (1, 1), (-1, -1), (1, -1), (-1, 1)]
}.items():
    smear_functions[f"smear{name}"] = lambda g, V=vecs: ic_smear_kernel(g, V)


# expose them as plain callables in the global namespace
globals().update(smear_functions)


# ── additional border variants (simple aliases) ──────────────────────────
def ic_makeborder2(g: Grid) -> Grid:
    return ic_makeborder(g)


def ic_makeborder2_maj(g: Grid) -> Grid:
    c = topcol(g)
    b = ic_makeborder(g).grid
    b[b == 1] = c
    return g.newgrid(b)


# ── connect variant ──────────────────────────────────────────────────────
def ic_connectXY(g: Grid) -> Grid:
    return ic_connect_kernel(g, True, True)


# ── spreading helpers (BFS) ──────────────────────────────────────────────
def ic_spread(g: Grid) -> Grid:
    q = list(zip(*np.nonzero(g.grid)))
    out = np.copy(g.grid)
    done = out != 0
    while q:
        r, c = q.pop(0)
        col = out[r, c]
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and not done[nr, nc]:
                out[nr, nc] = col
                done[nr, nc] = True
                q.append((nr, nc))
    return g.newgrid(out)


def ic_spread_minor(g: Grid) -> Grid:
    maj = topcol(g)
    mask = (g.grid != 0) & (g.grid != maj)
    q = list(zip(*np.nonzero(mask)))
    out = np.copy(g.grid)
    done = mask
    while q:
        r, c = q.pop(0)
        col = out[r, c]
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and not done[nr, nc]:
                out[nr, nc] = col
                done[nr, nc] = True
                q.append((nr, nc))
    return g.newgrid(out)


# ── more cropping helpers ────────────────────────────────────────────────
def bottom_half(g: Grid) -> Grid:
    primitive_assert(g.size[0] > 1, "Grid too small to crop")
    return g.newgrid(g.grid[-g.grid.shape[0]//2:],
                     position=(g.position[0] + g.grid.shape[0]//2
                               + g.grid.shape[0] % 2, g.position[1]))


# ── embedding helper stub (wrap) ─────────────────────────────────────────
def ic_wrap(line: Grid, area: Grid) -> Grid:
    """Simple stub: overlay line on area respecting positions."""
    return overlay(area, line)


# ── column / row split helpers ───────────────────────────────────────────
def ic_splitcolumns(g: Grid) -> List[Grid]:
    return [g.newgrid(g.grid[:, i:i+1], position=(g.position[0], g.position[1]+i))
            for i in range(g.grid.shape[1])]


def ic_splitrows(g: Grid) -> List[Grid]:
    return [g.newgrid(g.grid[i:i+1], position=(g.position[0]+i, g.position[1]))
            for i in range(g.grid.shape[0])]


# ── list utilities ───────────────────────────────────────────────────────
def mklist(g: Grid, h: Grid) -> List[Grid]:
    return [g, h]


def lcons(g: Grid, lst: List[Grid]) -> List[Grid]:
    return [g] + lst


# ── repetition ───────────────────────────────────────────────────────────
def repeatY(g: Grid) -> Grid:
    return Grid(np.tile(g.grid, (2, 1)), position=g.position)


# ── additional gravity directions ────────────────────────────────────────
def gravity_up(g: Grid) -> Grid:
    return gravity(g, dy=-1)


def gravity_left(g: Grid) -> Grid:
    return gravity(g, dx=-1)


# ── pick-max family (simple versions) ────────────────────────────────────
def _area(gr: Grid) -> int:
    return gr.grid.shape[0] * gr.grid.shape[1]

def pickmax_count(lst: List[Grid]) -> Grid:
    primitive_assert(lst, "empty list")
    return max(lst, key=lambda g: g.count())

def pickmax_neg_count(lst: List[Grid]) -> Grid:
    primitive_assert(lst, "empty list")
    return min(lst, key=lambda g: g.count())

def pickmax_size(lst: List[Grid]) -> Grid:
    primitive_assert(lst, "empty list")
    return max(lst, key=_area)

def pickmax_neg_size(lst: List[Grid]) -> Grid:
    primitive_assert(lst, "empty list")
    return min(lst, key=_area)

def pickmax_cols(lst: List[Grid]) -> Grid:
    primitive_assert(lst, "empty list")
    return max(lst, key=lambda g: len(np.unique(g.grid)))

def pickmax_x_pos(lst: List[Grid]) -> Grid:
    return max(lst, key=lambda g: g.position[0])

def pickmax_x_neg(lst: List[Grid]) -> Grid:
    return min(lst, key=lambda g: g.position[0])

def pickmax_y_pos(lst: List[Grid]) -> Grid:
    return max(lst, key=lambda g: g.position[1])

def pickmax_y_neg(lst: List[Grid]) -> Grid:
    return min(lst, key=lambda g: g.position[1])

def pickmax_interior_count(lst: List[Grid]) -> Grid:
    return max(lst, key=lambda g: np.count_nonzero(ic_interior(g).grid))

def pickmax_neg_interior_count(lst: List[Grid]) -> Grid:
    return min(lst, key=lambda g: np.count_nonzero(ic_interior(g).grid))

# register the pickmax helpers for easy use
globals().update({
    "pickmax_count": pickmax_count,
    "pickmax_neg_count": pickmax_neg_count,
    "pickmax_size": pickmax_size,
    "pickmax_neg_size": pickmax_neg_size,
    "pickmax_cols": pickmax_cols,
    "pickmax_x_pos": pickmax_x_pos,
    "pickmax_x_neg": pickmax_x_neg,
    "pickmax_y_pos": pickmax_y_pos,
    "pickmax_y_neg": pickmax_y_neg,
    "pickmax_interior_count": pickmax_interior_count,
    "pickmax_neg_interior_count": pickmax_neg_interior_count,
})

# ───────────────── END OF PATCH BLOCK ────────────────────────────────────
