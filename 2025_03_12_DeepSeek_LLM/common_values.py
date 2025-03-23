CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 600

# for prompt engineering
PROMPT_PREFIX = "Represent below as a Python list.\n\n"
PROMPT_SUFFIX = """in the following format.

At this time, each node is represented in the format of Python list "[node No.,
X position (px), Y position (px), shape (rectangle, round rectangle or circle),
width (px), height (px), connection line shape (solid or dashed), background color,
connection line color, list of node No. s of other nodes pointed to by the connection line]".

At this time, the color is represented in the format of tuple (R, G, B), between 0 and 255, and
X position range is 0-1000 and Y position range is 0-600.

It is important to draw a representation of high readability."""