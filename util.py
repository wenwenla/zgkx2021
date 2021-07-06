from PIL import ImageDraw


def _draw_with_color_and_width(image, position, transform, color, width):
    p = transform @ position
    draw = ImageDraw.Draw(image)
    draw.ellipse(((p[0] - width / 2, p[1] - width / 2), (p[0] + width / 2, p[1] + width / 2)), fill=color)


def draw_target_area(image, position, transform, r):
    width = r * 2 * transform[0, 0]
    _draw_with_color_and_width(image, position, transform, 'yellow', width)


def draw_uav(image, position, transform):
    width = 3
    _draw_with_color_and_width(image, position, transform, 'red', width)


def draw_obstacle(image, position, transform, collision, r):
    width = r * 2 * transform[0, 0]
    _draw_with_color_and_width(image, position, transform, 'grey' if not collision else 'red', width)


def draw_tr(image, tr, transform):
    width = 1
    for u in tr:
        for p in u:
            _draw_with_color_and_width(image, p, transform, 'black', width)


def draw_grid(image, row, col, cell_size):
    height = row * cell_size
    width = col * cell_size
    draw = ImageDraw.Draw(image)
    for r in range(row + 1):
        draw.line(((0, r * cell_size), (width, r * cell_size)), fill='black')
    for c in range(col + 1):
        draw.line(((c * cell_size, 0), (c * cell_size, height)), fill='black')


def draw_rect(image, row, col, cell_size, color):
    pos_y = row * cell_size
    pos_x = col * cell_size
    draw = ImageDraw.Draw(image)
    draw.rectangle((pos_x, pos_y, pos_x + cell_size, pos_y + cell_size), color)


def draw_ball(image, row, col, cell_size, color='red'):
    pos_y = row * cell_size + cell_size / 2
    pos_x = col * cell_size + cell_size / 2
    draw = ImageDraw.Draw(image)
    draw.ellipse(((pos_x - cell_size / 2, pos_y - cell_size / 2), (pos_x + cell_size / 2, pos_y + cell_size / 2)), fill=color, outline=color)