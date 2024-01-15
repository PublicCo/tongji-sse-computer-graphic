import turtle

def draw_point(x,y):
    turtle.speed(1000)
    turtle.penup()
    turtle.goto(x,y)
    turtle.tracer(0)
    turtle.dot(2,'black')

def draw_ellipse_with_turtle(rx, ry, xc, yc):

    x = 0
    y = ry
    # 初值
    p1 = ry**2 - rx**2 * ry + 0.25 * rx**2

    # 斜率
    dx = 2 * ry**2 * x
    dy = 2 * rx**2 * y
    while dy>dx :
        turtle.tracer(1)
        draw_point(xc + x, yc + y)
        draw_point(xc - x, yc + y)
        draw_point(xc + x, yc - y)
        draw_point(xc - x, yc - y)
        x += 1
        if p1 < 0:
            dx = 2 * ry**2 * x
            p1 = p1 + dx + ry**2
        else:
            y -= 1
            dx = 2 * ry**2 * x
            dy = 2 * rx**2 * y
            p1 = p1 + dx - dy + ry**2
        turtle.tracer(0)

    # 另一部分
    p2 = ry**2 * (x + 0.5)**2 + rx**2 * (y - 1)**2 - rx**2 * ry**2
    while y > 0:
        turtle.tracer(1)
        draw_point(xc + x, yc + y)
        draw_point(xc - x, yc + y)
        draw_point(xc + x, yc - y)
        draw_point(xc - x, yc - y)
        # 中点在外部
        y -= 1
        if p2 > 0:
            dy = 2 * rx**2 * y
            p2 = p2 - dy + rx**2
        else:
            # 中点在内部
            x += 1
            dx = 2 * ry**2 * x
            dy = 2 * rx**2 * y
            p2 = p2 + dx - dy + rx**2
        turtle.tracer(0)
    draw_point(x-rx,0)
    draw_point(x+rx,0)
    turtle.done()

# 分别是横轴半径、纵轴半径、中心x坐标、中心y坐标
draw_ellipse_with_turtle(200, 100, 0, 0)
