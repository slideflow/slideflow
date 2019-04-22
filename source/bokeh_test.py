from bokeh.models import Button
from bokeh.io import curdoc

bt = Button(label='Click me')

def change_click():
    print('I was clicked')

bt.on_click(change_click)

curdoc().add_root(bt)