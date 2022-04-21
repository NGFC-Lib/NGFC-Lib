import ipywidgets as widgets 
from ipywidgets import AppLayout, Box, Button, GridspecLayout, Layout 
from IPython.display import display

def select_units():

    unit_list = [
        'Heater',
        'Cooler',
        'Heat Exchanger',
        'Reactor',
        'Mixer',
        'Flash',
        'Splitter',
        'Compressor',
        'Expander',
        'Turbine'
        ]

    a = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[0],
        disabled=False,
        style = {'description_width': 'initial'})

    b = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[1],
        disabled=False,
        style = {'description_width': 'initial'})

    c = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[2],
        disabled=False,
        style = {'description_width': 'initial'})

    d = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[3],
        disabled=False,
        style = {'description_width': 'initial'})

    e = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[4],
        disabled=False,
        style = {'description_width': 'initial'})

    f = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[5],
        disabled=False,
        style = {'description_width': 'initial'})

    g = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[6],
        disabled=False,
        style = {'description_width': 'initial'})

    h = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[7],
        disabled=False,
        style = {'description_width': 'initial'})

    i = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[8],
        disabled=False,
        style = {'description_width': 'initial'})

    j = widgets.Dropdown(
        options=['0', '1', '2'],
        value='1',
        description=unit_list[9],
        disabled=False,
        style = {'description_width': 'initial'})


    def build_unit_lists(a,b,c,d,e,f,g,h,i, j):
        num_heaters = int(a)
        num_coolers = int(b)
        num_heatex = int(c)
        num_reactors = int(d)
        num_mixers = int(e)
        num_flash = int(f)
        num_splitters = int(g)
        num_compressors = int(h)
        num_expanders = int(i)
        num_turbines = int(j)
        unit_nums = [num_heaters, num_coolers, num_heatex, num_reactors, num_mixers, num_flash, num_splitters, 
                     num_compressors, num_expanders, num_turbines]

        count = 0
        list_unit_base = ['heater_1', 'heater_2','cooler_1', 'cooler_2', 'heatex_1', 'heatex_2', 'StReactor_1',
                        'StReactor_2', 'mixer2to1_1','mixer2to1_2', 'flash_1', 'flash_2', 'splitter1to2_1', \
                        'splitter1to2_2','compressor_1', 'compressor_2', 'expander_1', 'expander_2', 'turbine_1', 'turbine_2']

        list_unit_all = []
        list_inlet_all = []
        list_outlet_all = []

        for i in range(len(unit_nums)):
            if unit_nums[i]==0:
                count += 2
            elif unit_nums[i]==1:
                list_unit_all.append(list_unit_base[count])
                count += 2
            else:
                list_unit_all.append(list_unit_base[count])
                list_unit_all.append(list_unit_base[count+1])
                count += 2 
        
        
        for i in range(len(list_unit_all)):
            if (list_unit_all[i]=='mixer2to1_1') or (list_unit_all[i]=='mixerer2to1_2'):
                list_inlet_all.append(list_unit_all[i] + '.inlet_1')
                list_inlet_all.append(list_unit_all[i] + '.inlet_2')

            else:
                list_inlet_all.append(list_unit_all[i] + '.inlet')

        for i in range(len(list_unit_all)):
            if (list_unit_all[i]=='splitter1to2_1') or (list_unit_all[i]=='splitter1to2_2'):
                list_outlet_all.append(list_unit_all[i] + '.outlet_1')
                list_outlet_all.append(list_unit_all[i] + '.outlet_2')

            else:
                list_outlet_all.append(list_unit_all[i] + '.outlet')


        list_unit_all.insert(0, 'inlet_feed')
        list_unit_all.insert(1, 'outlet_product')
        list_unit_all.insert(2, 'outlet_exhaust')
        list_inlet_all.insert(0, 'outlet_product.inlet')
        list_inlet_all.insert(1, 'outlet_exhaust.inlet')
        list_outlet_all.insert(0, 'inlet_feed.outlet')
        #display(list_unit_all)
        return list_unit_all, list_inlet_all, list_outlet_all       

    w = widgets.interactive(build_unit_lists, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i, j=j)

    return w 

def RL_options():

    k = widgets.IntSlider(
    value=300000,
    min=0,
    max=500000,
    step=50000,
    description='Episodes:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d')

    l = widgets.FloatSlider(
    value=0.8,
    min=0,
    max=1,
    step=0.1,
    description='Greedy:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',)

    m = widgets.FloatSlider(
    value=5e-6,
    min=0,
    max=5e-3,
    step=5e-7,
    description='e-greedy increment:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format= '.6f',)

    n = widgets.FloatSlider(
    value=0.01,
    min=0,
    max=0.1,
    step=0.01,
    description='Learning rate:',
    style = {'description_width': 'initial'},
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format= '.2f',)

    o = widgets.Dropdown(
    options=['True', 'False'],
    value='False',
    description= 'GNN Enabled',
    disabled=False,
    style = {'description_width': 'initial'})

    def store_values(k, l, m, n, o):

        episodes = k
        greedy = l
        greedy_inc = m
        learning = n
        GNN = o

        return k, l, m, n, o

    r = widgets.interactive(store_values, k=k, l=l, m=m, n=n, o=o)


    return r


