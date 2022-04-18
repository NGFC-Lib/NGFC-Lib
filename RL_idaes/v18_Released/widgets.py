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
        'Expander'
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

    # view = widgets.HBox([a, b, c, d, e, f, g, h, i])

    # header_button = widgets.HTML(
    #     value='<{size}>Select Max. Number of Units</{size}>'.format(size='h2'))


    # w = AppLayout(header=header_button,
    #       left_sidebar=None,
    #       center=grid,
    #       right_sidebar=None,
    #       footer=None)


    def build_unit_lists(a,b,c,d,e,f,g,h,i):
        num_heaters = int(a)
        num_coolers = int(b)
        num_heatex = int(c)
        num_reactors = int(d)
        num_mixers = int(e)
        num_flash = int(f)
        num_splitters = int(g)
        num_compressors = int(h)
        num_expanders = int(i)
        unit_nums = [num_heaters, num_coolers, num_heatex, num_reactors, num_mixers, num_flash, num_splitters, 
                     num_compressors, num_expanders]

        count = 0
        list_unit_base = ['heater_1', 'heater_2','cooler_1', 'cooler_2', 'heatex_1', 'heatex_2', 'StReactor_1',
                        'StReactor_2', 'mixer2to1_1','mixer2to1_2', 'flash_1', 'flash_2', 'splitter1to2_1', \
                        'splitter1to2_2','compressor_1', 'compressor_2', 'expander_1', 'expander_2']
        list_unit_all = []
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
        
        list_unit_all.insert(0, 'inlet_feed')
        list_unit_all.insert(1, 'outlet_product')
        list_unit_all.insert(2, 'outlet_exhaust')
        #display(list_unit_all)
        return list_unit_all       

    w = widgets.interactive(build_unit_lists, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i)

    return w 



