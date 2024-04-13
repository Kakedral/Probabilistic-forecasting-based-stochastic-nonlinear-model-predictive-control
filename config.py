"""
Config file
Copyright (c) 2021, Kiet Tuan Hoang
Last edited: 27.06.2021
"""

""" 
Color scheme map for plotting
NTNU colors       : https://innsida.ntnu.no/wiki/-/wiki/Norsk/Farger+i+grafisk+profil
RWTH aachen colors: https://www9.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaadpbhq

"""
config_color_map = {
# Palette
'yellow_pastel' : (255/255,250/255,129/255),
'purple_pastel' : (193/255,179/255,215/255),
'skin_color' : (252/255,169/255,133/255),
'green_pastel': (181/255,225/255,174/255),
'red_pastel' : (255/255,105/255,97/255),


# Blue colors (NTNU)
'NTNU_blue'            : (0/255 , 80/255, 158/255),
'NTNU_blue_light'      : (37/255, 58/255, 85/255),
'NTNU_blue_dark'       : (62/255, 98/255, 138/255),
'NTNU_blue_very_light' : (207/255, 218/255, 241/255),

# Green colors (NTNU)
'NTNU_green_dark'  : (73/255 , 49/255 , 43/255),
'NTNU_green_light' : (124/255, 137/255, 52/255),

'Good_green'       : (0.13,0.545,0.13),


# Purple colors (NTNU)
'NTNU_green_dark'  : (73/255 , 49/255 , 43/255),
'NTNU_green_light' : (124/255, 137/255, 52/255),
'NTNU_orange' : (255/255, 172/255, 103/255),

# Yellow color (RWTH Aachen)
'RWTH_yellow_100' : (244/255,172/255,103),

# Red colors (RWTH Aachen)
'RWTH_red_100' : (0.800000, 0.027451, 0.117647),
'RWTH_red_75'  : (0.847059, 0.360784, 0.254902),
'RWTH_red_50'  : (0.901961, 0.588235, 0.474510),
'RWTH_red_25'  : (0.952941, 0.803922, 0.733333)
}

''' Auxillarily helper functions'''
def config_sublist_generator(list, n):
    """ helper function for generating sublist with n length from a array list (credits to Even Masdal)
                # Arguments:
                    list       : the array to be used for computing sublists
                    n          : an integer which describes the length of the resulting list

                # Output:
                    sublists   : an matrix with all of the sublists
    """
    sublists = list.copy();
    sublists.extend([list[-1]] * (n - 1));
    for i in range(len(sublists) - n + 1):
        yield sublists[i:i + n];


# cf = foreground color, cb = background color
def mix_colors(cf, cb):
    a = cb[-1] + cf[-1] - cb[-1] * cf[-1] # fixed alpha calculation
    r = (cf[0] * cf[-1] + cb[0] * cb[-1] * (1 - cf[-1])) / a
    g = (cf[1] * cf[-1] + cb[1] * cb[-1] * (1 - cf[-1])) / a
    b = (cf[2] * cf[-1] + cb[2] * cb[-1] * (1 - cf[-1])) / a
    return [r,g,b,a]

