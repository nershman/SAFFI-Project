# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-05-15 17:25:44
# @Last Modified by:   sma
# @Last Modified time: 2021-05-15 17:28:51


import sys
sys.path.append('../') #make parent path visible so we can import modules from other folders.

from netmums import scrapehelpers as scr #requires making parent path visible


products = 'baby formula OR bottle-fed OR veggie OR vegetable OR baby food OR veg puree OR fruit puree OR fruit food OR applesauce OR cereal OR  porridge OR oats OR oatmeal OR jar food OR baby food OR  premade OR puree OR  purée OR yoghurt OR pudding'
hazards = scr.get_concerns()

hazards = ["\""+i+"\"" for i in hazards]
hazards.append('\"recall\"')
hazards.append('\"product recall\"')

search_queries = scr.make_combo_list(['site:en-gb.facebook.com/*/posts/ OR site:www.facebook.com/*/posts/'],
									  hazards, [products])