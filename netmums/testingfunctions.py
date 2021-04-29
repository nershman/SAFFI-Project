# -*- coding: utf-8 -*-
# @Author: sma
# @Date:   2021-04-19 15:01:45
# @Last Modified by:   sma
# @Last Modified time: 2021-04-29 09:03:53

#test combo list
from itertools import repeat, permutations
l1 = ["a", "b"]
l2 = ["c", "d"]

temp = list(product(l1,l2,l2))
make_combo_list(l1,l2)
make_combo_list(l1,["c","d",2])
make_combo_list(l1,2)
make_combo_list(l1,"oxo")


#second beautifull soup..

import re

html = requests.get('https://www.netmums.com/coffeehouse/becoming-mum-ttc-64/early-pregnancy-signs-symptoms-537/797571-8-weeks-pregnant-got-shingles.html').text
soup = BeautifulSoup(html, 'html.parser')
temp_posts = soup.find_all('div', {'class': re.compile('DesktopPostCardstyle__PostContainer-')})

temp_post = temp_posts[0]


#get the comment text (its inside of a nested div but still finds it :)

temp_post.find('div', {'class': re.compile('DesktopPostCardstyle__PostContent-')}).text

#get the likes count
temp_post.find('div', {'class': re.compile('LikeCounterstyle__Container-')}).text

#get the post date


#get the username
temp_post.find('div', {'class': re.compile('__UserPseudo-')}).text
#DesktopPostCardstyle__PostContainer-t45zrm-5 hwvlv