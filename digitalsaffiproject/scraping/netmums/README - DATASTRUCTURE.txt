search scrape:

{'http': [
			{'link':str,'title':str,'blurb':str},
			{'link':str,'title':str,'blurb':str},
			{'link':str,'title':str,'blurb':str},
			...
		 ],

'http:':{...},
...
}

reformatted search scrape:
When doing the second step of netmums scrape, getting full thread data from URLs, 
the organization is modified: instead of search query being the dict key, link url becomes the dict key.
This is to remove duplicate URLs


{'url_string':		#title attrib gets updated
		{'title':str, 'query':(set of queryURLS where the link appeared),
		 'posts':[
		 				{'author':str, 'likes':str, 'body':str, 'quotes_y':[{author:str,text:str}], 'quotes_w:'[str], 'urls':[str]}

		 		 ]
		}
}
	
#NETMUMS DATA
#we have a list of post dict, where each post_dict is

post_dict = {'username': get_post_username(post_soup),
				'likes': get_post_likes(post_soup),
				'date': get_post_date(post_soup),
				'quotes_y': quotes_y,
				'quotes_w': quotes_w,
				'body':body_text,
				'body_urls':body_urls}

##FACEBOOK DATA (examples)

pprint(fb_search[list(fb_search.keys())[1]])
>
{'data': [{'available': None,
           'comments': None,
           'comments_full': None,
           'factcheck': None,
           'images_description': None,
           'images_lowquality': None,
           'images_lowquality_description': None,
           'likes': None,
           'link': None,
           'post_id': None,
           'post_text': None,
           'post_url': 'https://m.facebook.com/AhSengDurian/posts/1576229315761364',
           'reactors': None,
           'shared_text': None,
           'shared_time': None,
           'shared_user_id': None,
           'shared_username': None,
           'shares': None,
           'text': None,
           'time': None,
           'user_id': None,
           'user_url': None,
           'username': None,
           'video_watches': None,
           'w3_fb_url': None}],
 'query': {'site:en-gb.facebook.com/*/posts/ OR site:www.facebook.com/*/posts/ '
           '"biphenyls" baby formula OR bottle-fed OR veggie OR vegetable OR '
           'baby food OR veg puree OR fruit puree OR fruit food OR applesauce '
           'OR cereal OR  porridge OR oats OR oatmeal OR jar food OR baby food '
           'OR  premade OR puree OR  pur??e OR yoghurt OR pudding'}}



{'data': [{'available': True,
           'comments': 1,
           'comments_full': [{'comment_id': '152235283140873',
                              'comment_text': 'Indian quote available sir?\n'
                                              'My WhatsApp number\n'
                                              '+91 9597112633',
                              'comment_time': None,
                              'commenter_meta': None,
                              'commenter_name': 'Ranjith Ranjith',
                              'commenter_url': 'https://facebook.com/profile.php?id=100052913081363&fref=nf&rc=p&refid=52&__tn__=R'}],
           'factcheck': None,
           'images_description': [],
           'images_lowquality': [],
           'images_lowquality_description': [],
           'likes': 0,
           'link': None,
           'post_id': '152119279819140',
           'post_text': 'Two vacancies available\n'
                        '1. Sales Assistant in Minimart\n'
                        '1. Factory Production Assistant\n'
                        '\n'
                        'Salary: $1,200 - $1,400+OT\n'
                        '\n'
                        'Location: Woodlands\n'
                        '\n'
                        'Interested, please email to gcchua@gmail.com\n'
                        '\n'
                        'Thank you.',
           'post_url': 'https://facebook.com/story.php?story_fbid=152119279819140&id=115916103439458',
           'reactors': None,
           'shared_text': 'HENG LAI HENG TRADING ENTERPRISE\nSales Assistant',
           'shared_time': None,
           'shared_user_id': None,
           'shared_username': None,
           'shares': 0,
           'text': 'Two vacancies available\n'
                   '1. Sales Assistant in Minimart\n'
                   '1. Factory Production Assistant\n'
                   '\n'
                   'Salary: $1,200 - $1,400+OT\n'
                   '\n'
                   'Location: Woodlands\n'
                   '\n'
                   'Interested, please email to gcchua@gmail.com\n'
                   '\n'
                   'Thank you.\n'
                   '\n'
                   'HENG LAI HENG TRADING ENTERPRISE\n'
                   'Sales Assistant',
           'time': datetime.datetime(2020, 7, 12, 10, 18),
           'user_id': '115916103439458',
           'user_url': 'https://facebook.com/HengLaiHengTrading/?__tn__=C-R',
           'username': 'Heng Lai Heng Trading Enterprise',
           'video_watches': None,
           'w3_fb_url': None}],
 'query': {'site:en-gb.facebook.com/*/posts/ OR site:www.facebook.com/*/posts/ '
           '"biphenyls" baby formula OR bottle-fed OR veggie OR vegetable OR '
           'baby food OR veg puree OR fruit puree OR fruit food OR applesauce '
           'OR cereal OR  porridge OR oats OR oatmeal OR jar food OR baby food '
           'OR  premade OR puree OR  pur??e OR yoghurt OR pudding'}}
