def deEmojify(text):
    #remove emoji (FIXME: doesnt remove all of them.)
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean(text):
    #remove URLs
    pdf_regex='http[\S]+pdf[\S]*'
    regex = r'http\S+'
    text = re.sub(regex, 'urlpostedtopdf', text)
    text = re.sub(regex, 'urlpostedtosomething', text)
    #TODO: remove emails

    #replace commas and semicolons with spaces.
    text = re.sub('[;,&\+]+', ' ', text)
    #remove hyphens
    text = re.sub('[-]+', ' ', text)
    return text