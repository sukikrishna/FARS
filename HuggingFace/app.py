def df_cleaning(df, col):
    # Drop rows with na values
    df.dropna(inplace = True)
    
    new_col_name = 'new_' + col
    
    df[new_col_name] = df[col].copy() 
    
    # Remove unwanted formatting characters
    format_strs = dict.fromkeys(['<br /><br />', '&#34', 'br', '&quot', '<br />'], ' ')
    
    for key in format_strs:
        df[new_col_name] = df[new_col_name].apply(lambda review: review.replace(key, format_strs[key]))
    # removing quotes produces smthg like this --> 'The product has great ;sound; --> we must remove punctuation

    
    # Case normalization (lower case)
    df[new_col_name] = df[new_col_name].str.lower()
    
    remove_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": "",
                   "(": "", ")":""}
    for key, val in remove_dict.items():
        df[new_col_name] = df[new_col_name].apply(
            lambda x: x.replace(key, val))
        
    # Remove stopwords
    stop_lst = stopwords.words('english')
    #stop_lst += (["can't", "i'm" "i'd", "i've", "i'll", "that's", "there's", "they're"])
    # ****Do we not have to take stopwords out BEFORE removing punctuation? Otherwise words with punct like “cant” remains there
    df[new_col_name] = df[new_col_name].apply(lambda text_body: " ".join([word for word in text_body.split() if word not in (stop_lst)]))
    
    # Removing Unicode Chars (punctuation, URL, @)
    df[new_col_name] = df[new_col_name].apply(lambda rev: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", rev))
    
    # Lemmatization
    word_lemmatizer = WordNetLemmatizer()
    df[new_col_name] = df[new_col_name].apply(lambda txt: " ".join([(word_lemmatizer.lemmatize(word)) for word in txt.split()]))
    
    return df