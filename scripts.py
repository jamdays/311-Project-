import pandas as pd
import numpy as np
import re
import json
__all__ = ['clean_data']

def clean_data(dataframe: pd.DataFrame, clean_type="normal") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes in as input a dataframe and returns two dataframes:
    X: a dataframe containing the cleaned features
    T: a dataframe containing the cleaned target variable
    """
    # get everything except label
    selected_features = ["Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
                         "Q2: How many ingredients would you expect this food item to contain?",
                         "Q3: In what setting would you expect this food to be served? Please check all that apply",
                         "Q4: How much would you expect to pay for one serving of this food item?",
                         "Q5: What movie do you think of when thinking of this food item?",
                         "Q6: What drink would you pair with this food item?",
                         "Q7: When you think about this food item, who does it remind you of?",
                         "Q8: How much hot sauce would you add to this food item?"]
    df = dataframe[selected_features].copy()

    # Add indicator variables to categorical variable people
    people_col = "Q7: When you think about this food item, who does it remind you of?"

    # Get rid of NaNs
    df[people_col] = df[people_col].fillna('')

    # Hardcoded categories
    unique_categories = {"Friends", "Teachers", "Strangers", "Parents", "Siblings"}

    # Create new indicators for each unique category
    for category in unique_categories:
        df[category] = df[people_col].str.contains(category, regex=False).astype(int)

    # Drop the original people column
    df.drop(columns=[people_col], inplace=True)
    
    movies = "Q5: What movie do you think of when thinking of this food item?"

    # Turn movies into BoW indicators if we're including them
    if clean_type=="normal":
        global vocab
        movie_vocab = json.load(open("movie_vocab.json", "r"))
        def clean_movie(movie):
            movie = str(movie)
            for char in movie:
                if not char.isalpha() and not char.isspace():
                    movie = movie.replace(char, '')
            movie = movie.lower()
            movie_as_list = movie.split()
            return movie_as_list

        df[movies] = df[movies].apply(clean_movie)

        new_column_names = [f"movie_{word}" for word in movie_vocab.keys()]

        bag_of_word_indicators = pd.DataFrame([
            [1 if word in words else 0 for word in movie_vocab]
            for words in df[movies]
        ], columns=new_column_names)


        df = pd.concat([df, bag_of_word_indicators], axis=1)
    df.drop(columns=[movies], inplace=True)

    # Clean price

    def get_price(frame):
        cols = frame["Q4: How much would you expect to pay for one serving of this food item?"]
        cols = [max(find_digits(str(x))) if len(find_digits(str(x))) != 0 else words_to_digits(str(x)) for x in cols]
        return cols[0: len(cols)]

    def find_digits(string):
        ##careful for index here change it later
        return [x for x in range(51) if (string.find(str(x)) != -1 and 
                                        (string.find(str(x)) == 0 or string[(string.find(str(x)) -1)] != "." or
                                        string.find(str(x)) == 1 or string[(string.find(str(x)) -2)] != "."))]

    def words_to_digits(string):
        tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
        }
        ones = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9
        }
        teens = {
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
            "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
        }
        for key in tens.keys():
            if string.lower().find(key) != -1:
                return tens[key]
        for key in teens.keys():
            if string.lower().find(key) != -1:
                return teens[key]
        for key in ones.keys():
            if string.lower().find(key) != -1:
                return ones[key]
        return -1

    df["price"] = get_price(df)
    df.drop(columns=["Q4: How much would you expect to pay for one serving of this food item?"], inplace=True)

    def is_digit(char: chr) -> bool:
        return char in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


    def is_number(chars: str) -> bool:
        x = True
        if chars in ["zero", "one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "twenty", "fifty",
                    "hundred"]:
            return x
        for s in chars:
            if not is_digit(s):
                x = False
        return x


    def string_num_to_int(chars: str) -> int:
        """
        precondition: chars in ["zero", "one", "two", "three", "four", "five",
                                "six", "seven", "eight", "nine", "ten", "eleven",
                                "twelve", "twenty", "fifty", "hundred"]
        """
        if chars == "zero":
            return 0
        if chars == "one":
            return 1
        if chars == "two":
            return 2
        if chars == "three":
            return 3
        if chars == "four":
            return 4
        if chars == "five":
            return 5
        if chars == "six":
            return 6
        if chars == "seven":
            return 7
        if chars == "eight":
            return 8
        if chars == "nine":
            return 9
        if chars == "ten":
            return 10
        if chars == "eleven":
            return 11
        if chars == "twelve":
            return 12
        if chars == "twenty":
            return 20
        if chars == "fifty":
            return 50
        if chars == "hundred":
            return 100
        return 0


    def clean_num_ingredients(num_ingredients) -> int:
        str_num = str(num_ingredients)
        clean_nums_present = []
        nums_present = [m.group() for m in re.finditer(r'\d+', str_num)]
        words = ["zero", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "twenty", "fifty", "hundred"]
        pattern = r'\b(?:' + r'|'.join(map(re.escape, words)) + r')\b'
        word_nums = re.findall(pattern, str_num, re.IGNORECASE)
        for n in nums_present:
            clean_nums_present.append(int(n))
        for n in word_nums:
            clean_nums_present.append(string_num_to_int(n.lower()))
        if len(clean_nums_present) > 0:
            return round(sum(clean_nums_present)/len(clean_nums_present))
        elif "," in str_num:
            return str_num.count(",") + 1
        elif " " in str_num:
            return str_num.count(" ") + 1
        else:
            return 0

    def clean_complexity(complexity: str) -> int:
        return int(complexity)
    if df["Q2: How many ingredients would you expect this food item to contain?"].dtype == "int64":
        df["num_ingredients"] = df["Q2: How many ingredients would you expect this food item to contain?"]
    else:
        df["num_ingredients"] = df["Q2: How many ingredients would you expect this food item to contain?"].apply(clean_num_ingredients)
    df.drop(columns=["Q2: How many ingredients would you expect this food item to contain?"], inplace=True)

    df["complexity"] = df["Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"].apply(clean_complexity)
    df.drop(columns=["Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)"], inplace=True)

    # Add indicator variables to categorical variable time

    time_col = "Q3: In what setting would you expect this food to be served? Please check all that apply"

    # Get rid of NaNs
    df[time_col] = df[time_col].fillna('')

    # Hardcoded categories
    unique_categories = {"Week Day lunch", "At a party", "Late night snack", "Weekend lunch", "Week day dinner", "Weekend dinner"}

    # Create new indicators for each unique category
    for category in unique_categories:
        df[category] = df[time_col].str.contains(category, regex=False).astype(int)

    # Drop the original people column
    df.drop(columns=[time_col], inplace=True)

    # Turn drinks into BoW if we're using them
    column_name = "Q6: What drink would you pair with this food item?"
    if clean_type=="normal" or clean_type=="no_movies":
        global drink_vocab
        drink_vocab = json.load(open("drink_vocab.json", "r"))

        def clean_text(text):
            """
            Cleans text by removing punctuation, converting to lowercase,
            and tokenizing words while updating the global vocabulary.
            """
            text = str(text)  # Ensure input is a string
            text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])  # Keep letters and spaces
            text = text.lower().strip()  # Convert to lowercase and remove extra spaces
            words = text.split()  # Tokenize into words

            return words  # Return cleaned tokenized words


        # Apply text cleaning function only to the 6th column
        df[column_name] = df[column_name].apply(clean_text)

        new_column_names = [f"drink_{word}" for word in drink_vocab.keys()]

        # Construct Bag of Words representation only for the 6th column
        bag_of_word_indicators = pd.DataFrame([
            [1 if word in words else 0 for word in drink_vocab]  # Encode word presence
            for words in df[column_name]
        ], columns=new_column_names)

        # Merge BoW indicators with the original DataFrame (keeping all columns intact)
        df = pd.concat([df, bag_of_word_indicators], axis=1)

    # Drop the original text column
    df.drop(columns=[column_name], inplace=True)

    # Add indicator variables to categorical variable hot sauce

    hot_sauce_col = "Q8: How much hot sauce would you add to this food item?"

    # Get rid of NaNs
    df[hot_sauce_col] = df[hot_sauce_col].fillna('None')

    # Hardcoded categories
    unique_categories = {"mild", "medium", "hot", "None"}

    # Create new indicators for each unique category
    for category in unique_categories:
        df[category] = df[hot_sauce_col].str.contains(category, regex=False).astype(int)

    # Drop the original people column
    df.drop(columns=[hot_sauce_col], inplace=True)
    return df, dataframe["Label"]


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data_combined_modified.csv")
    X, T = clean_data(df)
    with open("data/clean_data_no_movies.csv", "w", encoding="utf-8") as f:
        X.to_csv(f, index=False)
    print(X)
    print(T)
