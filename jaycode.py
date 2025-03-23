import re


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


if __name__ == "__main__":
    stdr = "none, I hate you"
    print(clean_num_ingredients(stdr))
