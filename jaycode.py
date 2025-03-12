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


def clean_num_ingredients(num_ingredients: str) -> int:
    x = 0
    i = 1
    nums_present = []
    clean_nums_present = []
    while x < len(num_ingredients):
        while i <= len(num_ingredients):
            if is_number(num_ingredients[x:i]):
                found_num = i
                i += 1
                while is_number(num_ingredients[x:i]) and i <= len(num_ingredients):
                    found_num = i
                    i += 1
                nums_present.append(num_ingredients[x:found_num])
                x = found_num
                i = found_num
            i += 1
        x += 1
        i = x + 1

    for num in nums_present:
        if is_digit(num[0]):
            clean_nums_present.append(int(num))
        else:
            clean_nums_present.append(string_num_to_int(num))
    if len(clean_nums_present) > 0:
        return round(sum(clean_nums_present)/len(clean_nums_present))
    elif "," in num_ingredients:
        return num_ingredients.count(",") + 1
    elif " " in num_ingredients:
        return num_ingredients.count(" ") + 1
    else:
        return 0


def clean_complexity(complexity: str) -> int:
    return int(complexity)


if __name__ == "__main__":
    str = "100 or 200"
    print(clean_num_ingredients(str))
