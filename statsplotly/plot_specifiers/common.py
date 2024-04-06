import re

from statsplotly import constants


def smart_title(title_string: str) -> str:
    """Split string at _, capitalizes words, and joins with space."""
    title_string = title_string.strip()
    if len(title_string) == 0:
        return title_string
    return " ".join(
        [
            (
                "".join([word[0].upper(), word[1:]])
                if (len(word) >= constants.MIN_CAPITALIZE_LENGTH)
                and not (any(letter.isupper() for letter in word))
                else word
            )
            for word in re.split(" |_", title_string)
        ]
    )


def smart_legend(legend_string: str) -> str:
    """Cleans and capitalizes axis legends for figure."""
    legend_string = legend_string.strip()
    if len(legend_string) == 0:
        return legend_string
    return " ".join(
        [
            "".join([w[0].upper(), w[1:]]) if i == 0 else w
            for i, w in enumerate(re.split("_", legend_string))
        ]
    )