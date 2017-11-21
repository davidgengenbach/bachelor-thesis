import re

TAG_REGEXP = r"<meta property=\"og:video:tag\" content=\"(.+?)\" \/\>"
def get_tags_from_html(html):
    matches = re.findall(TAG_REGEXP, html)
    if not len(matches):
        return None
    tags = [x.lower().strip() for x in matches]
    return tags