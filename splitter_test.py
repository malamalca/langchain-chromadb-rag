from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

length_function = len

# The default list of split characters is [\n\n, \n, " ", ""]
# Tries to split on them in order until the chunks are small enough
# Keep paragraphs, sentences, words together as long as possible
splitter = RecursiveCharacterTextSplitter(
    #separators=["(.*). člen\n", "\d", "\n\n", "\n", " ", ""],
    separators=[r".*. .*len\n"],
    is_separator_regex=True,
    chunk_size=200, 
    chunk_overlap=100,
    length_function=length_function,
)
#text = requests.get("https://pisrs.si/api/datoteke/integracije/363508837").text

with open("gz-1.txt","r",encoding="utf-8") as f:
    text = f.read()

splits = splitter.split_text(text)
max_len = max([length_function(s) for s in splits])

for i, split in enumerate(splits):
    print(f"--- Split {i+1} ---")
    print(split)
    print()

print(f"Number of splits: {len(splits)}")
print(f"Max split length: {max_len}")