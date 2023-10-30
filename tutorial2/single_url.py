from langchain.document_loaders import youtube
import io

loader=youtube.YoutubeLoader.from_youtube_url("https://youtu.be/64eXtnwLj5U")
docs=loader.load()
print(docs)
with io.open("transcript.txt","w",encoding="utf-8")as f1:
    for doc in docs:
        f1.write(doc.page_content)
    f1.close()