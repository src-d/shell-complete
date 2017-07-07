from scrapy.spiders import Spider
from shell_scraper.items import ShellScraperItem
from scrapy.http import Request
import just

class MySpider(Spider):
    name = "github_spider"
    allowed_domains = ["github.com"]
    with open("repos.txt") as f:
        start_urls = [url.strip() for url in f.readlines()]

    def parse(self, response):
        base = "https://raw.githubusercontent.com"
        just.write(response.text,  "data" + response.url[len(base):])
