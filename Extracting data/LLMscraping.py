import asyncio
from crawl4ai import AsyncWebCrawler

# Import the necessary classes
from DifferentOdds import BettingTabsExtractor
from WinOrLose import OddsExtractor
from OverUnder import OverUnderOddsParser
from DoubleChance import DoubleChanceOddsParser
from BothTeamToScore import OddsParser
from AsianHandicap import AsianHandicapOddsParser
from DrawNoBet import DrawNoBetOddsParser
from MatchDetails import FixtureInfoExtractor

class Crawler:
    def __init__(self, urls):
        self.urls = urls
        self.proxy_config = {
            "server": "http://dc.oxylabs.io:8000",
            "username": "##",
            "password": "##"
        }
        self.semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks
        self.fail_count = 0  # Counter for failed fetch attempts
        self.failed_urls = "Redo.txt"  # File to store failed URLs

    def log_failed_url(self, url: str, stage: str):
        """Append a failed URL and its stage to the Redo.txt file."""
        try:
            with open(self.failed_urls, 'a') as file:
                file.write(f"('{url}', '{stage}')\n")
        except Exception as e:
            print(f"Failed to log URL to {self.failed_urls}: {e}")

    async def fetch_html(self, url: str, stage: str) -> str:
        """Fetch HTML content from a URL and log failures with the stage."""
        try:
            async with AsyncWebCrawler(verbose=True, proxy_config=self.proxy_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    screenshot=False,
                    bypass_cache=True,
                    magic=True,
                    delay_before_return_html=2.0
                )
                if result.success and hasattr(result, 'html') and result.html:
                    return result.html
                else:
                    print(f"Failed to fetch HTML for {url}")
                    self.log_failed_url(url, stage)
                    self.fail_count += 1
                    return ""
        except Exception as e:
            print(f"An error occurred while fetching {url}: {e}")
            self.log_failed_url(url, stage)
            self.fail_count += 1
            return ""

    def map_tab_to_fragment(self, tab_title: str) -> str:
        mapping = {
            "Draw No Bet": "#ha",
            "1X2": "#1x2",
            "Over/Under": "#ou",
            "Asian Handicap": "#ah",
            "Both Teams To Score": "#bts",
            "Double Chance": "#dc"
        }
        return mapping.get(tab_title, "")

    def parse_tab_data(self, tab_title: str, html_content: str):
        if tab_title == "1X2":
            return OddsExtractor.parse(html_content)
        elif tab_title == "Over/Under":
            return OverUnderOddsParser.parse(html_content)
        elif tab_title == "Double Chance":
            return DoubleChanceOddsParser.parse(html_content)
        elif tab_title == "Both Teams To Score":
            return OddsParser.parse(html_content)
        elif tab_title == "Asian Handicap":
            return AsianHandicapOddsParser.parse(html_content)
        elif tab_title == "Draw No Bet":
            return DrawNoBetOddsParser.parse(html_content)
        else:
            return ""

    def extract_filename_from_url(self, url: str) -> str:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path = parsed_url.path
        path_parts = path.strip('/').split('/')
        if len(path_parts) < 3:
            return None
        sport = path_parts[0]
        country = path_parts[1]
        league_season = path_parts[2]
        filename = f"{sport}-{country}-{league_season}.txt"
        return filename

    async def process_url(self, url: str, stage: str):
        async with self.semaphore:
            try:
                base_filename = self.extract_filename_from_url(url)
                if not base_filename:
                    print(f"Could not extract filename from URL: {url}")
                    return
                if base_filename.endswith(".txt"):
                    filename_base, extension = base_filename.rsplit('.', 1)
                    filename = f"{filename_base}-{stage}.{extension}"
                else:
                    filename = f"{base_filename}-{stage}"
                base_html = await self.fetch_html(url, stage)
                if not base_html:
                    return

                fixture_info = FixtureInfoExtractor.parse(base_html)
                tab_titles = BettingTabsExtractor.extract_tab_titles(base_html)
                if not tab_titles:
                    print(f"No betting tabs found for {url}.")
                    return

                tab_urls = [
                    url + self.map_tab_to_fragment(tab_title) for tab_title in tab_titles
                ]
                bookmaker_outputs = {}
                tab_failures = []  # Track failed tabs

                for tab_url, tab_title in zip(tab_urls, tab_titles):
                    if not tab_url.endswith("#"):
                        tab_html = await self.fetch_html(tab_url, stage=f"Tab: {tab_title}")
                        if not tab_html:
                            tab_failures.append(tab_title)
                            continue
                        parsed_string = self.parse_tab_data(tab_title, tab_html)
                        if not parsed_string:
                            continue
                        bookmaker_sections = parsed_string.strip().split('\n\n')
                        for section in bookmaker_sections:
                            lines = section.strip().split('\n')
                            if lines:
                                bookmaker_line = lines[0]
                                bookmaker_name = bookmaker_line.replace("Bookmaker: ", "").strip()
                                if bookmaker_name not in bookmaker_outputs:
                                    bookmaker_outputs[bookmaker_name] = []
                                bookmaker_outputs[bookmaker_name].extend(lines[1:])

                combined_data = [fixture_info]
                for bookmaker, outputs in bookmaker_outputs.items():
                    combined_data.append(f"Bookmaker: {bookmaker}\n")
                    combined_data.extend(outputs)
                    combined_data.append('\n')

                with open(filename, 'a', encoding='utf-8') as output_file:
                    output_file.write('\n')
                    output_file.write('\n'.join(combined_data))
                    output_file.write('\n#\n')
                print(f"Results appended to {filename}")

                if tab_failures:
                    print(f"Base URL {url} processed successfully but failed tabs: {', '.join(tab_failures)}")
            except Exception as e:
                print(f"An error occurred while processing {url}: {e}")
                self.log_failed_url(url, stage)
                self.fail_count += 1

    async def run(self):
        tasks = [self.process_url(url, stage) for stage, url in self.urls]
        await asyncio.gather(*tasks)
        print(f"Total fetch failures: {self.fail_count}")

# Example usage:
# if __name__ == "__main__":
#     example_urls = [
#         ("Main", "https://www.betexplorer.com/football/england/premier-league-2022-2023/arsenal-wolves/M1w8YmqE/"),
#         ("Qualification", "https://www.betexplorer.com/football/england/premier-league-2022-2023/chelsea-leicester/GveqIO4A/")
#     ]
#     crawler = Crawler(example_urls)
#     asyncio.run(crawler.run())
